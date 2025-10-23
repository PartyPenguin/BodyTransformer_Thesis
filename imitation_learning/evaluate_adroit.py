import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
from absl import app, flags
import minari
from tqdm import tqdm

from models.networks import BodyNet, Transformer, BodyTransformer, GNNModule

FLAGS = flags.FLAGS

# Keep flags that define the model architecture
flags.DEFINE_string('env', 'D4RL/relocate/expert-v2', 'Environment name, can be [door-expert-v2, hammer-expert-v2, relocate-expert-v2]')
flags.DEFINE_string('network_type', 'graph', 'Type of network to use, can be [mlp, transformer, soft_bias_transformer, body_transformer]')
flags.DEFINE_integer('nlayers', 16, 'Number of transformer layers')
flags.DEFINE_integer('embedding_dim', 64, 'Dimension of the embeddings')
flags.DEFINE_integer('dim_feedforward', 1024, 'Dimension of the feedforward network layer')
flags.DEFINE_integer('nheads', 5, 'Number of heads in the transformer layer')
flags.DEFINE_string('use_positional_encoding', 'False', 'Whether to use positional encoding')
flags.DEFINE_string('is_mixed', "False", 'Whether to interleave masked and unmasked attention layers')
flags.DEFINE_integer('first_hard_layer', 0, 'Index of the first masked layer for the bot-mix')

# Add flags for evaluation
flags.DEFINE_string('checkpoint_path', None, 'Path to the model checkpoint.')
flags.DEFINE_integer('eval_episodes', 200, 'Number of episodes to run for evaluation.')
flags.DEFINE_integer('seed', 100, 'Random seed.')

flags.mark_flag_as_required('checkpoint_path')


def evaluate(net, env, device, mean_inputs, std_inputs, eval_episodes):
    net.eval()
    
    returns = []
    successes = []
    
    with tqdm(total=eval_episodes, desc="Evaluating episodes") as pbar:
        for ep in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            ret = 0
            success = 0
            
            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(device)
                
                # Normalize observation
                obs_tensor = (obs_tensor - mean_inputs.reshape(-1)) / (std_inputs.reshape(-1) + 1e-8)
                
                action = net.mode(obs_tensor.unsqueeze(0)).squeeze().detach().cpu().numpy()
                action = np.clip(action, -1, 1)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                ret += reward
                done = terminated or truncated
                obs = next_obs
                
                if done:
                    success = info.get('success', 0)

            returns.append(ret)
            successes.append(success)
            pbar.update(1)

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes)

    print("\nEvaluation Results:")
    print(f"Mean return: {mean_return:.2f}")
    print(f"Std return: {std_return:.2f}")
    print(f"Success rate: {success_rate:.2f}")

    return mean_return, std_return, success_rate


def main(_):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    dataset = minari.load_dataset(FLAGS.env, download=True)
    env = dataset.recover_environment()
    
    sample = dataset.sample_episodes(n_episodes=1)
    action_dim = sample[0].actions.shape[-1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embedding_dim = FLAGS.embedding_dim
    nheads = FLAGS.nheads
    dim_feedforward = FLAGS.dim_feedforward
    nbodies = 25
    
    is_mixed = False if FLAGS.is_mixed == "False" else True
    use_positional_encoding = False if FLAGS.use_positional_encoding == "False" else True

    if FLAGS.network_type == 'body_transformer':
        embedding_dim *= nheads
        net = BodyTransformer(nbodies, FLAGS.env, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, num_layers=FLAGS.nlayers, is_mixed=is_mixed, use_positional_encoding=use_positional_encoding, first_hard_layer=FLAGS.first_hard_layer)
    elif FLAGS.network_type == 'graph':
        net = GNNModule(input_dim=embedding_dim, hidden_dim=dim_feedforward, heads=nheads, num_layers=FLAGS.nlayers)
    
    model = BodyNet(FLAGS.env, net, action_dim=action_dim, embedding_dim=embedding_dim, global_input=FLAGS.network_type=='mlp', device=device)
    
    checkpoint = torch.load(FLAGS.checkpoint_path, map_location=device)
    
    # Check if the checkpoint contains the expected keys
    if 'model_state_dict' not in checkpoint:
        print("Error: Checkpoint does not contain 'model_state_dict'")
        return
    
    if 'mean_inputs' not in checkpoint or 'std_inputs' not in checkpoint:
        print("Error: Checkpoint does not contain normalization statistics")
        return
    
    # Manually filter state dict to handle size mismatches
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    filtered_state_dict = {}
    mismatched_keys = []
    
    for k, v in checkpoint_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
        else:
            mismatched_keys.append(k)
    
    if mismatched_keys:
        print(f"Warning: {len(mismatched_keys)} keys had size mismatches and were skipped")
        # Only show first few mismatched keys to avoid spam
        if len(mismatched_keys) <= 10:
            for k in mismatched_keys:
                checkpoint_shape = checkpoint_state_dict[k].shape if k in checkpoint_state_dict else "N/A"
                model_shape = model_state_dict.get(k, "N/A")
                print(f"  {k}: checkpoint {checkpoint_shape}, model {model_shape}")
        else:
            print(f"  First 5 mismatches:")
            for k in mismatched_keys[:5]:
                checkpoint_shape = checkpoint_state_dict[k].shape if k in checkpoint_state_dict else "N/A"
                model_shape = model_state_dict.get(k, "N/A") 
                print(f"    {k}: checkpoint {checkpoint_shape}, model {model_shape}")
            print(f"  ... and {len(mismatched_keys) - 5} more")

    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    
    # Report loading statistics
    total_checkpoint_keys = len(checkpoint_state_dict)
    loaded_keys = len(filtered_state_dict)
    missing_keys = len(load_result.missing_keys)
    
    print(f"\nModel loading summary:")
    print(f"  Total keys in checkpoint: {total_checkpoint_keys}")
    print(f"  Successfully loaded keys: {loaded_keys}")
    print(f"  Missing keys: {missing_keys}")
    
    if missing_keys > 0 and missing_keys <= 10:
        print(f"  Missing keys: {load_result.missing_keys}")
    elif missing_keys > 10:
        print(f"  First 5 missing keys: {load_result.missing_keys[:5]}")
        print(f"  ... and {missing_keys - 5} more missing keys")

    mean_inputs = checkpoint['mean_inputs'].to(device)
    std_inputs = checkpoint['std_inputs'].to(device)
    print(f"\nLoaded checkpoint from {FLAGS.checkpoint_path}")

    model.to(device)
    
    evaluate(model, env, device, mean_inputs, std_inputs, FLAGS.eval_episodes)


if __name__ == '__main__':
    app.run(main)
