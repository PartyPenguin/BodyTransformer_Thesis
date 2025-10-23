python evaluate_adroit.py \
    --checkpoint_path /home/nicolas/Documents/development/python/BodyTransformerOld/imitation_learning/checkpoints/env_D4RL_hammer_expert-v2_model_graph_seed_42_nheads_4_embedding_dim_64_dim_feedforward_256_nlayers_3_epoch_80.pth \
    --env D4RL/hammer/expert-v2 \
    --network_type graph \
    --eval_episodes 200 \
    --nlayers 3 \
    --embedding_dim 64 \
    --dim_feedforward 256 \
    --nheads 4 \
    --seed 423

python evaluate_adroit.py \
    --checkpoint_path /home/nicolas/Documents/development/python/BodyTransformerOld/imitation_learning/checkpoints/env_D4RL_hammer_expert-v2_model_graph_seed_25_nheads_4_embedding_dim_64_dim_feedforward_256_nlayers_3_epoch_80.pth \
    --env D4RL/hammer/expert-v2 \
    --network_type graph \
    --eval_episodes 200 \
    --nlayers 3 \
    --embedding_dim 64 \
    --dim_feedforward 256 \
    --nheads 4 \
    --seed 254

python evaluate_adroit.py \
    --checkpoint_path /home/nicolas/Documents/development/python/BodyTransformerOld/imitation_learning/checkpoints/env_D4RL_hammer_expert-v2_model_graph_seed_0_nheads_4_embedding_dim_64_dim_feedforward_256_nlayers_3_epoch_120.pth \
    --env D4RL/hammer/expert-v2 \
    --network_type graph \
    --eval_episodes 200 \
    --nlayers 3 \
    --embedding_dim 64 \
    --dim_feedforward 256 \
    --nheads 4 \
    --seed 34
