# for max_feedback in 100 400 1000 2000; do
#     echo $max_feedback
#     for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
#         CUDA_VISIBLE_DEVICES=0 python train_MRN.py log_save_tb=false env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 agent.params.alpha_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=$max_feedback reward_batch=10 reward_update=50 feed_type=1 num_meta_steps=1000
#     done
# done

CUDA_VISIBLE_DEVICES=0 python train_MRN.py log_save_tb=false env=walker_walk seed=6789 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 agent.params.alpha_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 num_meta_steps=1000