# for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
for seed in 12345 23451 34512 45123 51234; do
    CUDA_VISIBLE_DEVICES=4 python train_MRN.py log_save_tb=false env=quadruped_walk seed=$seed agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 num_unsup_steps=9000 num_train_steps=1000000 num_interact=30000 max_feedback=1000 reward_batch=100 reward_update=50 feed_type=1 num_meta_steps=3000
done