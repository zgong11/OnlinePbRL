# for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
for seed in 12345 23451 34512 45123 51234; do
    CUDA_VISIBLE_DEVICES=3 python train_PEBBLE_semi_dataaug.py log_save_tb=false env=cheetah_run seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=1000 inv_label_ratio=100 feed_type=1 threshold_u=0.999 mu=4
done