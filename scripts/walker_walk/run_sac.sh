# for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
for seed in 12345 23451 34512 45123 51234; do
    CUDA_VISIBLE_DEVICES=0 python train_SAC.py log_save_tb=false env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=1000000
done