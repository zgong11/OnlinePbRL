# for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
for seed in 12345 23451 34512 45123 51234; do
    CUDA_VISIBLE_DEVICES=5 python train_QPA.py log_save_tb=false env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=1000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 max_reward_buffer_size=10
done