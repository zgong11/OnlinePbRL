# for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
for seed in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train_SAC.py log_save_tb=false env=metaworld_window-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_train_steps=1000000
done