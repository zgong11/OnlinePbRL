for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    CUDA_VISIBLE_DEVICES=1 python train_SAC.py log_save_tb=false env=cheetah_run seed=$seed agent.params.actor_lr=0.001 agent.params.critic_lr=0.001 agent.params.alpha_lr=0.001 num_train_steps=1000000
done