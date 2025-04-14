for max_feedback in 100 400 1000 2000; do
    echo $max_feedback
    for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
        CUDA_VISIBLE_DEVICES=3 python train_PEBBLE_semi_dataaug.py log_save_tb=false env=metaworld_door-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 agent.params.alpha_lr=0.0003 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_unsup_steps=9000 num_train_steps=1000000 num_interact=5000 max_feedback=$max_feedback reward_batch=10 reward_update=50 inv_label_ratio=10 feed_type=1 threshold_u=0.99 mu=4
    done
done