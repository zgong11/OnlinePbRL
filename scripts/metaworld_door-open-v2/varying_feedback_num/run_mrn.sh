# for max_feedback in 100 400 1000 2000; do
for max_feedback in 100 400 2000; do
    echo $max_feedback
    reward_batch=$(( (max_feedback / 10) < 50 ? (max_feedback / 10) : 50 ))
    echo $reward_batch
    # for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
    for seed in 12345 23451 34512 45123 51234; do
        CUDA_VISIBLE_DEVICES=4 python train_MRN.py log_save_tb=false env=metaworld_door-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_unsup_steps=9000 num_train_steps=1000000 num_interact=5000 max_feedback=$max_feedback reward_batch=$reward_batch reward_update=10 feed_type=1 num_meta_steps=10000
    done
done