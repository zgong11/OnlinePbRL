# for max_feedback in 100 400 1000 2000; do
for max_feedback in 400 1000 2000; do
    echo $max_feedback
    reward_batch=$(( (max_feedback / 10) < 50 ? (max_feedback / 10) : 50 ))
    echo $reward_batch
    # for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 68790; do
    for seed in 12345 23451 34512 45123 51234; do
        CUDA_VISIBLE_DEVICES=1 python train_PEBBLE.py log_save_tb=false env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=$max_feedback reward_batch=$reward_batch reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
    done
done