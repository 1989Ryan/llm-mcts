for seed in 0; do
for subset in InDistributation; do

base_port=8189

base_port=$((base_port+2))

python3 learned_policy/inference.py \
--gpus 0 \
--language_model_type_pretrain 'fine_tune_pretrain' \
--max_episode_length 70 \
--num_mini_batch 1 \
--model_type 'gpt2' \
--model_name_or_path 'gpt2' \
--seed 0 \
--base-port 8191 \
--eval \
--subset InDistributation \
--test_examples 100 \
--interactive_eval \
--interactive_eval_path interactive_eval/InDistribution/seed0 \
--pretrained_model_dir learned_policy/checkpoint/saved_model_latest.p
# --pretrained_model_dir learned_policy/checkpoint/model.pt

done
done