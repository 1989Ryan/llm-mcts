base_port=8189

base_port=$((base_port+2))
export PYTHONPATH="$PWD"
python3 learned_policy/train.py \
--gpus 7 \
--language_model_type_pretrain 'fine_tune_pretrain' \
--max_episode_length 70 \
--num_mini_batch 32 \
--model_type 'gpt2' \
--model_name_or_path 'gpt2' \
--seed 0 \
--base-port 8191 \
--eval \
--subset InDistributation \
--test_examples 100 \
--interactive_eval \
--interactive_eval_path interactive_eval/InDistribution/seed0 \
