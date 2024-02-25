export PYTHONPATH="$PWD"


# nohup python vh/data_gene/gen_data/vh_init.py --port "8093" --task all --mode simple --usage train --num-per-apartment 50 > log/8093.log 2>&1 &

# Generate Goal data

python vh/data_gene/gen_data/vh_init.py --port "8083" --task all --mode simple --usage train --num-per-apartment 500 
python vh/data_gene/testing_agents/gene_data.py --mode simple \
    --dataset_path ./vh/dataset/env_task_set_500_full.pik\
    --base-port 8104 
python vh/data_gene/gen_data/vh_init.py --port "8083" --task all --mode full --usage train --num-per-apartment 500 
python vh/data_gene/testing_agents/gene_data.py --mode full \
    --dataset_path ./vh/dataset/env_task_set_500_full.pik\
    --base-port 8104 

# nohup python vh/data_gene/gen_data/vh_init.py   --port "8095" --task all --mode simple --usage test --num-per-apartment 50 > log/8095.log 2>&1 &

# nohup python vh/data_gene/gen_data/vh_init.py   --port "8096" --task all --mode full --usage test --num-per-apartment 50 > log/8096.log 2>&1 &

# nohup python vh/data_gene/gen_data/vh_init.py   --port "8097" --task all --mode simple --unseen-apartment --usage test  --num-per-apartment 100 > log/8097.log 2>&1 &
# nohup python vh/data_gene/gen_data/vh_init.py  --port "8098" --task all --mode full --unseen-apartment --usage test  --num-per-apartment 100 > log/8098.log 2>&1 &

# # # sleep 5
# nohup python vh/data_gene/gen_data/vh_init.py   --port "8100" --task unseen_comp --mode full --usage test --num-per-apartment 50 > log/8100.log 2>&1 &

# nohup python vh/data_gene/gen_data/vh_init.py   --port "8100" --task all --mode full --unseen-item --usage test --num-per-apartment 50 > log/8100.log 2>&1 &
# nohup python vh/data_gene/gen_data/vh_init.py   --port "8101" --task all --mode simple --unseen-item --usage test  --num-per-apartment 50 > log/8101.log 2>&1 &
# nohup python vh/data_gene/gen_data/vh_init.py   --port "8102" --task unseen_comp --mode full --unseen-item  --usage test --num-per-apartment 50 > log/8102.log 2>&1 &


# Generate expert data


# python vh/data_gene/gen_data/vh_init.py --port "8093" --task all --mode simple --usage train --num-per-apartment 500 
# python vh/data_gene/testing_agents/gene_data.py --mode simple \
#     --dataset_path ./vh/dataset/env_task_set_500_simple.pik\
#     --base-port 8103 

# nohup python vh/data_gene/testing_agents/gene_data.py --mode simple \
#     --dataset_path ./vh/dataset/env_task_set_50_simple.pik\
#     --base-port 8103 > log/expert_data_8103.log 2>&1 &
# nohup python vh/data_gene/testing_agents/gene_data.py --mode full \
#     --dataset_path ./vh/dataset/env_task_set_50_full.pik\
#     --base-port 8104  > log/expert_data_8104.log 2>&1 &
# python vh/data_gene/testing_agents/gene_data.py --mode simple \
#     --dataset_path ./vh/dataset/env_task_set_50_simple.pik\
#     --base-port 8103 --task put_fridge

# python vh/data_gene/testing_agents/gene_data.py --mode full \
#     --dataset_path ./vh/dataset/env_task_set_50_full.pik\

python vh/data_gene/gen_data/vh_init.py   --port "8095" --task all --mode simple --usage test --num-per-apartment 50

python vh/data_gene/gen_data/vh_init.py   --port "8096" --task all --mode full --usage test --num-per-apartment 50 

python vh/data_gene/gen_data/vh_init.py   --port "8097" --task all --mode simple --unseen-apartment --usage test  --num-per-apartment 50 
python vh/data_gene/gen_data/vh_init.py  --port "8098" --task all --mode full --unseen-apartment --usage test  --num-per-apartment 50 

python vh/data_gene/gen_data/vh_init.py   --port "8099" --task unseen_comp --mode full --usage test --num-per-apartment 50 

python vh/data_gene/gen_data/vh_init.py   --port "8100" --task all --mode full --unseen-item --usage test --num-per-apartment 50 
python vh/data_gene/gen_data/vh_init.py   --port "8101" --task all --mode simple --unseen-item --usage test  --num-per-apartment 50 
python vh/data_gene/gen_data/vh_init.py   --port "8102" --task unseen_comp --mode full --unseen-item  --usage test --num-per-apartment 50 