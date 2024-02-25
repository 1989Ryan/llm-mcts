import argparse

import torch
import pdb


def get_args():
    parser = argparse.ArgumentParser(description='Pre-Trained Language Models for Interactive Decision-Making')

    ## Exec
    parser.add_argument('--exec_file', type=str, default='./vh/vh_sim/simulation/unity_simulator/v2.2.5/linux_exec.v2.2.5_beta.x86_64')
    parser.add_argument('--base-port', type=int, default=8080)
    parser.add_argument('--graphics', action='store_true', default=False)
    parser.add_argument('--display', type=str, default="0")
    parser.add_argument('--use-editor', action='store_true', default=False, help='whether to use an editor or executable')
    

    ## Env
    parser.add_argument('--obs_type', type=str, default='partial', help='partial | full')
    parser.add_argument('--n_agent', type=int, default=1)
    parser.add_argument('--env_id', type=int, default=1)
    parser.add_argument('--max_episode_length', type=int, default=100, help='max_episode_length')
        
    ## Data
    parser.add_argument('--data_dir', type=str, default='./data')

    ## Model
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument("--model_type", default='gpt2', type=str, help='pretrained model name')
    parser.add_argument("--model_name_or_path", default='gpt2', type=str, help="Path to pre-trained model")
    parser.add_argument('--language_model_type_pretrain', type=str, default='fine_tune_pretrain')

    ## Train
    parser.add_argument('--train_epoch', type=int, default=500)
    parser.add_argument('--num_mini_batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-5, help='goal model learning rate (default: 7e-4)')
    
    ## Test
    parser.add_argument('--test_examples', type=int, default=100)
    parser.add_argument('--eval', action='store_true', default=False, help='eval')
    parser.add_argument('--interactive_eval', action='store_true', default=False, help='interactive_eval')
    parser.add_argument('--interactive_eval_path', type=str, default='')
    parser.add_argument('--subset', type=str, default='InDistributation')
    
    ## Checkpoint
    parser.add_argument('--save_dir', default='./vh/trained_model/2/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--pretrained_model_dir', default='')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use during training')
    parser.add_argument('--debug', type=int, default=0)
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args







    
