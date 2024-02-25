import os
import sys
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from arguments import get_args
import init_path
from utils_bc import utils

from bc_agent import BC_Agent
from models.bc_model import BC_MODEL

from utils_bc import utils_interactive_eval
from utils_bc.utils import save_model, load_pretrained_model
from utils_bc.utils_llm import get_pretrained_tokenizer
from interactive_interface import interactive_interface_fn
from data_loader import UnityGraphDataset


def get_logger(args, log_path):
    if os.path.exists(log_path):
        os.remove(log_path)

    import logging
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.INFO)

    output_file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler(sys.stdout)

    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    logging = a_logger
    return logging


def main():
    args = get_args()
    main_single(0, args)


def main_single(gpu, args):
    random.seed(args.seed + gpu)
    np.random.seed(args.seed + gpu)
    torch.manual_seed(args.seed + gpu)
    torch.cuda.manual_seed_all(args.seed + gpu)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


    if not args.eval:
        log_dir = os.path.expanduser('/'.join(args.save_dir.split('/')[:-1]))
        utils.cleanup_log_dir(log_dir)

        ## tensorboard
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        # writer = SummaryWriter(log_dir=tensorboard_dir)
        

    args = init_path.get_logger_path(args)

    logging = get_logger(args, args.log_path)
    
    torch.cuda.set_device(gpu)
    

    
    ## initial path
    args = init_path.initialize_path(args)
    args = init_path.load_data_info(args)

    # for ele in args.env_task_set:
    #     print(ele['task_goal'])
    # vh_envs = utils_interactive_eval.connect_env(args, logging)
    ## Model
    model = BC_MODEL(args)

    if not args.eval:
        model = model.cuda()
    else:
        model = model.cuda()


    action_criterion = nn.CrossEntropyLoss()
    obj_criterion = nn.CrossEntropyLoss()


    ## Agent
    agent = BC_Agent(
        args,
        model,
        action_criterion,
        obj_criterion,
        logging,
        gpu
    )

    ## load pretrained model
    # agent, best_top1, start_epoch = load_pretrained_model(args, agent, gpu, logging)

    
    ## Training
    
    tokenizer = get_pretrained_tokenizer(model_type=args.model_type, model_name_or_path=args.model_name_or_path)
    expert_data_path = "./expert_actions/expert_5_full/"
    list_files = os.listdir(expert_data_path)
    expert_data_path_2 = "./expert_actions/expert_5_simple/"
    list_files_2 = os.listdir(expert_data_path_2)
    trainset = UnityGraphDataset(args, args.data_info, expert_data_path, list_files, tokenizer)
    trainset_2 = UnityGraphDataset(args, args.data_info, expert_data_path_2, list_files_2, tokenizer)
    trainset_combine = torch.utils.data.ConcatDataset([trainset, trainset_2])
    trainloader = data.DataLoader(trainset_combine, batch_size=args.num_mini_batch, shuffle=True, num_workers=8)
    agent.run(trainloader, epoch=1000, mode='train')

    


if __name__ == "__main__":
    main()

