import os
import sys
import pdb
import pickle
import random

def initialize_path(args):
    if args.debug:
        args.test_examples = 1
        
    if args.model_type=='gpt2':
        args.data_info = pickle.load(open(os.path.join(args.data_dir, 'data_info.p'), "rb"))

    ## initial scene graph
    if args.interactive_eval:
        args.model_exploration_p = 1
        args.env_data_dir = './dataset/env_task_set_50_simple.pik' 
        print('loading inital scene graph from %s' % args.env_data_dir)
        env_task_set = pickle.load(open(args.env_data_dir, 'rb'))
        # args.env_data_dir = './dataset/env_task_set_50_full.pik' 
        # print('loading inital scene graph from %s' % args.env_data_dir)
        # env_task_set_full = pickle.load(open(args.env_data_dir, 'rb'))
        
        # print(len(env_task_set))
        env_task_set = [env for env in env_task_set]
        # env_task_set += [env for env in env_task_set_full]
        # print(len(env_task_set))
        # print(env_task_set[0])
        args.env_task_set = env_task_set

    return args


def get_logger_path(args):
    if len(args.save_dir)==0:
        log_path = 'checkpoint/debug_%s.log' % args.base_port
    else:
        log_path = '%s/log.log' % '/'.join(args.save_dir.split('/')[:-1])

    if args.interactive_eval and args.eval:
        interactive_eval_dir = os.path.join('/'.join(args.pretrained_model_dir.split('/')[:-1]), args.interactive_eval_path)
        os.makedirs(interactive_eval_dir, exist_ok=True)
        log_path = '%s/log_interactive_eval.log' % (interactive_eval_dir)
    
    if args.debug:
        log_path = 'checkpoint/debug_%s.log' % args.base_port

    args.log_path = log_path
    return args



def load_data_info(args):
    data_info = args.data_info
    
    args.graph_node_class_names = data_info['graph_node_class_names']
    args.vocabulary_node_class_name_word_index_dict = data_info['vocabulary_node_class_name_word_index_dict']
    args.vocabulary_node_class_name_index_word_dict = data_info['vocabulary_node_class_name_index_word_dict']

    args.graph_node_states = data_info['graph_node_states']
    args.vocabulary_node_state_word_index_dict = data_info['vocabulary_node_state_word_index_dict']
    args.vocabulary_node_state_index_word_dict = data_info['vocabulary_node_state_index_word_dict']

    args.action_names = data_info['action_names']
    args.vocabulary_action_name_word_index_dict = data_info['vocabulary_action_name_word_index_dict']
    args.vocabulary_action_name_index_word_dict = data_info['vocabulary_action_name_index_word_dict']

    args.max_trajectory_len = data_info['max_steps']
    data_info["max_node_length"] = 200
    args.max_node_length = data_info['max_node_length']
    
    args.data_info = data_info
    
    return args








