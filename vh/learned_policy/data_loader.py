import os
import pdb
import re
import glob
import pickle
import random
import numpy as np

from torch.utils.data import Dataset

import sys
curr_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(curr_file_path))

from utils_bc.utils_llm import get_pretrained_tokenizer
from utils_bc.utils_data_process import *
from utils_bc.utils_graph import state_one_hot, filter_redundant_nodes
# from utils_bc.utils_interactive_eval import *


def get_observation_input(args, data_info, input_obs, agent_id):
    ## ----------------------------------------------------------------------------
    ## node name
    ## ----------------------------------------------------------------------------
    input_obs_node_gpt2_token = [data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding'][node['class_name']] 
                                #  for node in input_obs["nodes"] if node['class_name'] in data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding']]
                                 for node in input_obs if node['class_name'] in data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding']]
    input_obs_node_gpt2_token_mask = [data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding_mask'][node['class_name']] 
                                    #   for node in input_obs["nodes"] if node['class_name'] in data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding']]
                                      for node in input_obs if node['class_name'] in data_info['vocabulary_node_class_name_word_index_dict_gpt2_padding']]

    input_obs_node_gpt2_token = np.stack(input_obs_node_gpt2_token)
    input_obs_node_gpt2_token_mask = np.stack(input_obs_node_gpt2_token_mask)
    input_obs_node_gpt2_token_padding = np.zeros([data_info['max_node_length']-len(input_obs_node_gpt2_token), data_info['max_node_class_name_gpt2_length']]) + data_info['gpt2_eos_token']
    input_obs_node_gpt2_token_mask_padding = np.zeros([data_info['max_node_length']-len(input_obs_node_gpt2_token), data_info['max_node_class_name_gpt2_length']])
    
    input_obs_node_gpt2_token = np.concatenate((input_obs_node_gpt2_token, input_obs_node_gpt2_token_padding), axis=0)
    input_obs_node_gpt2_token_mask = np.concatenate((input_obs_node_gpt2_token_mask, input_obs_node_gpt2_token_mask_padding), axis=0)

    input_obs_node = input_obs_node_gpt2_token
    input_obs_node_mask = input_obs_node_gpt2_token_mask


    ## ----------------------------------------------------------------------------
    ## node state
    ## ----------------------------------------------------------------------------
    input_obs_node_state = np.zeros([data_info['max_node_length'], len(data_info['vocabulary_node_state_word_index_dict'])])
    input_obs_node_state_mask = np.zeros([data_info['max_node_length']])

    input_obs_node_state_tem = [state_one_hot(data_info['vocabulary_node_state_word_index_dict'], node['states']) for node in input_obs]
    input_obs_node_state_tem = np.stack(input_obs_node_state_tem)
    input_obs_node_state[:len(input_obs_node_state_tem)] = input_obs_node_state_tem
    input_obs_node_state_mask[:len(input_obs_node_state_tem)] = 1


    ## ----------------------------------------------------------------------------
    ## node coordinate
    ## ----------------------------------------------------------------------------
    agent = [node for node in input_obs if node['id'] == agent_id+1] ## current agent
    assert len(agent)==1 and agent[0]['class_name']=='character'
    agent = agent[0]
    char_coord = np.array(agent['bounding_box']['center'])

    rel_coords = [np.array([0,0,0])[None, :] if 'bounding_box' not in node.keys() else (np.array(node['bounding_box']['center']) - char_coord)[None, :] for node in input_obs]
    bounds = [np.array([0,0,0])[None, :] if 'bounding_box' not in node.keys() else np.array(node['bounding_box']['size'])[None, :] for node in input_obs]
    rel_coords = np.concatenate([rel_coords, bounds], axis=2)

    input_obs_node_coords = np.zeros([data_info['max_node_length'], 6]) # 6: center, size
    input_obs_node_coords_mask = np.zeros([data_info['max_node_length']])
    input_obs_node_coords[:len(input_obs)] = np.concatenate(rel_coords, 0)
    input_obs_node_coords_mask[:len(input_obs)] = 1

    return input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask
    


def get_history_action_input(args, data_info, agent_id, acts, step_i, tokenizer):
    
    previous_acts = acts[:step_i]
    
    if len(previous_acts)>0:
        goal_actions = [tem[agent_id] for tem in previous_acts if '[putback]' in tem[agent_id] or '[putin]' in tem[agent_id] or '[close]' in tem[agent_id] or '[switchon]' in tem[agent_id]]
        
        goal_actions_parsed = [parse_language_from_action_script(tem) for tem in goal_actions]
        history_actions = get_history_action_input_language(goal_actions_parsed)
        history_actions = history_actions[-(data_info['max_task_num']-1):]
    history_action_gpt2_token = np.zeros([data_info['max_task_num']-1, data_info['max_history_action_gpt2_length']]) + data_info['gpt2_eos_token']
    history_action_gpt2_token_mask = np.zeros([data_info['max_task_num']-1, data_info['max_history_action_gpt2_length']])

    if len(previous_acts)>0:
        if len(history_actions)>0:
            history_action_tem = [tem for tem in history_actions if tem not in data_info['history_action_gpt2_padding']]
            history_action_gpt2 = {tem: tokenizer(tem)['input_ids'] for tem in history_action_tem}

            for k,v in history_action_gpt2.items():
                index = np.zeros([data_info['max_history_action_gpt2_length']])+data_info['gpt2_eos_token']
                mask = np.zeros([data_info['max_history_action_gpt2_length']])
                index[:len(v)] = v
                mask[:len(v)] = 1
                data_info['history_action_gpt2_padding'][k] = index
                data_info['history_action_gpt2_padding_mask'][k] = mask
            
            history_action_gpt2_padding = [data_info['history_action_gpt2_padding'][tem] for tem in history_actions]
            history_action_gpt2_padding_mask = [data_info['history_action_gpt2_padding_mask'][tem] for tem in history_actions]
            

            history_action_gpt2_padding = np.stack(history_action_gpt2_padding)
            history_action_gpt2_padding_mask = np.stack(history_action_gpt2_padding_mask)
            
            history_action_gpt2_token[:len(history_action_gpt2_padding)] = history_action_gpt2_padding
            history_action_gpt2_token_mask[:len(history_action_gpt2_padding_mask)] = history_action_gpt2_padding_mask
    
    return history_action_gpt2_token, history_action_gpt2_token_mask
        

def get_goal_input(args, data_info, agent_id, env_task_goal, init_unity_graph, tokenizer):
    
    task_goal = env_task_goal[0][agent_id]
    task_goal_languages = get_goal_language(task_goal, init_unity_graph)
    # print(task_goal_languages)
    task_goal_languages_tem = [task_goal_language for task_goal_language in task_goal_languages if task_goal_language not in data_info['subgoal_gpt2_padding']]

    task_goal_languages_gpt2 = {tem: tokenizer(tem)['input_ids'] for tem in task_goal_languages_tem}
    for k,v in task_goal_languages_gpt2.items():
        index = np.zeros([data_info['max_subgoal_gpt2_length']])+data_info['gpt2_eos_token']
        mask = np.zeros([data_info['max_subgoal_gpt2_length']])
        index[:len(v)] = v
        mask[:len(v)] = 1
        data_info['subgoal_gpt2_padding'][k] = index
        data_info['subgoal_gpt2_padding_mask'][k] = mask

    goal_gpt2_token = np.zeros([data_info['max_task_num'], data_info['max_subgoal_gpt2_length']]) + data_info['gpt2_eos_token']
    goal_gpt2_token_mask = np.zeros([data_info['max_task_num'], data_info['max_subgoal_gpt2_length']])

    subgoal_gpt2_padding = [data_info['subgoal_gpt2_padding'][task_goal_language] for task_goal_language in task_goal_languages]
    subgoal_gpt2_padding_mask = [data_info['subgoal_gpt2_padding_mask'][task_goal_language] for task_goal_language in task_goal_languages]

    subgoal_gpt2_padding = np.stack(subgoal_gpt2_padding)
    subgoal_gpt2_padding_mask = np.stack(subgoal_gpt2_padding_mask)

    goal_gpt2_token[:len(subgoal_gpt2_padding)] = subgoal_gpt2_padding
    goal_gpt2_token_mask[:len(subgoal_gpt2_padding_mask)] = subgoal_gpt2_padding_mask
    
    goal_index = goal_gpt2_token
    goal_index_mask = goal_gpt2_token_mask
        
    return goal_index, goal_index_mask



def get_action_output(data_info, input_obs, output_act):
    action_name = re.findall(r"\[([A-Za-z0-9_]+)\]", output_act)[-1]
    object_name = re.findall(r"\<([A-Za-z0-9_]+)\>", output_act)[-1]
    object_id = re.findall(r"\(([A-Za-z0-9_]+)\)", output_act)[-1]
    action_index = data_info['vocabulary_action_name_word_index_dict'][action_name]

    object_node_index = [tem_idx for tem_idx, node in enumerate(input_obs) if node['id']==int(object_id)]
    assert len(object_node_index)==1
    object_node_index = object_node_index[0]
    output_action = np.array([action_index, object_node_index])
    
    return output_action

def get_input(args, agent_id, data_info, init_graph, task_goal, all_agent_cur_observation, all_agent_action, tokenizer, out_action):    
    input_obs = all_agent_cur_observation[-1]
    step_i = len(all_agent_action)+1
    ## current observation
    input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask = get_observation_input(args, data_info, input_obs, agent_id)
    
    ## history
    history_action_index, history_action_index_mask = get_history_action_input(args, data_info, agent_id, all_agent_action, step_i, tokenizer)
    output_action = get_action_output(data_info, input_obs, out_action)
    ## get goal
    init_unity_graph = deepcopy(init_graph)
    goal_index, goal_index_mask = get_goal_input(args, data_info, agent_id, [task_goal], init_unity_graph, tokenizer)
    
    input_obs_node = input_obs_node[np.newaxis, :]
    input_obs_node_mask = input_obs_node_mask[np.newaxis, :]
    input_obs_node_state = input_obs_node_state[np.newaxis, :]
    input_obs_node_state_mask = input_obs_node_state_mask[np.newaxis, :]
    input_obs_node_coords = input_obs_node_coords[np.newaxis, :]
    input_obs_node_coords_mask = input_obs_node_coords_mask[np.newaxis, :]

    history_action_index = history_action_index[np.newaxis, :]
    history_action_index_mask = history_action_index_mask[np.newaxis, :]

    output_action = output_action[np.newaxis, :]

    goal_index = goal_index[np.newaxis, :]
    goal_index_mask = goal_index_mask[np.newaxis, :]
    
    input_data = [input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask, \
                    history_action_index, history_action_index_mask, goal_index, goal_index_mask, output_action]

    return input_data, False


class UnityGraphDataset(Dataset):
    def __init__(self, args, data_info, expert_data_path, list_files, tokenizer):
        # expert_data_path is a directory that contains all the expert data
        self.args = args
        self.data_info = data_info
        self.expert_data_path = expert_data_path
        self.tokenizer = tokenizer
        self.data = []
        self.list_files = [file for file in list_files if file.startswith('expert')]

        
       #  for file in list_files:
       #      if file.startswith('expert'):
       #          saved_info = pickle.load(open(os.path.join(expert_data_path, file), 'rb'))
       #          init_unity_graph = saved_info['init_unity_graph']
       #          task_goal = saved_info['goals']
       #          num_steps = len(saved_info['action'][0])
       #          all_agent_cur_observation = []
       #          for step in range(num_steps):
       #              all_agent_cur_observation.append(filter_redundant_nodes(saved_info['obs'][step]))
       #              all_agent_action = saved_info['action'][0][:step]
       #              out_action = saved_info['action'][0][step]
       #              if out_action is None:
       #                  break
       #              input_data, _ = get_input(args, 0, data_info, init_unity_graph, task_goal, all_agent_cur_observation, all_agent_action, tokenizer, out_action)
       #              self.data.append(input_data)
        print('data size: ', len(self.list_files)) 
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        saved_info = pickle.load(open(os.path.join(self.expert_data_path, self.list_files[index]), 'rb'))
        init_unity_graph = saved_info['init_unity_graph']
        task_goal = saved_info['goals']
        num_steps = len(saved_info['action'][0])
        all_agent_cur_observation = []
        step = random.choice(range(num_steps))
        for i in range(step+1):
            all_agent_cur_observation.append(filter_redundant_nodes(saved_info['obs'][i]))
        all_agent_action = saved_info['action'][0][:step]
        out_action = saved_info['action'][0][step]
        input_data, _ = get_input(self.args, 0, self.data_info, init_unity_graph, task_goal, all_agent_cur_observation, all_agent_action, self.tokenizer, out_action)
        return input_data

