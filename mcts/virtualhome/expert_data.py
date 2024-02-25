import pickle
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import os
import torch
from vh.learned_policy.data_loader import get_history_action_input_language, parse_language_from_action_script, get_goal_language
from tqdm import tqdm
import random

def load_data(path):
    return pickle.load(open(path, 'rb'))

def translate_dataset(device):
    translation_lm = SentenceTransformer('stsb-roberta-large').to(device)
    expert_files = os.listdir('./expert_actions/expert_full')
    expert_files = random.sample(expert_files, 100)
    expert_files = ['./expert_actions/expert_full/' + file for file in expert_files if file.endswith('.pik')]
    expert_files_2 = os.listdir('./expert_actions/expert_simple')
    expert_files_2 = random.sample(expert_files_2, 100)
    expert_files_2 = ['./expert_actions/expert_simple/' + file for file in expert_files_2 if file.endswith('.pik')]
    expert_files += expert_files_2
    # expert_files = expert_files_2
    task_goal_list = []
    obs_langs_list = []
    obs_langs_embd_list = []
    act_langs_list = []
    for file in tqdm(expert_files):
        if not file.endswith('.pik'):
            continue
        expert_data = load_data(file)
        task_goal = expert_data['goals']
        goal_lang = get_goal(task_goal, expert_data['init_unity_graph'])
        obs_langs = [get_observation(obs) for obs in expert_data['obs']]
        act_langs = get_action_list_all(expert_data['action'], len(expert_data['action'][0]))
        obs_langs_embd = translation_lm.encode(obs_langs, convert_to_tensor=True)
        obs_langs_embd_list.append(obs_langs_embd)
        task_goal_list.append(goal_lang)
        obs_langs_list.append(obs_langs)
        act_langs_list.append(act_langs)
    task_goal_embd_list = translation_lm.encode(task_goal_list, convert_to_tensor=True)
    torch.save(task_goal_embd_list, './expert_actions/task_goal_embd_list.pt')
    torch.save(obs_langs_embd_list, './expert_actions/obs_langs_embd_list.pt')
    pickle.dump(obs_langs_list, open('./expert_actions/obs_langs_list.pik', 'wb'))
    pickle.dump(act_langs_list, open('./expert_actions/act_langs_list.pik', 'wb'))
    pickle.dump(task_goal_list, open('./expert_actions/task_langs_list.pik', 'wb'))

def get_observation(obs_graph):
    list_obs_item = [node['class_name'] for node in obs_graph]
    # eliminate duplicate
    list_obs_item = list(set(list_obs_item))
    observation = 'Visible objects are: ' + ', '.join(list_obs_item)
    return observation

def get_action_list_valid(acts, step_i, agent_id=0):
    if len(acts)>0:
        actions_parsed = [parse_language_from_action_script(tem) for tem in acts]
        history_actions = []
        for act in actions_parsed:
            if 'walk' in act[0]:
                history_actions.append(f'{act[0]} to {act[1]}')
            elif act[3] is not None:
                if act[0] == 'putin':
                    history_actions.append(f'put {act[1]} in {act[3]}')
                elif act[0] == 'putback':
                    history_actions.append(f'put {act[1]} on {act[3]}')
            else:
                history_actions.append(f'{act[0]} {act[1]}')
    return history_actions

def get_action_list_all(acts, step_i, agent_id=0):
    previous_acts = acts[agent_id][:step_i]
    if len(previous_acts)>0:
        goal_actions = [tem for tem in previous_acts]
        
        goal_actions_parsed = [parse_language_from_action_script(tem) for tem in goal_actions]
        history_actions = []
        for act in goal_actions_parsed:
            if 'walk' in act[0]:
                history_actions.append(f'{act[0]} to {act[1]}')
            elif act[3] is not None:
                if act[0] == 'putin':
                    history_actions.append(f'put {act[1]} in {act[3]}')
                elif act[0] == 'putback':
                    history_actions.append(f'put {act[1]} on {act[3]}')
            else:
                history_actions.append(f'{act[0]} {act[1]}')
    return history_actions

def get_action_history(acts, step_i, agent_id=0):
    previous_acts = acts[:step_i]
    
    if len(previous_acts)>0:
        goal_actions = [tem[agent_id] for tem in previous_acts]
        
        goal_actions_parsed = [parse_language_from_action_script(tem) for tem in goal_actions]
        history_actions = []
        for act in goal_actions_parsed:
            if 'walk' in act[0]:
                history_actions.append(f'{act[0]} to {act[3]}')
            elif act[3] is not None:
                history_actions.append(f'{act[0]} {act[1]} at {act[3]}')
            else:
                history_actions.append(f'{act[0]} {act[1]}')
        history_actions = 'Completed actions: ' + ', '.join(history_actions) + '.'
    else:
        history_actions = 'Completed actions: None.'
    return history_actions

def get_goal(goal, init_unity_graph):
    goal = goal[0]
    task_goal_languages = get_goal_language(goal, init_unity_graph)
    task_goal = 'Goal: ' + ', '.join(task_goal_languages) + '.'
    return task_goal

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    translate_dataset(device)