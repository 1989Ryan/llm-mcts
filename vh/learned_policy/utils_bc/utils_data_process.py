import os
import sys
import re
import pdb
import pickle
import torch
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


def parse_language_from_action_script(action_script):
    act_name = re.findall(r"\[([A-Za-z0-9_]+)\]", action_script)[0]

    obj_name = re.findall(r"\<([A-Za-z0-9_]+)\>", action_script)[0]
    obj_id = re.findall(r"\(([A-Za-z0-9_]+)\)", action_script)[0]
    obj_id = int(obj_id)

    if '[putback]' in action_script or '[putin]' in action_script:
        obj_name2 = re.findall(r"\<([A-Za-z0-9_]+)\>", action_script)[1]
        obj_id2 = re.findall(r"\(([A-Za-z0-9_]+)\)", action_script)[1]
        obj_id2 = int(obj_id2)
    else:
        obj_name2 = None
        obj_id2 = None

    return act_name, obj_name, obj_id, obj_name2, obj_id2

def parse_language_from_goal_script(goal_script, goal_num, init_graph, template=0):
    goal_script_split = goal_script.split('_')
    
    if 'closed' in goal_script.lower():
        obj = goal_script_split[1]
        tar_node = [node for node in init_graph['nodes'] if node['id']==int(obj)]
        
        assert len(tar_node)==1
        
        if template==1:
            goal_language = 'could you please close the %s' % (tar_node[0]['class_name'])
        elif template==2:
            goal_language = 'please close the %s' % (tar_node[0]['class_name'])
        else:
            goal_language = 'close %s' % (tar_node[0]['class_name'])

    elif 'turnon' in goal_script.lower():
        obj = goal_script_split[1]
        tar_node = [node for node in init_graph['nodes'] if node['id']==int(obj)]
        assert len(tar_node)==1

        if template==1:
            goal_language = 'could you please turn on the %s' % (tar_node[0]['class_name'])
        elif template==2:
            goal_language = 'next turn on the %s' % (tar_node[0]['class_name'])
        else:
            goal_language = 'turn on %s' % (tar_node[0]['class_name'])


    elif 'on_' in goal_script.lower() or 'inside_' in goal_script.lower():
        # print(goal_script)
        rel = goal_script_split[0]
        obj = goal_script_split[1]
        tar = goal_script_split[2]
        tar_node = [node for node in init_graph['nodes'] if node['id']==int(tar)]
        assert len(tar_node)==1

        if template==1:
            goal_language = 'could you please place %d %s %s the %s' % (goal_num, obj, rel, tar_node[0]['class_name'])
        elif template==2:
            goal_language = 'get %d %s and put it %s the %s' % (goal_num, obj, rel, tar_node[0]['class_name'])
        else:
            goal_language = 'put %d %s %s the %s' % (goal_num, obj, rel, tar_node[0]['class_name'])
    else:
        pdb.set_trace()
    goal_language = goal_language.lower()
    return goal_language
    

def get_goal_language(task_goal, init_graph, template=0):
    goal_languages = [parse_language_from_goal_script(subgoal, subgoal_count, init_graph, template=template) for subgoal, subgoal_count in task_goal.items()]
    return goal_languages
        


def get_history_action_input_language(goal_actions, template=0):
    self_message_template = {}
    
    if template==1:
        self_message_template['putback'] = 'the robot placed one %s on the %s'
        self_message_template['putin'] = 'the robot placed one %s inside the %s'
        self_message_template['close'] = 'the robot closed the %s'
        self_message_template['switchon'] = 'the robot switched on the %s'
    elif template==2:
        self_message_template['putback'] = 'I just helped put one %s on the %s'
        self_message_template['putin'] = 'I just helped put one %s inside the %s'
        self_message_template['close'] = 'I just helped close the %s'
        self_message_template['switchon'] = 'I just helped switch on the %s'
    else:
        self_message_template['putback'] = 'I have put one %s on the %s'
        self_message_template['putin'] = 'I have put one %s inside the %s'
        self_message_template['close'] = 'I have closed the %s'
        self_message_template['switchon'] = 'I have switched on the %s'

    self_message = []
    for goal_action in goal_actions:
        if goal_action[0]=='putback':
            self_message.append(self_message_template['putback'] % (goal_action[1], goal_action[3]))
        elif goal_action[0]=='putin':
            self_message.append(self_message_template['putin'] % (goal_action[1], goal_action[3]))
        elif goal_action[0]=='close':
            self_message.append(self_message_template['close'] % (goal_action[1]))
        elif goal_action[0]=='switchon':
            self_message.append(self_message_template['switchon'] % (goal_action[1]))
        else:
            pdb.set_trace()
    return self_message



