import glob
import os
import torch
import torch.nn as nn
import pdb
import json
import torch.nn.functional as F
import numpy as np
import sys
import pickle 
import random
import re
from random import sample
# import vh

from envs.unity_environment import UnityEnvironment
with open('./data/object_info.json') as f:
    object_info = json.load(f)
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../')

from data_loader import *


def connect_env(args, logging=None):
    if logging is not None:
        logging.info('------------------------------------------------------------------')
        logging.info('connecting to %s' % args.exec_file)
        logging.info('------------------------------------------------------------------')

    executable_args = {
        'file_name': args.exec_file,
        'x_display': "1",
        # 'no_graphics': not args.graphics
    }
    
    vh_envs = UnityEnvironment(num_agents=args.n_agent,
                            max_episode_length=args.max_episode_length,
                            port_id=0,
                            env_task_set=args.env_task_set,
                            observation_types=[args.obs_type, args.obs_type],
                            use_editor=args.use_editor,
                            executable_args=executable_args,
                            base_port=args.base_port,
                            seed=args.seed,
                            flag_stop_early=False)
    
    return vh_envs




def get_interactive_input(args, agent_id, data_info, vh_envs, all_agent_cur_observation, all_agent_action, tokenizer):    
    input_obs = all_agent_cur_observation[-1]
    step_i = len(all_agent_action)+1

    ## current observation
    input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask = get_observation_input(args, data_info, input_obs, agent_id)
    
    ## history
    history_action_index, history_action_index_mask = get_history_action_input(args, data_info, agent_id, all_agent_action, step_i, tokenizer)

    ## get goal
    init_unity_graph = deepcopy(vh_envs.init_graph)
    goal_index, goal_index_mask = get_goal_input(args, data_info, agent_id, [vh_envs.task_goal], init_unity_graph, tokenizer)
    
    input_obs_node = input_obs_node[np.newaxis, :]
    input_obs_node_mask = input_obs_node_mask[np.newaxis, :]
    input_obs_node_state = input_obs_node_state[np.newaxis, :]
    input_obs_node_state_mask = input_obs_node_state_mask[np.newaxis, :]
    input_obs_node_coords = input_obs_node_coords[np.newaxis, :]
    input_obs_node_coords_mask = input_obs_node_coords_mask[np.newaxis, :]

    history_action_index = history_action_index[np.newaxis, :]
    history_action_index_mask = history_action_index_mask[np.newaxis, :]

    goal_index = goal_index[np.newaxis, :]
    goal_index_mask = goal_index_mask[np.newaxis, :]
    
    input_data = [input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask, \
                    history_action_index, history_action_index_mask, goal_index, goal_index_mask]

    return input_data, False






def check_logical_before_unity(agent_id, cur_action, actions_sofar, observations_sofar, logging, verbose=False):
    
    objects_grab = object_info['objects_grab']
    objects_inside = object_info['objects_inside']
    objects_surface = object_info['objects_surface']
    objects_switchonoff = object_info['objects_switchonoff']
    
    if 'plate' in objects_surface:
        del objects_surface[objects_surface.index('plate')]

    if verbose:
        logging.info('check_logical_before_unity: %s' % cur_action)

    bad_action_flag = False
    ignore_walk = False

    if cur_action is not None:
        interacted_object = re.findall(r"\<([A-Za-z0-9_]+)\>", cur_action)[0]
        interacted_object_id = int(re.findall(r"\(([A-Za-z0-9_]+)\)", cur_action)[0])

        if '[putback]' in cur_action or '[putin]' in cur_action:
            target_object = re.findall(r"\<([A-Za-z0-9_]+)\>", cur_action)[1]
            target_object_id = int(re.findall(r"\(([A-Za-z0-9_]+)\)", cur_action)[1])


        ## check whether the interactied object in the observation
        if len(observations_sofar)!=0:
            current_observation = observations_sofar[-1]
            interacted_object_in_observation = [node for node in current_observation[agent_id]['nodes'] if node['id']==interacted_object_id]

            if len(interacted_object_in_observation)==0:
                if verbose:
                    logging.info('model error: %s is not in observation' % interacted_object)
                bad_action_flag = True

            if '[putback]' in cur_action or '[putin]' in cur_action:
                interacted_target_in_observation = [node for node in current_observation[agent_id]['nodes'] if node['id']==target_object_id]

                if len(interacted_target_in_observation)==0:
                    if verbose:
                        logging.info('model error: %s is not in observation' % target_object)
                    bad_action_flag = True
              

        if '[putback]' in cur_action or '[putin]' in cur_action:
            agent_target_close_edge = [edge for edge in current_observation[agent_id]['edges'] if (edge['from_id']==agent_id+1 and edge['to_id']==target_object_id and edge['relation_type']=='CLOSE') or (edge['from_id']==target_object_id and edge['to_id']==agent_id+1 and edge['relation_type']=='CLOSE')]
            
            if verbose:
                logging.info('agent_target_close_edge: %s' % str(agent_target_close_edge))

            if len(agent_target_close_edge)==0:
                if verbose:
                    logging.info('model error: walk close to %s before putback/putin %s' % (target_object, interacted_object))
                bad_action_flag = True


            if '[putback]' in cur_action:
                if target_object not in objects_surface:
                    if verbose:
                        logging.info('model error: target %s is not a container' % (target_object))
                    bad_action_flag = True
            if '[putin]' in cur_action:
                if target_object not in objects_inside:
                    if verbose:
                        logging.info('model error: target %s is not a container' % (target_object))
                    bad_action_flag = True

                ## target contrainer is open
                if len(observations_sofar)!=0:
                    current_observation = observations_sofar[-1]
                    interacted_target_in_observation = [node for node in current_observation[agent_id]['nodes'] if (node['id']==target_object_id and 'OPEN' in node['states'])]
                    if len(interacted_target_in_observation)==0:
                        if verbose:
                            logging.info('model error: %s is closed' % target_object)
                        bad_action_flag = True


        ## objects are not grabable
        if '[grab]' in cur_action:
            if interacted_object not in objects_grab:
                if verbose:
                    logging.info('model error: %s is not grabable' % interacted_object)
                bad_action_flag = True
            else:
                if len(observations_sofar)!=0:
                    current_observation = observations_sofar[-1]
                    agent_edge = [edge for edge in current_observation[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
                    agent_obj_edge = [edge for edge in agent_edge if edge['from_id']==interacted_object_id or edge['to_id']==interacted_object_id]
                    agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']
                    agent_obj_hold_edge = [edge for edge in agent_obj_edge if 'HOLD' in edge['relation_type']]

                    if verbose:
                        logging.info('agent_obj_hold_edge: %s' % str(agent_obj_hold_edge))
                        logging.info('agent_obj_close_edge: %s' % str(agent_obj_close_edge))
                    
                    if len(agent_obj_hold_edge)>0:
                        if verbose:
                            logging.info('model error: grab %s that is already in hand' % interacted_object)
                        bad_action_flag = True


                    if len(agent_obj_close_edge)==0:
                        if verbose:
                            logging.info('model error: %s is grabable, but no close edge' % interacted_object)
                        bad_action_flag = True

        ## objects cannot be open
        elif '[open]' in cur_action:
            if interacted_object not in objects_inside:
                if verbose:
                    logging.info('model error: %s cannot be open' % interacted_object)
                bad_action_flag = True

            if len(observations_sofar)!=0:
                current_observation = observations_sofar[-1]
                interacted_object_in_observation = [node for node in current_observation[agent_id]['nodes'] if node['id']==interacted_object_id]
                interacted_object_in_observation_open = [node for node in interacted_object_in_observation if 'OPEN' in node['states']]
                if len(interacted_object_in_observation_open)>0:
                    if verbose:
                        logging.info('model error: %s is already open' % interacted_object)
                    bad_action_flag = True

        elif '[close]' in cur_action:
            if interacted_object not in objects_inside:
                if verbose:
                    logging.info('model error: %s cannot be closed' % interacted_object)
                bad_action_flag = True

            if len(observations_sofar)!=0:
                current_observation = observations_sofar[-1]
                interacted_object_in_observation = [node for node in current_observation[agent_id]['nodes'] if node['id']==interacted_object_id]
                interacted_object_in_observation_open = [node for node in interacted_object_in_observation if 'CLOSED' in node['states']]
                if len(interacted_object_in_observation_open)>0:
                    if verbose:
                        logging.info('model error: %s is already closed' % interacted_object)
                    bad_action_flag = True


        elif '[switchon]' in cur_action:
            if interacted_object not in objects_switchonoff:
                if verbose:
                    logging.info('model error: %s cannot be open' % interacted_object)
                bad_action_flag = True

            if len(observations_sofar)!=0:
                current_observation = observations_sofar[-1]
                interacted_object_in_observation = [node for node in current_observation[agent_id]['nodes'] if node['id']==interacted_object_id]
                interacted_object_in_observation_open = [node for node in interacted_object_in_observation if 'ON' in node['states']]
                if len(interacted_object_in_observation_open)>0:
                    if verbose:
                        logging.info('model error: %s is already open' % interacted_object)
                    bad_action_flag = True


        ## grab before put
        elif '[putback]' in cur_action or '[putin]' in cur_action:
            check_grab_obj = False
            for past_actions in actions_sofar:
                if past_actions[agent_id] is not None:
                    if 'grab' in past_actions[agent_id] and str(interacted_object_id) in past_actions[agent_id]:
                        check_grab_obj = True

            if not check_grab_obj:
                if verbose:
                    logging.info('model error: put before grab %s (%d)' % (interacted_object, interacted_object_id))
                bad_action_flag = True

        elif '[walk]' in cur_action:
            agent_edge = [edge for edge in current_observation[agent_id]['edges'] if (edge['from_id']==agent_id+1 and edge['to_id']==interacted_object_id and edge['relation_type']=='CLOSE') or (edge['to_id']==agent_id+1 and edge['from_id']==interacted_object_id and edge['relation_type']=='CLOSE')]
            if len(agent_edge)>0:
                ignore_walk = True

    return bad_action_flag, ignore_walk


def get_valid_actions(obs, agent_id):
    objects_grab = object_info['objects_grab']
    objects_inside = object_info['objects_inside']
    objects_surface = object_info['objects_surface']
    objects_switchonoff = object_info['objects_switchonoff']

    
    valid_action_space = {}
    
    node_id_name_dict = {node['id']:node['class_name'] for node in obs[0]['nodes']}
    for agent_action in ['walk', 'grab', 'putback', 'putin', 'switchon', 'open', 'close']:
        if agent_action == 'walk':
            room_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in ['kitchen', 'livingroom', 'bathroom', 'bedroom']]
            # if len(room_nodes)!=4:
            #     pdb.set_trace()

            ignore_objs = ['walllamp', 'doorjamb', 'ceilinglamp', 'door', 'curtains', 'candle', 'wallpictureframe', 'powersocket']
            ignore_objs_idx = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['class_name'] in ignore_objs]
            # interacted_object_idxs = [tem for tem in list(range(len(obs[agent_id]['nodes']))) if tem not in ignore_objs_idx]
            interacted_object_idxs = [(node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if idx not in ignore_objs_idx]
            
        elif agent_action == 'grab':
            agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
            agent_obj_hold_edge = [edge for edge in agent_edge if 'HOLD' in edge['relation_type']]
            if len(agent_obj_hold_edge)>1:
                continue

            ignore_objs = ['radio']
            ignore_objs_id = [node['id'] for node in obs[agent_id]['nodes'] if node['class_name'] in ignore_objs]
            grabbable_object_ids = [node['id'] for node in obs[agent_id]['nodes'] if node['class_name'] in objects_grab]
            grabbable_object_ids = [tem for tem in grabbable_object_ids if tem not in ignore_objs_id]
            agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in grabbable_object_ids or edge['to_id'] in grabbable_object_ids]
            agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

            if len(agent_obj_close_edge)>0:
                interacted_object_ids = []
                interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                interacted_object_ids = list(np.unique(interacted_object_ids))
                interacted_object_ids.remove(agent_id+1)
                # interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
                interacted_object_idxs = [(node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
            else:
                continue

        elif agent_action == 'open':
            agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
            
            container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
            # container_object_nodes = [node for node in container_object_nodes if 'CLOSED' in node['states']] ## contrainer is closed 
            container_object_ids = [node['id'] for node in container_object_nodes]

            agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
            agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

            
            if len(agent_obj_close_edge)>0:
                interacted_object_ids = []
                interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                interacted_object_ids = list(np.unique(interacted_object_ids))
                interacted_object_ids.remove(agent_id+1)
                # interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
                interacted_object_idxs = [(node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
            else:
                continue
            
        elif agent_action == 'close':
            agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
            
            container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
            container_object_nodes = [node for node in container_object_nodes if 'OPEN' in node['states']] ## contrainer is closed 
            container_object_ids = [node['id'] for node in container_object_nodes]

            agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
            agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

            
            if len(agent_obj_close_edge)>0:
                interacted_object_ids = []
                interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                interacted_object_ids = list(np.unique(interacted_object_ids))
                interacted_object_ids.remove(agent_id+1)
                interacted_object_idxs = [(node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
                # interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
            else:
                continue

        elif agent_action == 'switchon':
            agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
            
            container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_switchonoff]
            container_object_nodes = [node for node in container_object_nodes if ('OFF' in node['states'])] ## contrainer is closed 
            container_object_ids = [node['id'] for node in container_object_nodes]

            agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
            agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

            if len(agent_obj_close_edge)>0:
                interacted_object_ids = []
                interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                interacted_object_ids = list(np.unique(interacted_object_ids))
                interacted_object_ids.remove(agent_id+1)
                interacted_object_idxs = [(node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
            else:
                continue


        elif agent_action == 'putin' or agent_action == 'putback':
            agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
            agent_obj_hold_edge = [edge for edge in agent_edge if 'HOLD' in edge['relation_type']]
            
            ignore_objs_tars = [('fryingpan', 'kitchencounter'), ('mug', 'sofa'), \
                                ('pillow', 'kitchencounter'), ('pillow', 'sofa'), ('pillow', 'fridge'), \
                                ('pillow', 'kitchencabinet'), ('pillow', 'coffeetable'), ('pillow', 'bathroomcabinet'), \
                                ('keyboard', 'coffeetable'), ('keyboard', 'bathroomcabinet'), ('keyboard', 'cabinet'), ('keyboard', 'sofa'), \
                                ('dishbowl', 'bathroomcabinet'), ('hairproduct', 'sofa')]
                                
            ignore_objs = [tem[0] for tem in ignore_objs_tars]

            if len(agent_obj_hold_edge)==0:
                continue
            else:
                if len(agent_obj_hold_edge)!=1:
                    continue
                
                holding_obj_name = node_id_name_dict[agent_obj_hold_edge[0]['to_id']]
                ignore_tar = [tem[1] for tem in ignore_objs_tars if tem[0]==holding_obj_name]
                holding_obj_id = agent_obj_hold_edge[0]['to_id']
                if agent_action == 'putin':
                    container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
                    container_object_nodes = [node for node in container_object_nodes if node['class_name'] not in ignore_tar]
                    container_object_nodes = [node for node in container_object_nodes if 'OPEN' in node['states']] ## contrainer is open 
                    container_object_ids = [node['id'] for node in container_object_nodes]
                elif agent_action == 'putback':
                    container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_surface]
                    container_object_nodes = [node for node in container_object_nodes if node['class_name'] not in ignore_tar]
                    container_object_ids = [node['id'] for node in container_object_nodes]

                agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
                agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']
                
                if len(agent_obj_close_edge)>0:
                    interacted_object_ids = []
                    interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                    interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                    interacted_object_ids = list(np.unique(interacted_object_ids))
                    interacted_object_ids.remove(agent_id+1)
                    interacted_object_idxs = [(holding_obj_id, holding_obj_name, node["id"], node["class_name"]) for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
                    # interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
                else:
                    continue

        else:
            continue

        if len(interacted_object_idxs)==0:
            continue
        else:
            valid_action_space[agent_action] = interacted_object_idxs
    return valid_action_space

def get_valid_action_space(args, agent_action_idx, obs, agent_id):
    objects_grab = object_info['objects_grab']
    objects_inside = object_info['objects_inside']
    objects_surface = object_info['objects_surface']
    objects_switchonoff = object_info['objects_switchonoff']

    agent_action = args.vocabulary_action_name_index_word_dict[agent_action_idx]
    valid_action_space = []
    
    node_id_name_dict = {node['id']:node['class_name'] for node in obs[0]['nodes']}

    if agent_action == 'walk':
        room_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in ['kitchen', 'livingroom', 'bathroom', 'bedroom']]
        # if len(room_nodes)!=4:
        #     pdb.set_trace()

        ignore_objs = ['walllamp', 'doorjamb', 'ceilinglamp', 'door', 'curtains', 'candle', 'wallpictureframe', 'powersocket']
        ignore_objs_idx = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['class_name'] in ignore_objs]
        interacted_object_idxs = [tem for tem in list(range(len(obs[agent_id]['nodes']))) if tem not in ignore_objs_idx]
        
    elif agent_action == 'grab':
        agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
        agent_obj_hold_edge = [edge for edge in agent_edge if 'HOLD' in edge['relation_type']]
        if len(agent_obj_hold_edge)>0:
            return None, None

        ignore_objs = ['radio']
        ignore_objs_id = [node['id'] for node in obs[agent_id]['nodes'] if node['class_name'] in ignore_objs]
        grabbable_object_ids = [node['id'] for node in obs[agent_id]['nodes'] if node['class_name'] in objects_grab]
        grabbable_object_ids = [tem for tem in grabbable_object_ids if tem not in ignore_objs_id]
        agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in grabbable_object_ids or edge['to_id'] in grabbable_object_ids]
        agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

        if len(agent_obj_close_edge)>0:
            interacted_object_ids = []
            interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
            interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
            interacted_object_ids = list(np.unique(interacted_object_ids))
            interacted_object_ids.remove(agent_id+1)
            interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
        else:
            return None, None

    elif agent_action == 'open':
        agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
        
        container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
        container_object_nodes = [node for node in container_object_nodes if 'CLOSED' in node['states']] ## contrainer is closed 
        container_object_ids = [node['id'] for node in container_object_nodes]

        agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
        agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

        
        if len(agent_obj_close_edge)>0:
            interacted_object_ids = []
            interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
            interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
            interacted_object_ids = list(np.unique(interacted_object_ids))
            interacted_object_ids.remove(agent_id+1)
            interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
        else:
            return None, None
        
    elif agent_action == 'close':
        agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
        
        container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
        container_object_nodes = [node for node in container_object_nodes if 'OPEN' in node['states']] ## contrainer is closed 
        container_object_ids = [node['id'] for node in container_object_nodes]

        agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
        agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

        
        if len(agent_obj_close_edge)>0:
            interacted_object_ids = []
            interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
            interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
            interacted_object_ids = list(np.unique(interacted_object_ids))
            interacted_object_ids.remove(agent_id+1)
            interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
        else:
            return None, None

    elif agent_action == 'switchon':
        agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
        
        container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_switchonoff]
        container_object_nodes = [node for node in container_object_nodes if ('OFF' in node['states'])] ## contrainer is closed 
        container_object_ids = [node['id'] for node in container_object_nodes]

        agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
        agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']

        if len(agent_obj_close_edge)>0:
            interacted_object_ids = []
            interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
            interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
            interacted_object_ids = list(np.unique(interacted_object_ids))
            interacted_object_ids.remove(agent_id+1)
            interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
        else:
            return None, None


    elif agent_action == 'putin' or agent_action == 'putback':
        agent_edge = [edge for edge in obs[agent_id]['edges'] if edge['from_id']==agent_id+1 or edge['to_id']==agent_id+1]
        agent_obj_hold_edge = [edge for edge in agent_edge if 'HOLD' in edge['relation_type']]
        
        ignore_objs_tars = [('toiletpaper', 'bathroomcabinet'), ('fryingpan', 'kitchencounter'), ('mug', 'sofa'), \
                            ('pillow', 'kitchencounter'), ('pillow', 'sofa'), ('pillow', 'fridge'), \
                            ('pillow', 'kitchencabinet'), ('pillow', 'coffeetable'), ('pillow', 'bathroomcabinet'), \
                            ('keyboard', 'coffeetable'), ('keyboard', 'bathroomcabinet'), ('keyboard', 'cabinet'), ('keyboard', 'sofa'), \
                            ('dishbowl', 'bathroomcabinet'), ('hairproduct', 'sofa')]
                            
        ignore_objs = [tem[0] for tem in ignore_objs_tars]

        if len(agent_obj_hold_edge)==0:
            return None, None
        else:
            if len(agent_obj_hold_edge)!=1:
                pdb.set_trace()
            
            holding_obj_name = node_id_name_dict[agent_obj_hold_edge[0]['to_id']]
            ignore_tar = [tem[1] for tem in ignore_objs_tars if tem[0]==holding_obj_name]
            if agent_action == 'putin':
                container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_inside]
                container_object_nodes = [node for node in container_object_nodes if node['class_name'] not in ignore_tar]
                container_object_nodes = [node for node in container_object_nodes if 'OPEN' in node['states']] ## contrainer is open 
                container_object_ids = [node['id'] for node in container_object_nodes]
            elif agent_action == 'putback':
                container_object_nodes = [node for node in obs[agent_id]['nodes'] if node['class_name'] in objects_surface]
                container_object_nodes = [node for node in container_object_nodes if node['class_name'] not in ignore_tar]
                container_object_ids = [node['id'] for node in container_object_nodes]

            agent_obj_edge = [edge for edge in agent_edge if edge['from_id'] in container_object_ids or edge['to_id'] in container_object_ids]
            agent_obj_close_edge = [edge for edge in agent_obj_edge if edge['relation_type']=='CLOSE']
            
            if len(agent_obj_close_edge)>0:
                interacted_object_ids = []
                interacted_object_ids += [edge['from_id'] for edge in agent_obj_close_edge]
                interacted_object_ids += [edge['to_id'] for edge in agent_obj_close_edge]
                interacted_object_ids = list(np.unique(interacted_object_ids))
                interacted_object_ids.remove(agent_id+1)
                interacted_object_idxs = [idx for idx, node in enumerate(obs[agent_id]['nodes']) if node['id'] in interacted_object_ids]
            else:
                return None, None

    else:
        pdb.set_trace()
    

    if len(interacted_object_idxs)==0:
        return None, None
    else:
        if agent_action == 'grab' or agent_action == 'close' or agent_action == 'switchon' or agent_action == 'putin' or agent_action == 'putback':
            interacted_object_idxs = [tem for tem in interacted_object_idxs if obs[0]['nodes'][tem]['class_name'] in args.data_info['history_action_token']]
            if len(interacted_object_idxs)==0:
                return None, None
            else:
                interacted_object_idx = random.choice(interacted_object_idxs)
        else:
            interacted_object_idx = random.choice(interacted_object_idxs)
        return interacted_object_idxs, interacted_object_idx
    



def args_per_action(action):
    action_dict = {'turnleft': 0,
        'walkforward': 0,
        'turnright': 0,
        'walktowards': 1,
        'open': 1,
        'close': 1,
        'putback':1,
        'putin': 1,
        'put': 1,
        'grab': 1,
        'no_action': 0,
        'none': 0,
        'walk': 1,
        'switchon': 1,
        'switchoff': 1}
    return action_dict[action]


def can_perform_action(action, o1, o1_id, agent_id, graph, graph_helper=None, teleport=True):
    bad_action_flag = False
    if action == 'no_action':
        return None, True
    
    obj2_str = ''
    obj1_str = ''
    id2node = {node['id']: node for node in graph['nodes']}
    num_args = 0 if o1 is None else 1
    grabbed_objects = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['relation_type'] in ['HOLDS_RH', 'HOLD_LH']]
    if num_args != args_per_action(action):
        return None, True

    # if 'walk' not in action and 'turn' not in action:
    #     return None
    close_edge = len([edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id and edge['to_id'] == o1_id and edge['relation_type'] == 'CLOSE']) > 0
    if action == 'grab':
        if len(grabbed_objects) > 0:
            return None, True

    if action.startswith('walk'):
        if o1_id in grabbed_objects:
            return None, True

    if o1_id == agent_id:
        return None, True

    if o1_id == agent_id:
        return None, True

    if (action in ['grab', 'open', 'close', 'switchon']) and not close_edge:
        return None, True

    if action == 'open':
        # print(o1_id, id2node[o1_id]['states'])
        if graph_helper is not None:
            if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                return None, True
        if 'OPEN' in id2node[o1_id]['states'] or 'CLOSED' not in id2node[o1_id]['states']:
            return None, True

    if action == 'close':
        #print(o1_id, id2node[o1_id]['states'])
        if graph_helper is not None:
            if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_inside']:
                return None, True
        if 'CLOSED' in id2node[o1_id]['states'] or 'OPEN' not in id2node[o1_id]['states']:
            return None, True

    if action == 'switchon':
        # print(o1_id, id2node[o1_id]['states'])
        if graph_helper is not None:
            if id2node[o1_id]['class_name'] not in graph_helper.object_dict_types['objects_switchonoff']:
                return None, True
        if 'ON' in id2node[o1_id]['states'] or 'OFF' not in id2node[o1_id]['states']:
            return None, True

    if 'put' in action:
        if len(grabbed_objects) == 0:
            return None, True
        else:
            o2_id = grabbed_objects[0]
            if o2_id == o1_id:
                return None, True
            o2 = id2node[o2_id]['class_name']
            obj2_str = f'<{o2}> ({o2_id})'

    if o1 is not None:
        obj1_str = f'<{o1}> ({o1_id})'
    if o1_id in id2node.keys():
        if id2node[o1_id]['class_name'] == 'character':
            return None, True

    if action.startswith('put'):
        if graph_helper is not None:
            if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_inside']:
                action = 'putin'
            if id2node[o1_id]['class_name'] in graph_helper.object_dict_types['objects_surface']:
                action = 'putback'
        else:
            if 'CONTAINERS' in id2node[o1_id]['properties']:
                action = 'putin'
            elif 'SURFACES' in id2node[o1_id]['properties']:
                action = 'putback'


    if action.startswith('walk') and teleport:
        # action = 'walkto'
        action = 'walk'

    if obj2_str=='':
        action_str = f'[{action}] {obj1_str}'.strip()
    else:
        action_str = f'[{action}] {obj2_str} {obj1_str}'.strip()

    return action_str, False

    


