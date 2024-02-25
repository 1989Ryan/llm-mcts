import copy
import glob
import os, sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import pdb
import pickle
import json
import random
from copy import deepcopy

from utils_bc import utils_interactive_eval
from utils_bc.utils_graph import filter_redundant_nodes
from envs.utils.check_logical import check_env_bug



def sample_model_action(args, action_logits, object_logits, resampling, obs, agent_id, type='multinomial'):

    if type=='argmax':
        agent_action = int(action_logits.argmax())
        agent_obj = int(object_logits.argmax())
    elif type=='multinomial':
        action_dist  = torch.distributions.Multinomial(logits=action_logits, total_count=1)
        obj_dist = torch.distributions.Multinomial(logits=object_logits, total_count=1)
        agent_action =  int(torch.argmax(action_dist.sample(), dim=-1))
        agent_obj    =  int(torch.argmax(obj_dist.sample(), dim=-1))
    elif type=='multinomial_random':
        p = random.uniform(0, 1)
        if p < args.model_exploration_p:
            
            count = 0
            while 1:

                if resampling==-1 and count==0:
                    agent_action = int(torch.argmax(action_logits))
                else:
                    agent_action = int(torch.multinomial(action_logits, 1))
                    
                ## randomly select an action if stuck at a single action
                if count>50 or resampling>50:
                    agent_action = random.choice(list(args.vocabulary_action_name_word_index_dict.values()))
                

                object_logits_tem = deepcopy(object_logits)
                
                if agent_action==args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs, agent_id)

                    if agent_obj_space is not None:
                        not_agent_obj_space = [idx for idx in list(range(object_logits_tem.shape[1])) if idx not in agent_obj_space]
                        object_logits_tem[0][torch.tensor(not_agent_obj_space)] = -99999
                        object_logits_tem = F.softmax(object_logits_tem, -1)
                        
                        if resampling==-1 and count==0:
                            agent_obj = int(torch.argmax(object_logits_tem))
                        else:
                            agent_obj = int(torch.multinomial(object_logits_tem, 1))
                            
                        assert agent_obj in agent_obj_space
                        break

                count += 1
        else:
            count = 0
            while 1:
                action_logits_uniform = torch.ones_like(action_logits) / action_logits.shape[1]
                agent_action = int(torch.multinomial(action_logits_uniform, 1))
                count += 1
                
                if agent_action==args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs, agent_id)

                if agent_obj is not None:
                    break

    agent_action = args.vocabulary_action_name_index_word_dict[agent_action]
    resampling += 1
    return agent_action, agent_obj, resampling



def sample_action(args, obs, agent_id, action_logits, object_logits, all_actions, all_cur_observation, logging, all_graph_observation):

    graph_nodes = obs[agent_id]['nodes']
    agent_action = None
    agent_obj = None
    valid_action = False
    resampling = -1
    sample_model_action_type = 'argmax'

    while 1:
        if agent_action==None or agent_obj==None or agent_obj>=len(graph_nodes):
            agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits, resampling, obs, agent_id, type=sample_model_action_type)
        else:
            selected_node = graph_nodes[agent_obj]

            print(agent_action, selected_node['class_name'])
            action_obj_str, bad_action_flag = utils_interactive_eval.can_perform_action(agent_action, o1=selected_node['class_name'],
                                                           o1_id=selected_node['id'], agent_id=agent_id+1,
                                                           graph=obs[agent_id], teleport=True)

            bad_action_flag_v2, ignore_walk = utils_interactive_eval.check_logical_before_unity(agent_id, cur_action=action_obj_str, actions_sofar=all_actions, observations_sofar=all_graph_observation, logging=logging, verbose=False)
            
            if bad_action_flag or bad_action_flag_v2 or ignore_walk:
                agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits, resampling, obs, agent_id, type=sample_model_action_type)
            else:
                valid_action = True
                break
    
    if not valid_action:
        ignore_walk = False
        action_obj_str = None
    return action_obj_str, ignore_walk, resampling

    


def interactive_interface_fn(args, vh_envs, iteri, agent_model, data_info, logging, tokenizer):

    verbose = True
    valid_run = 0
    success_count = 0
    save_output = []
    camera_num = vh_envs.comm.camera_count()[1]
    save_data_all = []


    i = 0
    while 1:
        i += 1
        print('valid_run/current_run', valid_run, i)
        if valid_run>=args.test_examples:
            break

        all_cur_observation = []
        all_graph_observation = []
        all_actions = []
        all_rewards = []
        all_frames = []

        if True:
            obs = vh_envs.reset(task_id=i)
            obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
            all_cur_observation.append(deepcopy(obs[0]['nodes']))
            all_graph_observation.append(deepcopy(obs))
            B = 1
            steps = 0
            
            valid_run_tem = False
            success_run_tem = False

            
            while (1):
                if verbose:
                    logging.info('----------------------------------------------------------------------------------------------------')
        

                agent_id = 0
                agent_actions = {}
                agent_rewards = {}
                agent_ignore_walk = {}
                ignore_walk = None


                ## ----------------------------------------------------------------------------------------------------
                ## convert data format 
                ## ----------------------------------------------------------------------------------------------------
                data, bad_observation_flag = utils_interactive_eval.get_interactive_input(args, agent_id, data_info, vh_envs, all_cur_observation, all_actions, tokenizer)
                
                if bad_observation_flag:
                    logging.info('----------------------------------------------------------------------------------')
                    logging.info('interactive eval: convert data format fail!')
                    logging.info('----------------------------------------------------------------------------------')
                    valid_run_tem = False
                    break

                ## ----------------------------------------------------------------------------------------------------
                ## get action from model and check action
                ## ----------------------------------------------------------------------------------------------------
                action, obj = agent_model.get_action(data=data)

                action_logits = F.softmax(action[agent_id], dim=-1)
                object_logits = F.softmax(obj[agent_id], dim=-1)
                print(action_logits) 
                print(object_logits)
                action_obj_str, ignore_walk, resampling = sample_action(args, obs, agent_id, action_logits, object_logits, all_actions, all_cur_observation, logging, all_graph_observation)
                agent_actions[agent_id] = action_obj_str
                agent_ignore_walk[agent_id] = ignore_walk

                ## ----------------------------------------------------------------------------------------------------
                ## send action to the environment
                ## ----------------------------------------------------------------------------------------------------
                obs, rewards, dones, infos, success = vh_envs.step(agent_actions, ignore_walk=agent_ignore_walk, logging=logging) # next_obs

                if rewards==dones==infos==success==None:
                    logging.info('----------------------------------------------------------------------------------')
                    logging.info('interactive eval: unity action fail!')
                    logging.info('----------------------------------------------------------------------------------')
                    valid_run_tem = False
                    break

                ## ---------------------------------------------------------------------------------------------------------
                ## check action after send to Unity
                ## ---------------------------------------------------------------------------------------------------------
                obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
                env_bug_count_a0 = not check_env_bug(agent_actions[0], obs[0], agent_i=0, logging=logging)
                
                if env_bug_count_a0:
                    logging.info('----------------------------------------------------------------------------------')
                    logging.info('interactive eval: check_env_bug outside unity fail!')
                    logging.info('----------------------------------------------------------------------------------')
                    valid_run_tem = False
                    break
                
                ## ----------------------------------------------------------------------------------------------------
                ## reward
                ## ----------------------------------------------------------------------------------------------------
                reward = torch.tensor(rewards)                
                if reward[0] is not None:
                    agent_rewards[0] = reward[0]

                ## ----------------------------------------------------------------------------------------------------
                ## done, bad end
                ## ----------------------------------------------------------------------------------------------------
                all_cur_observation.append(deepcopy(obs[0]['nodes']))
                all_graph_observation.append(deepcopy(obs))
                all_actions.append(deepcopy(agent_actions))
                all_rewards.append(deepcopy(agent_rewards))


                ## ---------------------------------------------------------------------------------------------------------
                ## log
                ## ---------------------------------------------------------------------------------------------------------
                if verbose:
                    env_task_goal_write = ['%s_%d'%(k,v) for k,v in vh_envs.task_goal[0].items() if v>0]

                    logging.info('example %d, step %d, goal %s' % (i, steps, str(env_task_goal_write)))
                    logging.info( ('A0: Act: %s'%str(agent_actions[0])) )
                    logging.info( ('A0: Rew: %s'%str(agent_rewards[0])) )
                    
                    if agent_actions[0] is not None:
                        logging.info( ( 'ignore_walk: %s' % str(agent_ignore_walk[0]) ) )

                ## ---------------------------------------------------------------------------------------------------------
                ## break if done
                ## ---------------------------------------------------------------------------------------------------------
                steps += 1
                if np.any(dones):
                    valid_run_tem = True

                    if infos[0]['is_success']:
                        success_run_tem = True
                    break


            if valid_run_tem:
                
                valid_run += 1
                
                for tem in all_actions: logging.info(tem)

                if success_run_tem:
                    success_count += 1
                    print('-------------------------------------------------------------------')
                    print('success example')
                    print('-------------------------------------------------------------------')

            
        if args.interactive_eval:
            success_rate = 100. * success_count / valid_run if valid_run!=0 else 0

            if args.eval:
                logging.info(" {} / {} \n \
                            Total / Current_run / Valid / Success: {} / {} / {} / {} \n \
                            Success Rate: {:.3f}"
                            .format(args.pretrained_model_dir, args.subset,
                                    args.test_examples, i+1, valid_run, success_count,
                                    success_rate))
            else:
                logging.info(" {} / {} \n \
                            Total / Current_run / Valid / Success: {} / {} / {} / {} \n \
                            Success Rate: {:.3f}"
                            .format(args.save_dir, args.subset,
                                    args.test_examples, i+1, valid_run, success_count,
                                    success_rate))

    return success_rate

        

        


        































