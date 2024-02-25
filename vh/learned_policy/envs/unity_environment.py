import sys
import os
import atexit
import ipdb
import random
import numpy as np
import re
import pdb
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from envs.utils import utils_environment
from envs.utils import check_logical
from .base_environment import BaseEnvironment

import json
import numpy as np
from vh_sim.simulation.unity_simulator.comm_unity import UnityCommunication
from vh_sim.simulation.unity_simulator import utils_viz
from vh_mdp.vh_graph.envs import belief, vh_env


class UnityEnvironment(BaseEnvironment):
    def __init__(self,
                 num_agents=2,
                 max_episode_length=200,
                 env_task_set=None,
                 observation_types=None,
                 use_editor=False,
                 base_port=8080,
                 port_id=0,
                 recording=False,
                 output_folder=None,
                 file_name_prefix=None,
                 executable_args={},
                 seed=123,
                 flag_stop_early=False):

        self.seed = seed
        self.prev_reward = 0.
        self.rnd = random.Random(seed)
        self.port_id = 0.
        
        self.steps = 0
        self.env_id = None
        self.max_ids = {}

        self.python_graph = None
        self.env_task_set = env_task_set

        self.num_agents = num_agents
        self.max_episode_length = max_episode_length

        self.recording = recording
        self.base_port = base_port
        self.port_id = port_id
        self.output_folder = output_folder
        self.file_name_prefix = file_name_prefix

        self.default_width = 128
        self.default_height = 128
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = 1  # 0 TOP, 1 FRONT, 2 LEFT..

        
        if observation_types is not None:
            self.observation_types = observation_types
        else:
            self.observation_types = ['partial' for _ in range(num_agents)]

        self.agent_info = {
            0: 'Chars/Female1',
            1: 'Chars/Male1'
        }

        if self.num_agents==1:
            self.task_goal = {0: {}}
        elif self.num_agents==2:
            self.task_goal = {0: {}, 1: {}}

        self.changed_graph = False
        self.rooms = None
        self.id2node = None
        self.offset_cameras = None

        self.port_number = 8080
        self.executable_args = executable_args

        if use_editor:
            # Use Unity
            self.comm = UnityCommunication()
        else:
            # Launch the executable
            self.port_number = self.base_port + port_id
            self.comm = UnityCommunication(no_graphics=True, x_display="1", file_name = "./vh/vh_sim/simulation/unity_simulator/v2.2.5/linux_exec.v2.2.5_beta.x86_64", port=f"{self.port_number}")



        atexit.register(self.close)

        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)
        self.reset()

    def close(self):
        self.comm.close()

    def relaunch(self):
        self.comm.close()
        self.comm = UnityCommunication(
            no_graphics=True, x_display="1", file_name = "./vh/vh_sim/simulation/unity_simulator/v2.2.5/linux_exec.v2.2.5_beta.x86_64", port=f"{self.port_number}")



    def reward(self, agent_i):
        
        reward = 0.
        done = True

        has_cur_task_prediction_flag = False # if the prediction has current task
        
        if has_cur_task_prediction_flag:
            if self.cur_task[agent_i] is None:
                assert agent_i==1
                reward = 0
                done = False
                self.prev_reward = reward
                return reward, done, {'satisfied_goals': {}, 'unsatisfied_goals': {}}
                
            ## check current subgoal -> reward
            cur_task = {'_'.join(self.cur_task[agent_i].split('_')[:-1]): int(self.cur_task[agent_i].split('_')[-1])}
            goal_spec_cur = {goal_k: [goal_c, True, 2] for goal_k, goal_c in cur_task.items()}

            satisfied, unsatisfied = utils_environment.check_progress(self.get_graph(), cur_task)
            for key, value in satisfied.items():
                preds_needed, mandatory, reward_per_pred = goal_spec_cur[key]
                # How many predicates achieved
                value_pred = min(len(value), preds_needed)
                reward += value_pred * reward_per_pred

            self.prev_reward = reward
            
        else:
            reward = 0

        
        ## check all the subgoals -> done
        goal_spec = {goal_k: [goal_c, True, 2] for goal_k, goal_c in self.task_goal[0].items()}
        satisfied, unsatisfied = utils_environment.check_progress(self.get_graph(), self.task_goal[0])

        
        for key, value in satisfied.items():
            preds_needed, mandatory, reward_per_pred = goal_spec[key]
            if mandatory and unsatisfied[key] > 0:
                done = False

            if not has_cur_task_prediction_flag:
                value_pred = min(len(value), preds_needed)
                reward += value_pred * reward_per_pred

        return reward, done, {'satisfied_goals': satisfied, 'unsatisfied_goals': unsatisfied}



    def check_edge(self, script_list, graph, agent_i, fix_edge=False):
        correct_graph_flag = True
        if script_list is not None and len(script_list)>0:
            agent_nodes = [node for node in graph['nodes'] if node['id']==agent_i+1]
            assert len(agent_nodes)==1
            obj_name = re.findall(r"\<([A-Za-z0-9_]+)\>", script_list)[1]
            obj_id = int(re.findall(r"\(([A-Za-z0-9_]+)\)", script_list)[0])

            correct_graph_flag, graph = check_logical.check_env_bug_step(correct_graph_flag, script_list, obj_name, obj_id, graph, agent_i, fix_edge=fix_edge)

        return correct_graph_flag, graph
        


    def step(self, action_dict, ignore_walk=None, logging=None):

        script_lists = utils_environment.convert_action(self.num_agents, action_dict)
        
        num_actions = len([item for item, action in action_dict.items() if action is not None]) # how many actions are not None
        
        if ignore_walk is not None:
            
            if 'walk' in script_lists[0] and ignore_walk[0]:
                script_lists[0] = ''


            if self.num_agents==2:
                if 'walk' in script_lists[1] and ignore_walk[1]:
                    script_lists[1] = ''


        if num_actions!=0:
            if self.recording:
                success, message = self.comm.render_script(script_lists,
                                                           recording=True,
                                                           skip_animation=False,
                                                           camera_mode=['PERSON_FROM_BACK'],
                                                           file_name_prefix='task_{}'.format(self.task_id),
                                                           processing_time_limit=60,
                                                           image_synthesis=['normal'])
            else:
                success, message = self.comm.render_script(script_lists,
                                                           recording=False,
                                                           processing_time_limit=20,
                                                           skip_animation=True)

            if not success:
                return message, None, None, None, None
            
        
        rewards = []
        dones = []
        infos = []
        successes = []

        self.changed_graph = True
        graph = self.get_graph()
        

        self.python_graph_reset(deepcopy(graph))
        next_obs = self.get_observations() # partial observation

        
        
        if self.num_agents==1:
            if num_actions!=0:
                script_lists = script_lists
            else:
                script_lists = [None]
                success = True


        for agent_i, script_list in enumerate(script_lists):
            reward, done, info = self.reward(agent_i)
            
            info['gt_full_graph'] = deepcopy(graph)
            info['is_success'] = done
            infos.append(info)
            rewards.append(reward)
            dones.append(done)
            successes.append(success)
        
        
        infos[0]['bad_end'] = False
        if len(script_lists)==2:
            infos[1]['bad_end'] = False


    
        self.steps += 1
        if self.steps == self.max_episode_length:
            if len(script_lists)==1:
                dones = [True]
                infos[0]['bad_end'] = True
            elif len(script_lists)==2:
                dones = [True, True]
                infos[0]['bad_end'] = True
                infos[1]['bad_end'] = True

        
        ## check edges
        for agent_i, script_list in enumerate(script_lists):
            correct_graph_flag, next_obs[agent_i] = self.check_edge(script_list, next_obs[agent_i], agent_i, fix_edge=True)
            
        return next_obs, rewards, dones, infos, successes


    def python_graph_reset(self, graph):
        new_graph = utils_environment.inside_not_trans(graph) # clean each node to have just one node
        self.python_graph = new_graph
    
        self.env.reset(new_graph)
        self.env.to_pomdp()


    def reset(self, environment_graph=None, task_id=None):
        self.cur_task = {0: None, 1: None}

        print('--------------------------------------------------------------------------')
        print('task_id', task_id)
        print('--------------------------------------------------------------------------')
        
        # Make sure that characters are out of graph, and ids are ok
        if task_id is None:
            task_id = self.rnd.choice(list(range(len(self.env_task_set))))


        env_task = self.env_task_set[task_id]

        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_name = env_task['task_name']
        self.task_goal = env_task['task_goal']

        ## the second agent does not know the goal
        if self.num_agents==1:
            if 1 in self.task_goal:
                del self.task_goal[1]
        elif self.num_agents==2:
            self.task_goal[1] = {}
        

        old_env_id = self.env_id
        self.env_id = env_task['env_id']

        
        if False: # old_env_id == self.env_id:
            print("Fast reset")
            self.comm.fast_reset()
        else:
            self.comm.reset(self.env_id)

        s,g = self.comm.environment_graph()
        if self.env_id not in self.max_ids.keys():
            max_id = max([node['id'] for node in g['nodes']])
            self.max_ids[self.env_id] = max_id

        max_id = self.max_ids[self.env_id]
        
        if environment_graph is not None:
            # TODO: this should be modified to extend well
            updated_graph = utils_environment.separate_new_ids_graph(environment_graph, max_id)
            success, m = self.comm.expand_scene(updated_graph)
        else:
            updated_graph = utils_environment.separate_new_ids_graph(env_task['init_graph'], max_id)
            success, m = self.comm.expand_scene(updated_graph)
        
        if not success:
            print("Error expanding scene")
            pdb.set_trace()
            return None
        self.offset_cameras = self.comm.camera_count()[1]

        if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
            rooms = self.rnd.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        else:
            rooms = list(self.init_rooms)

        for i in range(self.num_agents):
            if i in self.agent_info:
                self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
            else:
                self.comm.add_character()

        _, self.init_unity_graph = self.comm.environment_graph()


        self.changed_graph = True
        graph = self.get_graph()

        
        self.python_graph_reset(graph)
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}

        obs = self.get_observations() # partial observation
        self.steps = 0
        self.prev_reward = 0.
        return obs

    def get_graph(self):
        if self.changed_graph:
            s, graph = self.comm.environment_graph()
            # print("CHAR", [node['bounding_box']['size'] for node in graph['nodes'] if node['id'] == 1][0])
            clothespile = [node['id'] for node in graph['nodes'] if node['class_name'] == 'clothespile']
            graph['nodes'] = [node for node in graph['nodes'] if node['id'] not in clothespile]
            graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in clothespile and edge['to_id'] not in clothespile]

            if not s:
                pdb.set_trace()
            self.graph = graph
            self.changed_graph = False
        return self.graph

    def get_observations(self):
        dict_observations = {}
        for agent_id in range(self.num_agents):
            obs_type = self.observation_types[agent_id]
            dict_observations[agent_id] = self.get_observation(agent_id, obs_type)
        return dict_observations

    def get_observation(self, agent_id, obs_type, info={}):

        if obs_type == 'partial':
            return self.env.get_observations(char_index=agent_id)

        elif obs_type == 'full':
            return self.env.vh_state.to_dict()

        elif obs_type == 'visible':
            raise NotImplementedError

        else:
            camera_ids = [self.offset_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
            if 'image_width' in info:
                image_width = info['image_width']
                image_height = info['image_height']
            else:
                image_width, image_height = self.default_width, self.default_height

            s, images = self.comm.camera_image(camera_ids, mode=obs_type, image_width=image_width, image_height=image_height)
            if not s:
                pdb.set_trace()
            return images[0]

    def get_action_space(self):
        ## get the id of visible objects
        dict_action_space = {}
        for agent_id in range(self.num_agents):
            if self.observation_types[agent_id] not in ['full', 'partial']:
                raise NotImplementedError
            obs_type = self.observation_types[agent_id]

            visible_graph = self.get_observation(agent_id, obs_type)
            dict_action_space[agent_id] = [node['id'] for node in visible_graph['nodes']]
        return dict_action_space






