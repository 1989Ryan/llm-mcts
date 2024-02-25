from mcts.mcts.mcts import MCTSAgent
from mcts.virtualhome.llm_policy import LLMPolicy
from mcts.virtualhome.belief import Belief, container_classes, surface_classes
from mcts.virtualhome.llm_model import LLM_Model
from vh.data_gene.envs.unity_environment import UnityEnvironment
from vh.vh_mdp.vh_graph.envs.vh_env import VhGraphEnv
from vh.learned_policy.utils_bc.utils_interactive_eval import get_valid_actions
from vh.learned_policy.data_loader import get_goal_language
import pickle
import argparse
import time
import copy

def clean_graph(state, goal_spec, last_opened):
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate in goal_spec:
        elements = predicate.split('_')
        nodes_missing += [int(x) for x in elements if x.isdigit()]
        for x in elements[1:]:
            if x.isdigit():
                nodes_missing += [int(x)]
            else:
                nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == x]
    nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == 'character' or node['category'] in ['Rooms', 'Doors']]

    id2node = {node['id']: node for node in state['nodes']}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside.keys():
                inside[edge['from_id']] = []
            inside[edge['from_id']].append(edge['to_id'])
    
    while (len(nodes_missing) > 0):
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [node_in for node_in in inside[node_missing] if node_in not in ids_interaction]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']
                break
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
                break
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)


    new_graph = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
            "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph

class mcts_vh_env:
    def __init__(self, graph, goal_spec, task_goal):
        super().__init__()
        self.vh_pyenv = VhGraphEnv()
        self.goal_spec = goal_spec
        self.task_goal = task_goal
        self.vh_pyenv.pomdp = True
        self.model = None
        self.env_task_set = pickle.load(open('./vh/dataset/env_task_set_3_simple.pik', 'rb'))
        self.history = []
        self.init_history = []
        self.cur_state_graph = graph
        self.cur_state = self.vh_pyenv.get_vh_state(graph)
        self.init_state = copy.deepcopy(self.cur_state)
        self.init_graph = self.init_state.to_dict()
        self.belief = None
    
    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph['edges']:
            key = (edge['from_id'], edge['to_id'])
            if key not in edge_dict:
                edge_dict[key] = [edge['relation_type']]
                new_edges.append(edge)
            else:
                if edge['relation_type'] not in edge_dict[key]:
                    edge_dict[key] += [edge['relation_type']]
                    new_edges.append(edge)

        graph['edges'] = new_edges
        return graph

    def sample_belief(self):
        new_graph = self.belief.sample_from_belief()
        new_graph = self.filtering_graph(new_graph)
        return new_graph

    def update_and_sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        new_graph = self.filtering_graph(new_graph)
        return new_graph
    

    def check_progress(self, state, goal_spec):
        """TODO: add more predicate checkers; currently only ON"""
        count = 0
        for key, value in goal_spec.items():
            if key.startswith('off'):
                count += value
        id2node = {node['id']: node for node in state['nodes']}
        for key, value in goal_spec.items():
            elements = key.split('_')
            for edge in state['edges']:
                if elements[0] in ['on', 'inside']:
                    if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count += 1
                elif elements[0] == 'offOn':
                    if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count -= 1
                elif elements[1] == 'offInside':
                    if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count -= 1
                elif elements[0] == 'holds':
                    if edge['relation_type'].lower().startswith('holds') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                        count += 1
                elif elements[0] == 'sit':
                    if edge['relation_type'].lower().startswith('on') and edge['to_id'] == int(elements[2]) and edge['from_id'] == int(elements[1]):
                        count += 1
            if elements[0] == 'turnOn':
                if 'ON' in id2node[int(elements[1])]['states']:
                    count += 1
        goals = sum([value[0] for key, value in goal_spec.items()])
        if count < goals:
            reward = 0
        else:
            reward = 10
        return reward

    def graph_to_text(self, obs_graph):
        list_obs_item = [node['class_name'] for node in obs_graph['nodes']]
        list_obs_item = list(set(list_obs_item))
        observation = 'Visible objects are: ' + ', '.join(list_obs_item)
        return observation

    def get_observations_text(self, graph_env=None, char_index=0):
        par_obs = self.vh_pyenv.get_observations()
        par_obs_text = self.graph_to_text(par_obs)
        return par_obs_text
    
    def reset(self, obs=None, graph=None, goal_spec=None, task_goal=None):
        if graph is None:
            graph = self.init_graph
        else:
            self.belief = Belief(graph, agent_id=1)
            graph = self.sample_belief()
            self.init_graph = graph
        obs = self.vh_pyenv.reset(graph)

        self.cur_state_graph = graph
        self.cur_state = self.vh_pyenv.get_vh_state(graph)
        self.init_state = copy.deepcopy(self.cur_state)
        self.init_graph = copy.deepcopy(graph)
        self.goal_spec = goal_spec if goal_spec is not None else self.goal_spec
        self.task_goal = task_goal if task_goal is not None else self.task_goal

        self.history = []
        self.init_history = []
        valid_action_space = self.get_valid_action(obs)
        return obs, valid_action_space

    def get_valid_action(self, obs, agent_id=0):
        valid_action_space = []
        valid_action_space_dict = get_valid_actions(obs, agent_id)
        for action in valid_action_space_dict:
            interact_item_idxs = valid_action_space_dict[action]
            action = action.replace('walktowards', 'walk')
            if 'put' in action:
                
                valid_action_space += [
                    f'[{action}] <{grab_name}> ({grab_id}) <{item_name}> ({item_id})'
                        for grab_id, grab_name, item_id, item_name in interact_item_idxs]
            else:
                valid_action_space += [
                    f'[{action}] <{item_name}> ({item_id})'
                        for item_id, item_name in interact_item_idxs if item_name not in ['wall', 'floor', 'ceiling', 'curtain', 'window']]
        
        return valid_action_space
    
    def action_to_text(self, action):
        return None
    
    def get_action_from_text(self, text_action):
        return None

    def copy_env(self):
        self.reset(self.init_graph, self.goal_spec, self.task_goal)
        return self

    def get_goal(self):
        goal = self.task_goal[0]
        task_goal_languages = get_goal_language(goal, self.init_graph)
        task_goal = 'Goal: ' + ', '.join(task_goal_languages) + '.'
        return task_goal

    @staticmethod
    def get_goal_(formal_goal, init_graph):
        task_goal_languages = get_goal_language(formal_goal, init_graph)
        task_goal = 'Goal: ' + ', '.join(task_goal_languages) + '.'
        return task_goal

    def update(self, action, obs):
        self.vh_pyenv.step({0: action}) 
        self.cur_state = self.vh_pyenv.vh_state
        self.cur_state_graph = self.vh_pyenv.state
        obs = self.vh_pyenv._mask_state(self.cur_state_graph, 0)
        text_obs = self.graph_to_text(obs)
        if action is not None:
            self.history.append(action)
        valid_actions = self.get_valid_action([obs])
        reward = self.check_progress(self.cur_state_graph, self.goal_spec)
        if reward <= 0:
            done = False 
        else:
            done = True
        self.init_graph = copy.deepcopy(self.cur_state_graph)
        self.init_state = copy.deepcopy(self.cur_state)
        return text_obs, reward, done, self.history, valid_actions   

    def update_(self, action, obs):
        if action is not None:
            self.vh_pyenv.step({0: action}) 
        self.cur_state_graph = self.update_and_sample_belief(obs)
        self.cur_state = self.vh_pyenv.get_vh_state(self.cur_state_graph)
        text_obs = self.graph_to_text(obs)
        if action is not None:
            self.history.append(action)
        valid_actions = self.get_valid_action([obs])
        reward = self.check_progress(self.cur_state_graph, self.goal_spec)
        if reward <= 0:
            done = False 
        else:
            done = True
        self.init_graph = copy.deepcopy(self.cur_state_graph)
        self.init_state = copy.deepcopy(self.cur_state)
        return text_obs, reward, done, self.history, valid_actions   
    
    def step(self, action):
        obs = self.vh_pyenv._mask_state(self.cur_state_graph, 0)
        valid_actions = self.get_valid_action([obs])
        try:
            self.cur_state, succeed = self.vh_pyenv.transition(self.cur_state, {0: action}) 
        except:
            print(action)
        self.cur_state_graph = self.cur_state.to_dict()
        obs = self.vh_pyenv._mask_state(self.cur_state_graph, 0)
        plate_ids = []
        text_obs = self.graph_to_text(obs)
        self.history.append(action)

        valid_actions = self.get_valid_action([obs])
        reward = self.check_progress(self.cur_state_graph, self.goal_spec)
        if reward <= 0:
            done = False 
        else:
            done = True
        return text_obs, reward, done, self.history, valid_actions

class mcts_agent:
    def __init__(self, env, graph_env, model, args):
        self.env = env
        self.graph_env = graph_env
        self.model = model
        self.args = args
        self.mcts = MCTSAgent(self.env, self.graph_env, self.model, self.args)

    def reset(self):
        self.mcts.reset()

    def step(self, obs, reward, done):
        return self.mcts.step(obs, reward, done)

    def get_action(self, obs):
        return self.mcts.get_action(obs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exploration_constant', default=24, type=int)
    parser.add_argument('--bonus_constant', default=1, type=int)
    parser.add_argument('--max_episode_len', default=50, type=int)
    parser.add_argument('--max_depth', default=20, type=int)
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--simulation_per_act', default=2, type=int)
    parser.add_argument('--simulation_num', default=100, type=int)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--uct_type', default='PUCT', type=str)
    parser.add_argument('--mode', default='simple', type=str)
    parser.add_argument('--save_cache', action='store_true', default=False)
    parser.add_argument('--load_cache', action='store_true', default=False)
    parser.add_argument('--seen_item', action='store_true', default=False)
    parser.add_argument('--seen_apartment', action='store_true', default=False)
    parser.add_argument('--seen_comp', action='store_true', default=False)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--model', default="gpt-3.5-turbo-0125", type=str)
    return parser.parse_args()

def find_test_data_file_path(args):
    file_path = f'./vh/dataset/env_task_set_50_{args.mode}_'
    if not args.seen_item:
        file_path += 'unseen_item.pik'
    elif not args.seen_apartment:
        file_path += 'unseen_apartment.pik'
    elif not args.seen_comp:
        file_path += 'unseen_composition.pik'
    else:
        file_path += 'seen.pik'
    return file_path

def test():
    args = parse_args()
    file_path = find_test_data_file_path(args)
    env_task_set = pickle.load(open(file_path, 'rb'))
    executable_args = {
                    'file_name': "./vh/vh_sim/simulation/unity_simulator/v2.2.5/linux_exec.v2.2.5_beta.x86_64",
                    'x_display': "1",
                    'no_graphics': True
    }
    llm_model = LLM_Model("cuda:0", args.model)
    vhenv = UnityEnvironment(num_agents=1,
                                max_episode_length=100,
                                port_id=2,
                                env_task_set=env_task_set,
                                observation_types=["partial"],
                                use_editor=False,
                                executable_args=executable_args,
                                base_port=8084)
    goal_spec = vhenv.get_goal(vhenv.task_goal[0], vhenv.agent_goals[0])
    graph = vhenv.get_graph()
    container_name2id = {}

    for node in graph['nodes']:
        if node['class_name'] in container_classes or node['class_name'] in surface_classes:
            container_name2id[node['class_name']] = node['id']

    env = mcts_vh_env(graph, goal_spec, vhenv.task_goal)
    print(goal_spec)
    obs, valid_actions = env.reset(
        goal_spec=goal_spec,
        task_goal=vhenv.task_goal,
        graph=graph,
    )
    agent = MCTSAgent(args, env, uct_type=args.uct_type, use_llm=True)
    # llm_policy = LLMPolicy(device="cuda:0")
    print(vhenv.task_goal)
    history = []
    done = False
    succ = 0
    total = 0
    for i in range(len(vhenv.env_task_set)):

        agent.llm_policy.prompt_buffer.clear()
        obs = vhenv.reset(task_id=i)
        # if 'setup_table' in vhenv.task_name or 'put_dishwasher' in vhenv.task_name:
        #     continue
        goal_spec = vhenv.get_goal(vhenv.task_goal[0], vhenv.agent_goals[0])

        graph = vhenv.get_graph()
        plate_ids = []
        task_goal = vhenv.task_goal[0]
        goal = agent.env.get_goal_(task_goal, graph)
        formal_goal = llm_model.interpret_goal(goal, container_name2id)
        task_goal_ = {0: {key: num[0] for key, num in formal_goal.items()},
                      1: {key: num[0] for key, num in formal_goal.items()}}
        _, valid_actions = agent.env.reset(
            goal_spec=formal_goal,
            task_goal=task_goal_,
            graph=graph,
        )
        agent.env.update_(None, obs[0]) 
        # agent.env.update(None, obs[0]) 
        history = []
        
        # print(vhenv.task_goal)
        done = False
        for i in range(30):
            print(" ---------------------- Step: ", i, " ---------------------- ")
            action = agent.search(obs, history, i, valid_actions, done)
            # action = agent.llm_policy.act(history, obs, valid_actions, agent.env.get_goal()) 
            # ob, reward, done, history, valid_actions = env.step(agent.valid_action_dict[action])
            graph = vhenv.get_graph()
            plate_ids = []

            obs, reward, done, info, success = vhenv.step({0: action})
            agent.env.update_(action, obs[0]) 
            valid_actions = agent.env.get_valid_action(obs)
            history.append(action)
            if done:
                succ += 1
                break
        total += 1
        agent.root = None
        agent.state_dict = {}
        time.sleep(5)
        print("succ rate: ", succ / total)

if __name__ == "__main__" :
    test()