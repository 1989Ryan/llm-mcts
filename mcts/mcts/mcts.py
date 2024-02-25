# adapted from https://github.com/jys5609/MC-LAVE-RL.git

import numpy as np
from tqdm import tqdm
import mcts.mcts.utils as utils
from collections import defaultdict
from mcts.virtualhome.llm_policy import LLMPolicy

DISCOUNT_FACTOR = 0.95

class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.look = None
        self.inv = None
        self.state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None
        self.history = []
        self.parent = None
        self.parent_action_id = None
        self.best_action_node = None
        

        self.N = 0
        self.children = []
        self.children_probs = []
        self.reward = reward/(1-DISCOUNT_FACTOR)
        self.score = 0
        self.done = done
        self.predicted_reward = 0
        self.use_llm = False


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Q_hat = 0
        self.Rs = []
        self.children = None
        self.children_id = None


class MCTSAgent:
    def __init__(self, args, env, policy=None, name='MCTS', 
                uct_type='PUCT', valid_action_dict=None, actions_info=None,
                  log_dir=None, visited_transitions=None, replay_file=None,
                  use_llm=True):
        self.env = env
        self.name = name
        # self.num_actions = env.action_num
        self.best_action_node = None
        self.uct_type = uct_type
        self.seed = args.seed
        self.round = args.round
        self.root = None


        self.exploration_constant = args.exploration_constant
        self.bonus_constant = args.bonus_constant
        self.max_depth = args.max_depth
        self.simulation_per_act = args.simulation_per_act
        self.discount_factor = args.discount_factor
        self.visited_transitions = visited_transitions

        self.action_selection_temp = 0.1 / (self.round + 1)

        self.policy = policy
        self.actions = [] if actions_info is None else actions_info[0]
        self.actions_e = [] if actions_info is None else actions_info[1]

        self.action_values = defaultdict(set)   # Ex: {north: [3.01, 2.00, 5.01]}

        self.maxlen_obs = 150
        self.maxlen_look = 150
        self.maxlen_inv = 50
        self.maxlen_action = 12
        self.simulation_num = args.simulation_num
        self.use_llm = use_llm
        if use_llm:
            self.llm_policy = LLMPolicy(device="cuda:0") 
        self.q_network = None
        # self.valid_action_dict = env.action_dict
        # self.valid_action_dict = {} if valid_action_dict is None else valid_action_dict
        self.state_dict = {}
        self.action_embedding = {}
        self.replay_file = replay_file

    def search(self, ob, history, cur_depth, valid_actions, done):
        '''
        Search the best action with probs
        :return: best action
        '''
        init_history = history.copy()
        # if '*** You have won ***' in next_state_text or '*** You have died ***' in next_state_text:
        #     score = int(next_state_text.split('you scored ')[1].split(' out of')[0])
        #     reward = score - state_node.score
        #     info['score'] = score

        # self.write_buffer(state_node, best_action_node, ob, reward, done, info)

        # if self.root is not None and self.root.best_action_node.children is not None:
        #     self.root = self.root.best_action_node.children
        #     self.root.parent = None
        # else:
        self.root = self.build_state(ob, history, valid_actions, done, use_llm=self.use_llm)

        for _ in tqdm(range(self.simulation_num)):
        # for _ in tqdm(range(self.simulation_per_act * len(self.root.children))):
            self.env.reset()
            self.env.history = init_history.copy()
            _, root = self.simulate(self.root, 0)
            self.root = root
        # select best action by Q-value
        best_action_node_idx = self.greedy_action_node(self.root, 0, 0, if_print=True)
        # select best action by Count
        # best_action_node = self.max_visit_action_node(self.root)
        best_action_node = self.root.children[best_action_node_idx]
        self.root.best_action_node = best_action_node
        return self.root.best_action_node.action

    @staticmethod
    def state_id(history: list):
        return ' '.join(history)

    def rebuild_state(self, state, ob, history, valid_actions, done, reward=0, prev_action='<s>', use_llm=False):
        state.id = self.state_id(history)
        # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
        state.valid_actions = valid_actions
        state.use_llm = use_llm

        if not use_llm:
            state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)

        # elif state.id in self.state_dict.keys():
        #     state.children_probs = self.state_dict[state.id].children_probs
        self.state_dict[state.id] = state
        for valid_action in state.valid_actions:
            if isinstance(state.valid_actions, dict):
                state.children.append(ActionNode(state.valid_actions[valid_action]))
            else:
                state.children.append(ActionNode(valid_action))

        return state

    def build_state(self, ob, history, valid_actions, done, reward=0, prev_action='<s>', use_llm=False):
        state = StateNode()
        state.ob = ob
        # state.look = info['look']
        # state.inv = info['inv']
        state.state = ob
        state.done = done
        # state.state = ob + info['look'] + info['inv']
        state.reward = reward
        # state.score = info['score']
        state.prev_action = prev_action
        state.history = history
        state.id = self.state_id(history)
        # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
        state.valid_actions = valid_actions
        state.use_llm = use_llm

        if not use_llm:
            state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)

            
        else:
            state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
                history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95)
            
        self.state_dict[state.id] = state
        for valid_action in state.valid_actions:
            if isinstance(state.valid_actions, dict):
                state.children.append(ActionNode(state.valid_actions[valid_action]))
            else:
                state.children.append(ActionNode(valid_action))

        return state

        
    def simulate(self, state_node, depth):

        if state_node.done or depth == self.max_depth:
            return 0, state_node

        best_action_node_idx = self.greedy_action_node(state_node, self.exploration_constant, self.bonus_constant)
        best_action_node = state_node.children[best_action_node_idx]
        rollout_next = False
        ob, reward, done, history, valid_actions = self.env.step(best_action_node.action)
        next_state_id = self.state_id(history)
        # path_of_nodes.append((state_node, best_action_node))
        if next_state_id == best_action_node.children_id:
            next_state_node = best_action_node.children
            if next_state_node.use_llm == False:
                next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
                # best_action_node.children[index] = next_state_node
                next_state_node.parent = state_node
                rollout_next = True
        else: 
            next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
            next_state_node.parent = state_node
            best_action_node.children = next_state_node
            best_action_node.children_id = next_state_node.id
            rollout_next = True


        if rollout_next:
            if self.use_llm:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
            else:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
        else:
            r, next_state_node = self.simulate(next_state_node, depth+1)
            R = reward + self.discount_factor * r

        state_node.N += 1
        best_action_node.N += 1
        best_action_node.children = next_state_node
        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))
        state_node.best_action_node = best_action_node       
        return R, state_node

    def max_visit_action_node(self, state_node):
        children_count = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            children_count.append(child.N)

        children_count = children_count / np.max(children_count)
        count_based_probs = children_count ** (1/self.action_selection_temp) / (np.sum(children_count ** (1/self.action_selection_temp)))
        return np.random.choice(state_node.children, p=count_based_probs)

    def greedy_action_node(self, state_node, exploration_constant, bonus_constant, if_print=False):
        best_value = -np.inf
        best_children = []
        best_children_prob = []
        for i in range(len(state_node.children)):
            child = state_node.children[i]
            assert len(state_node.children_probs) == len(state_node.children), print(state_node.children_probs)
            child_prob = state_node.children_probs[i]
            
            if exploration_constant == 0:
                ucb_value = child.Q
            elif self.uct_type == 'UCT':
                ucb_value = child.Q + exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1))
                # print(child.Q, exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1)))
            elif self.uct_type == 'PUCT':
                # print(child_prob)
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N) / (child.N + 1)
            elif self.uct_type == 'MC-LAVE':
                if child.action in self.action_embedding.keys():
                    action_e = self.action_embedding[child.action]
                else:
                    action_e = utils.vectorize(child.action)
                    self.action_embedding[child.action] = action_e

                actions = list(self.action_values.keys())
                if child.action in actions:
                    actions.pop(actions.index(child.action))

                actions_e = []
                for a in actions:
                    actions_e.append(self.action_embedding[a])

                near_act, near_idx = utils.find_near_actions(action_e, actions, np.array(actions_e), threshold=0.8)
                if len(near_idx) == 0:
                    child.Q_hat = 0
                else:
                    near_Qs = set()
                    for a in near_act:
                        near_Qs.add(np.mean(list(self.action_values[a])))
                    near_Qs = list(near_Qs)
                    child.Q_hat = utils.softmax_value(near_Qs)

                ucb_value = child.Q \
                            + exploration_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child_prob \
                            + bonus_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child.Q_hat

            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(i)
                best_children_prob.append(child_prob)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [i]
                best_children_prob = [child_prob]
        if if_print:
            for c in state_node.children:
                if c.N > 0:
                    print(c.action, c.Q, c.N)
        best_children_prob = np.array(best_children_prob) / np.sum(best_children_prob)
        output_action_index = np.argmax(best_children_prob)
        return best_children[output_action_index]

    def rollout(self, state_node, depth):
        if state_node.done or depth == self.max_depth:
            return 0
        action_node = np.random.choice(state_node.children, 1)[0]
        action = action_node.action

        ob, reward, done, history, valid_actions = self.env.step(action)
        if done:
            print("Done!")
        next_state_id = self.state_id(history)


        if next_state_id == action_node.children_id:
            next_state_node = action_node.children
        else:
            next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=action)
            next_state_node.parent = state_node
            action_node.children = next_state_node
            action_node.children_id = next_state_node.id
        r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
        return r