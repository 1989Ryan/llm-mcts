import gym
import vh_graph
import time
import ipdb

print('Imports done')
env = gym.make('vh_graph-v0')
print('Env created')
state_path = 'example_graph.json'
task_goals = ['(inside cutting_board[2012] bathroom[1])']

print('Restart...')
s = env.reset(state_path , task_goals[0])
env.to_pomdp()

objects, predicates = env.get_objects_and_predicates()
ipdb.set_trace()

for i in range(3):
    actions = env.get_action_space()
    print(actions)
    print(actions[-1])
    rewards, states, infos = env.step(actions[-1])
    

ipdb.set_trace()
