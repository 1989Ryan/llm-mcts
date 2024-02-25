# vh_mdp

## Requirement
- virtualhome (https://github.com/xavierpuigf/virtualhome/tree/virtualhome_pkg)
## Usage
```python
import gym
import vh_graph

env = gym.make('vh_graph-v0')
state_path = PATH_TO_STATE_JSON_FILE
task_goals = ['(ontop phone247 kitchen_counter230)']

s = env.reset(state_path , task_goals)
rewards, states, infos = env.step("[walk] <bedroom> (67)")

s = env.reset(state_path , task_goals)
env.to_pomdp()
rewards, states, infos = env.step("[walk] <bedroom> (67)")
```
