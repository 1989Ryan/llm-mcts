import os
import re
import random
import numpy as np

from gym.spaces.space import Space
from vh.vh_sim.simulation.evolving_graph.utils import load_graph_dict, graph_dict_helper


verbose = True
def printf(s):
    if verbose:
        print(s)


class Graph(Space):

    r"""A graph space composed of variable size of tuple
    Example::
        >>> Graph()
    """

    def __init__(self, max_nodes=200):
        
        assert max_nodes <= 300
        self.max_nodes = max_nodes
        self.helper = graph_dict_helper(max_nodes=max_nodes)
        super(Graph, self).__init__()

    def sample(self, base_graph_path):
        
        helper = self.helper
        max_nodes = self.max_nodes

        #sampled_base_graph_path = random.sample(self.base_graph, 1)[0]
        base_graph = load_graph_dict(base_graph_path)
        helper.initialize(base_graph)

        graph = base_graph

        # add random stuff
        max_node_to_place = max_nodes - len(graph["nodes"])
        n = random.randint(max_node_to_place - 20, max_node_to_place)
        helper.add_random_objs_graph_dict(graph, n=max(n, 0))
        helper.set_to_default_state(graph, None, id_checker=lambda v: v >= 2000)
        helper.random_change_object_state({}, graph, id_checker=lambda v: True)

        graph = self.check(graph)
        return graph
        
    def check(self, graph):

        helper = self.helper

        helper.open_all_doors(graph)
        helper.ensure_light_on(graph, id_checker=lambda v: True)
        helper.check_binary(graph, id_checker=lambda v: True, verbose=False)
        return graph

    def contains(self, x):
        raise NotImplementedError
        
    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, Graph) and self.n == other.n


class Task(object):
    def __init__(self, goal):
        self.goal = goal
        self.goal_id = int(goal.split('_')[-1])
        self.done = False

    def measure_progress(self, state: dict, char_id):
        if not self.done:
            node_id_char = [x['id'] for x in state['nodes'] if x['class_name'] == 'character'][char_id]
            edges_goal = [x for x in state['edges']
                            if x['relation_type'] == 'CLOSE' and x['from_id'] == self.goal_id and x['to_id'] == node_id_char]
            self.done = len(edges_goal) > 0

        return int(self.done)


# class Task(object):

#     # Input `goal` is represented as propositional form
#     # For example, 
#     # (or (and (ontop plate1 table13) (ontop plate3 table13)) (ontop plate4 table13))
#     # TODO: Negate is not supported now
#     def __init__(self, goal):

#         if isinstance(goal, str):
#             self.goal = self._parse_goal_str(goal)
#         else:
#             raise TypeError

#     def measure_progress(self, state: dict):
        
#         def _divide(ele):
#             if isinstance(ele, Predicate):
#                 valid = self.is_valid(ele, state)
#                 if valid:
#                     printf("Predicate: {} is valid".format(ele))
#                     cur_progress = ele.weight
#                 else:
#                     cur_progress = 0
#                 return valid, cur_progress

#             elif isinstance(ele, Clause):
#                 logical_op = ele.logical_op
#                 results = []
#                 child_progresses = []
#                 for i in ele.elements:
#                     valid, child_progress = _divide(i)
#                     results.append(valid)
#                     child_progresses.append(child_progress)

#                 if logical_op == 'and':
#                     valid = all(results)
#                 elif logical_op == 'or':
#                     valid = any(results)
#                 else:
#                     raise ValueError

#                 if valid:
#                     printf("Clause: {} is valid".format(ele))
#                     cur_progress = ele.weight
#                 else:
#                     cur_progress = sum(child_progresses)

#                 return valid, cur_progress
#             else:
#                 raise TypeError

#         valid, progress = _divide(self.goal)
#         progress /= self.num_nodes

#         return progress

#     def is_valid(self, ele, state: dict):
#         if ele.type == "relation":
#             edge = {"from_id": self._extract_id(ele.subject), "relation_type": ele.relation, "to_id": self._extract_id(ele.object)}
#             return edge in state["edges"]
#         elif ele.type == "state":
#             state_node = ele.state
#             node_id = self._extract_id(ele.object)
#             # find node
#             for node in state["nodes"]:
#                 if node["id"] == node_id:
#                     return state_node in node["states"]
        
#             print("Node {} not found!".format(ele.object))
#             return False        # node not found
#         else:
#             raise ValueError

#     def _extract_id(self, o):
#         m = re.search(r'\[(.+?)\]', o)
#         id = int(m.group(1))
#         return id

#     # Parse the propositional string into Propositional Directed Acylic Graph (PDAG)
#     def _parse_goal_str(self, goal_str):
        
#         stack = []
#         clauses = []
#         predicates = []
        
#         for i, c in enumerate(goal_str):
#             if c == '(':
#                 stack.append(i)
#             elif c == ')':
#                 start = stack.pop()
#                 s = goal_str[start+1:i]

#                 if '(' not in s and ')' not in s:
#                     # leaf node
#                     s = s.split(' ')
#                     predicates.append(Predicate(s[0], *s[1:]))
#                 else:
#                     # non-leaf node
#                     elements = []

#                     for clause in clauses:
#                         if str(clause) in s:
#                             elements.append(clause)
#                             s = s.replace(str(clause), '')

#                     for predicate in predicates:
#                         if str(predicate) in s:
#                             elements.append(predicate)
#                             s = s.replace(str(predicate), '')

#                     logical_op = s.split(' ')[0]
                    
#                     clauses.append(Clause(logical_op, elements))

#         max_weight = len(clauses) + len(predicates)
#         self.num_nodes = max_weight
#         # ex: (ontop plate1 table13)
#         if len(clauses) == 0 and len(predicates) == 1:
#             clauses.append(predicates[0])
#             goal = predicates[0]
#         else:
#             goal = clauses[-1]

#         def _reassign_predicate_weight(ele, is_parent_or, parent_weight):

#             if isinstance(ele, Predicate):
#                 if is_parent_or:
#                     ele.weight = parent_weight

#             elif isinstance(ele, Clause):
#                 is_parent_or = True if ele.logical_op == 'or' else False
#                 parent_weight = ele.weight
#                 for i in ele.elements:
#                     _reassign_predicate_weight(i, is_parent_or, parent_weight)
#             else:
#                 raise TypeError

#         _reassign_predicate_weight(goal, True, max_weight)

#         return goal

class Clause():

    def __init__(self, logical_op, elements):
        self.logical_op = logical_op
        self.elements = elements
        self.weight = sum([element.weight for element in self.elements]) + 1

    def __str__(self):
        elements_str = []
        for ele in self.elements:
            elements_str.append('({})'.format(str(ele)))

        return '{} '.format(self.logical_op) + ' '.join(elements_str)


class Predicate():

    def __init__(self, a, b, c=None, value=True):

        self.weight = 1
        self.value = value
        if c is None:
            # ex: door open
            self.type = 'state'
            self.state = a
            self.object = b
        else:
            # ex: ontop plate3 table1
            self.type = 'relation'
            self.relation = a
            self.subject = b
            self.object = c

    def is_relation(self):
        return self.type == 'relation'

    def is_state(self):
        return self.type == 'state'

    def __str__(self):
        if self.type == 'state':
            if self.value:
                return '{} {}'.format(self.state, self.object)
            else:
                return 'not {} {}'.format(self.state, self.object)
        else:
            if self.value:
                return '{} {} {}'.format(self.relation, self.subject, self.object)
            else:
                return 'not {} {} {}'.format(self.relation, self.subject, self.object)



def _test_task_measurement():
    
    '''
    setup_table = Task('(or (and (ontop plate[1] table[13]) (ontop plate[3] table[13])) (ontop plate[4] table[13]))')
    print(setup_table.goal)
    graph = {
        "nodes": [{"class_name": "plate", "id": 1}, 
                  {"class_name": "plate", "id": 3}, 
                  {"class_name": "plate", "id": 4}, 
                  {"class_name": "table", "id": 13}, ], 
        "edges": [{"from_id": 4, "relation_type": "ontop", "to_id": 13}, {"from_id": 3, "relation_type": "ontop", "to_id": 13}]
    }
    '''
    task = Task('(and (and (ontop plate[1] table[10]) (ontop plate[2] table[10])) (or (ontop plate[3] table[10]) (ontop plate[4] table[10])))')
    print(task.goal)
    graph = {
        "nodes": [{"class_name": "plate", "id": 1}, 
                  {"class_name": "plate", "id": 2}, 
                  {"class_name": "plate", "id": 3}, 
                  {"class_name": "plate", "id": 4}, 
                  {"class_name": "table", "id": 10}, ], 
        "edges": [{"from_id": 4, "relation_type": "ontop", "to_id": 10}, {"from_id": 3, "relation_type": "ontop", "to_id": 10}, {"from_id": 1, "relation_type": "ontop", "to_id": 10}]
    }
    progress = task.measure_progress(graph)
    print(progress)


if __name__ == '__main__':
    import ipdb
    _test_task_measurement()
