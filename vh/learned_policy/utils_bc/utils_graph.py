import pdb
import numpy as np

def state_one_hot(vocabulary_node_state_word_index_dict, states):
    one_hot = np.zeros(len(vocabulary_node_state_word_index_dict))
    for state in states:
        one_hot[vocabulary_node_state_word_index_dict[state.lower()]] = 1
    return one_hot


def filter_redundant_nodes(each_agent_graph):
    valid_node_id = []
    new_each_agent_graph = []
    
    for node in each_agent_graph:
        if node['id'] not in valid_node_id:
            valid_node_id.append(node['id'])
            new_each_agent_graph.append(node)
                
    return new_each_agent_graph



