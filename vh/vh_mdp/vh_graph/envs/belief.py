import numpy as np
import random
from vh_sim.simulation.evolving_graph.utils import load_graph_dict, load_name_equivalence
from  vh_sim.simulation.evolving_graph.environment import EnvironmentState, EnvironmentGraph, GraphNode
import scipy.special
import pdb
import pdb
import sys
import  vh_sim.simulation.evolving_graph.utils as vh_utils
import json
import ipdb
import copy
from scipy.special import softmax



def get_graph_info(graph):
    id2node = {x['id']:x for x in graph['nodes']}
    name2node = {}
    name2id = {}
    for node in graph['nodes']:
        if node['class_name'] not in name2node:
            name2node[node['class_name']] = []
        name2node[node['class_name']].append(node)
    
        if node['class_name'] not in name2id:
            name2id[node['class_name']] = []
        name2id[node['class_name']].append(node['id'])

    return id2node, name2node, name2id

class Belief():
    def __init__(self, graph_gt, agent_id, prior=None, forget_rate=0.0, seed=None):
       
        # Possible beliefs for some objects
        self.container_restrictions = {
            'book': ['cabinet', 'kitchencabinet']
        }

        self.id_restrictions_inside = {

        }

        self.debug = False

        self.high_prob = 1e9
        self.low_prob = -1e9
        self.name_equivalence = load_name_equivalence()
        self.map_properties_to_pred = {
            'ON': ('on', True),
            'OPEN': ('open', True),
            'OFF': ('on', False),
            'CLOSED': ('open', False)
        }
        
        self.map_edges_to_pred = {
            'INSIDE': 'inside',
            'CLOSE': 'close',
            'ON': 'ontop',
            'FACING': 'facing'
        }
        self.house_obj = [
                'floor',
                'wall',
                'ceiling'
        ]

        self.class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain', 'window', 'clothespile']
        
        self.agent_id = agent_id
        self.grabbed_object = []


        self.categories_delete = ['Doors']

        self.graph_helper = vh_utils.graph_dict_helper()
        self.binary_variables = self.graph_helper.binary_variables
        self.graph_init = graph_gt

        self.sampled_graph = copy.deepcopy(graph_gt)

        self.states_consider = ['OFF', 'CLOSED']
        self.edges_consider = ['INSIDE', 'ON']
        self.node_to_state_belief = {}
        self.room_node = {}
        self.room_nodes = []
        self.container_ids = []
        self.surface_ids = []

        # Binary Variable Dict
        self.bin_var_dict = {}
        for bin_var in self.binary_variables:
            self.bin_var_dict[bin_var.negative] = [[bin_var.positive, bin_var.negative], 0]
            self.bin_var_dict[bin_var.positive] = [[bin_var.positive, bin_var.negative], 1]

        id2node = {}
        for x in graph_gt['nodes']:
            id2node[x['id']] = x

        # Door edges: Will be used to make the graph walkable
        self.door_edges = {}
        for edge in self.graph_init['edges']:
            if edge['relation_type'] == 'BETWEEN':
                if id2node[edge['from_id']]['category'] == 'Doors':
                    if edge['from_id'] not in self.door_edges.keys():
                        self.door_edges[edge['from_id']] = []
                    self.door_edges[edge['from_id']].append(edge['to_id'])

        # Assume that between 2 nodes there is only one edge
        self.edge_belief = {}
        self.init_belief()
        
        self.first_belief = copy.deepcopy(self.edge_belief) 
        self.first_room = copy.deepcopy(self.room_node) 
        self.rate = forget_rate

        self.ids_to_update_mess = []

    def update(self, origin, final):

        dist_total = origin - final
        ratio = (1 - np.exp(-self.rate*np.abs(origin-final)))
        return origin - ratio*dist_total

    def reset_to_priot_if_invalid(belief_node):
        # belief_node: [names, probs]
        if belief_node[1].max() == self.low_prob:

            belief_node[1] = prior

    def update_to_prior(self):
        for node_name in self.edge_belief:
            self.edge_belief[node_name]['INSIDE'][1] = self.update(self.edge_belief[node_name]['INSIDE'][1], self.first_belief[node_name]['INSIDE'][1])

        for node in self.room_node:
            self.room_node[node][1] = self.update(self.room_node[node][1], self.first_room[node][1])


    def _remove_house_obj(self, state):
        delete_ids = [x['id'] for x in state['nodes'] if x['class_name'].lower() in self.house_obj]
        state['nodes'] = [x for x in state['nodes'] if x['id'] not in delete_ids]
        state['edges'] = [x for x in state['edges'] if x['from_id'] not in delete_ids and x['to_id'] not in delete_ids]
        return state

    def init_belief(self):
        # Set belief on object states
        # for objects have state, init=0.5
        for node in self.sampled_graph['nodes']:
            object_name = node['class_name']
            bin_vars = self.graph_helper.get_object_binary_variables(object_name)
            bin_vars = [x for x in bin_vars if x.default in self.states_consider]
            belief_dict = {}
            for bin_var in bin_vars:
                belief_dict[bin_var.positive] = 0.5
            self.node_to_state_belief[node['id']] = belief_dict

        # TODO: ths class should simply have a surface property
        container_classes = [
        'bathroomcabinet',
        'kitchencabinet',
        'cabinet',
        'fridge',
        'stove',
        # 'kitchencounterdrawer',
        'dishwasher',
        'microwave']

        
        # Solve these cases
        objects_inside_something = set([edge['from_id'] for edge in self.sampled_graph['edges'] if edge['relation_type'] == 'INSIDE'])
        
        # Set belief for edges
        # TODO: this should be specified in the properties
        object_containers = [node for node in self.sampled_graph['nodes'] if node['class_name'] in container_classes]
        # object_containers = [node for node in self.sampled_graph['nodes'] if 'CAN_OPEN' in node['properties'] and 'CONTAINERS' in node['properties']]
        



        grabbable_nodes = [node for node in self.sampled_graph['nodes'] if 'GRABBABLE' in node['properties']]

        # TODO: this should be 0
        # grabbable_nodes = [node for node in grabbable_nodes if node['id'] in objects_inside_something]


        self.room_nodes = [node for node in self.sampled_graph['nodes'] if node['category'] == 'Rooms']


        self.room_ids = [x['id'] for x in self.room_nodes]
        container_ids = [x['id'] for x in object_containers]
        self.container_ids = [None] + container_ids
        self.room_index_belief_dict = {x: it for it, x in enumerate(self.room_ids)}

        self.container_index_belief_dict = {x: it for it, x in enumerate(self.container_ids) if x is not None}

        for obj_name in self.container_restrictions:
            possible_classes = self.container_restrictions[obj_name]
            # the ids that should not go here
            restricted_ids = [x['id'] for x in object_containers if x['class_name'] not in possible_classes]
            self.id_restrictions_inside[obj_name] = np.array([self.container_index_belief_dict[rid] for rid in restricted_ids])

        """TODO: better initial belief"""
        object_room_ids = {}
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE' and edge['to_id'] in self.room_ids:
                object_room_ids[edge['from_id']] = edge['to_id']

        
        nodes_inside_ids = [x['from_id'] for x in self.sampled_graph['edges'] if x['to_id'] not in self.room_ids and x['relation_type'] == 'INSIDE']
        nodes_inside = [node for node in self.sampled_graph['nodes'] if node['id'] in nodes_inside_ids and 'GRABBABLE' not in node['states']]

        objects_for_belief_reasoning = grabbable_nodes+nodes_inside
        # pdb.set_trace()
        for node in objects_for_belief_reasoning:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            id1 = node['id']
            self.edge_belief[id1] = {}
            

            init_values = np.ones(len(container_ids)+1)/(1.+len(container_ids))
            if node['class_name'] in self.id_restrictions_inside.keys():
                init_values[self.id_restrictions_inside[node['class_name']]] = self.low_prob

            # The probability of being inside itself is 0
            if id1 in container_ids:
                init_values[1+container_ids.index(id1)] = self.low_prob

            self.edge_belief[id1]['INSIDE'] = [[None]+container_ids, init_values]
        
        # Room belief. Will be used for nodes that are not in the belief
        for node in self.sampled_graph['nodes']:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            if node not in self.room_nodes:
                room_array = np.ones(len(self.room_ids))
                self.room_node[node['id']] = [self.room_ids, room_array]
        self.sampled_graph['edges'] = []

        


    def reset_belief(self):
        self.sampled_graph['edges'] = []
        self.init_belief()

    def sample_from_belief(self, as_vh_state=False, ids_update=None):
        id2node, name2node, name2id = get_graph_info(self.sampled_graph)


        # Sample states
        for node in self.sampled_graph['nodes']:
            if ids_update is not None and node['id'] not in ids_update:
                continue
            if node['id'] not in self.node_to_state_belief:
                continue
            belief_node = self.node_to_state_belief[node['id']]
            states = []
            for var_name, var_belief_value in belief_node.items():
                rand_number = random.random()
                value_binary = 1 if rand_number < var_belief_value else 0
                states.append(self.bin_var_dict[var_name][0][value_binary])
            node['states'] = states
        
        node_inside = {}
        object_grabbed = []


        # Sample edges
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                node_inside[edge['from_id']] = edge['to_id']

            if edge['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                object_grabbed.append(edge['to_id']) 


        for node in self.sampled_graph['nodes']:
            # if node['id'] in self.ids_to_update_mess:
            #     pdb.set_trace()

            if ids_update is not None and node['id'] not in ids_update:
                continue

            if node['id'] not in self.edge_belief:
                if node['id'] not in self.room_node.keys():
                    continue

                node_room_cands =  self.room_node[node['id']]
                node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))

                prob = softmax(node_room_cands[1])[node_room_cands[0].index(node_room)]
                
                # final_rel = (node_room, 'INSIDE', prob)
                final_rel = (node_room, 'INSIDE', 0.5) ## if assume the target is room, give prob=0.5

                
                # print('-----------------------------------')
                # tem2 = [tem['class_name'] for tem in self.sampled_graph['nodes'] if tem['id']==node['id']]
                # print(tem2)
                # print(final_rel, prob)
                # if tem2[0] in ['waterglass', 'wineglasss', 'plate', 'cutleryfork'] and prob>1-1e-5:
                #     pdb.set_trace()
                # print('-----------------------------------')
                

            else:
                edge_belief_inside = self.edge_belief[node['id']]['INSIDE']
                


                if node['id'] in node_inside:
                    # The relationships between unseen objects should stay the same
                    sample_inside = node_inside[node['id']]
                    
                    if sample_inside not in edge_belief_inside[0]:
                        ## if sample_inside is room and asked for update
                        assert sample_inside in self.room_ids 
                        sample_inside = np.random.choice(edge_belief_inside[0], p=scipy.special.softmax(edge_belief_inside[1]))
                        prob = softmax(edge_belief_inside[1])[edge_belief_inside[0].index(sample_inside)]
                    else:
                        ## keep the original edge
                        prob = softmax(edge_belief_inside[1])[edge_belief_inside[0].index(sample_inside)]

                else:
                    try:
                        sample_inside = np.random.choice(edge_belief_inside[0], p=scipy.special.softmax(edge_belief_inside[1]))
                        prob = softmax(edge_belief_inside[1])[edge_belief_inside[0].index(sample_inside)]
                    
                    except:
                        print('Error with {}'.format(node['id']))
                        pdb.set_trace()






                if sample_inside is None:
                    # Sample in a room
                    node_room_cands =  self.room_node[node['id']]
                    node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                    prob = softmax(node_room_cands[1])[node_room_cands[0].index(node_room)]
                    
                    # final_rel = (node_room, 'INSIDE', prob)
                    final_rel = (node_room, 'INSIDE', 0.5) ## if assume the target is room, give prob=0.5

                    # print('-----------------------------------')
                    # tem2 = [tem['class_name'] for tem in self.sampled_graph['nodes'] if tem['id']==node['id']]
                    # print(tem2)
                    # print(final_rel, prob)
                    # if tem2[0] in ['waterglass', 'wineglasss', 'plate', 'cutleryfork'] and prob>1-1e-5:
                    #     pdb.set_trace()
                    # print('-----------------------------------')
                    
                
                else:
                    if sample_inside == node['id']:
                        pass
                    final_rel = (sample_inside, 'INSIDE', prob)

            if final_rel[1] == 'INSIDE':
                node_inside[node['id']] = final_rel[0]
            new_edge = {'from_id': node['id'], 'to_id': final_rel[0], 'relation_type': final_rel[1], 'prob': final_rel[2]}
            self.sampled_graph['edges'].append(new_edge)

        

        # Include the doors
        # if ids_update is None:
        for node_door in self.door_edges.keys():
            node_1, node_2 = self.door_edges[node_door]
            self.sampled_graph['edges'].append({'to_id': node_1, 'from_id': node_door, 'relation_type': 'BETWEEN', 'prob': 1.})
            self.sampled_graph['edges'].append({'to_id': node_2, 'from_id': node_door, 'relation_type': 'BETWEEN', 'prob': 1.})

        if as_vh_state:
            return self.to_vh_state(self.sampled_graph)
        return self.sampled_graph

    def to_vh_state(self, graph):
        state = self._remove_house_obj(graph)
        vh_state = EnvironmentState(EnvironmentGraph(state), 
                                    self.name_equivalence, instance_selection=True)
        return vh_state

    def canopen_and_open(self, node):
        return 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']

    def is_surface(self, node):
        return 'SURFACE' in node['properties']



    def update_graph_from_gt_graph(self, gt_graph, opponent_messsage=None):
        """
        Updates the current sampled graph with a set of observations
        """
        for edge in gt_graph['edges']:
            if 'prob' in edge:
                ipdb.set_trace()

        # Here we have a graph sampled from our belief, and want to update it with gt graph
        id2node = {}
        for x in gt_graph['nodes']:
            id2node[x['id']] = x

        ids_to_update = []
        self.update_to_prior()
        
        if opponent_messsage is not None and len(opponent_messsage)>0:
            ids_to_update = self.update_from_messsage(gt_graph, opponent_messsage, ids_to_update)

        self.update_from_gt_graph(gt_graph)


        char_node = self.agent_id


        inside = {}
        for x in gt_graph['edges']:
            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'], id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])
                    raise Exception

                inside[x['from_id']] = x['to_id']

        for node in self.sampled_graph['nodes']:
            if node['id'] in id2node.keys():
                states_graph_old = id2node[node['id']]['states']
                object_name = id2node[node['id']]['class_name']
                bin_vars = self.graph_helper.get_object_binary_variables(object_name)
                bin_vars = [x for x in bin_vars if x.default in self.states_consider]
                bin_vars_missing = [x for x in bin_vars if x.positive not in states_graph_old and x.negative not in states_graph_old]
                states_graph = states_graph_old + [x.default for x in bin_vars_missing]
                # fill out the rest of info regarding the states
                node['states'] = states_graph
                id2node[node['id']]['states'] = states_graph



        edges_keep = []
        
        for edge in self.sampled_graph['edges']:

            if (edge['from_id'] == char_node and edge['relation_type'] == 'INSIDE'):
                continue

            # Grabbed objects we don't need to keep them
            if edge['from_id'] == char_node and 'HOLD' in edge['relation_type']:
                continue

            # If the object should be visible but it is not in the observation, remove close relationship
            if (edge['from_id'] == char_node or edge['to_id'] == char_node) and edge['relation_type'] == 'CLOSE':
                continue

            # Objects that are visible, we do not care anymore
            if edge['from_id'] in id2node.keys() and edge['from_id'] != char_node:
                continue

            # The object is not visible but the container is visible

            ## false belief
            if edge['to_id'] in id2node.keys() and edge['to_id'] != char_node:
                # If it is a room and we have not seen it, the belief remains
                ## node should appear but not appear in the new observation
                if id2node[edge['to_id']]['category'] == 'Rooms' and edge['relation_type'] == 'INSIDE':
                    if inside[char_node] == edge['to_id']:
                        if edge['from_id'] not in id2node.keys():
                            ids_to_update.append(edge['from_id'])
                        else:
                            pass
                        continue
                else:
                    if edge['relation_type'] == 'ON':
                        ids_to_update.append(edge['from_id'])
                        continue
                    if edge['relation_type'] == 'INSIDE' and 'OPEN' in id2node[edge['to_id']]['states']:
                        ids_to_update.append(edge['from_id'])
                        continue
            edges_keep.append(edge)

        ## add the probability of edges
        for edge in gt_graph['edges']:
            try:
                assert 'prob' not in edge
            except:
                ipdb.set_trace()
            edge['prob'] = 1.
        for edge in edges_keep:
            assert 'prob' in edge
        self.sampled_graph['edges'] = edges_keep + gt_graph['edges']
        

            

        # For objects that are inside in the belief, character should also be close to those, so that when we open the object
        # we are already close to what is inside

        nodes_close = [x['to_id'] for x in gt_graph['edges'] if x['from_id'] == char_node and x['relation_type'] == 'CLOSE']
        inside_belief = {}
        for edge in edges_keep:
            if edge['relation_type'] == 'INSIDE':
                if edge['to_id'] not in inside_belief: inside_belief[edge['to_id']] = []
                inside_belief[edge['to_id']].append(edge['from_id'])

        for node in nodes_close:
            if node not in inside_belief:
                continue

            for node_inside in inside_belief[node]:
                close_edges = [
                        {'from_id': char_node, 'to_id': node_inside, 'relation_type': 'CLOSE', 'prob': 1.},
                        {'to_id': char_node, 'from_id': node_inside, 'relation_type': 'CLOSE', 'prob': 1.}
                ]
                self.sampled_graph['edges'] += close_edges


        # if len([edge for edge in self.sampled_graph['edges'] if edge['from_id']==458 and edge['prob']>0.9 and edge['relation_type'] in ['INSIDE', 'ON']])>0:
        #     pdb.set_trace()

        # sample new edges that have not been seen
        self.sample_from_belief(ids_update=ids_to_update)
        
        return self.sampled_graph
    



    def update_from_messsage(self, gt_graph, opponent_messsage, ids_to_update):
        self.ids_to_update_mess = []
        id2node, name2node, name2id = get_graph_info(self.sampled_graph)
        
    
        for message in opponent_messsage:
            if 'ans_env_yes' in message:
                message = message['ans_env_yes'].split(' | ')
                
                ## based on the id
                obj_id_dict = {}
                for tem in message:
                    obj_id = int(tem.split(' ')[1][1:-1])
                    obj_name = tem.split(' ')[0]
                    rel = tem.split(' ')[3]
                    tar_name = tem.split(' ')[-2]
                    tar_id = int(tem.split(' ')[-1][1:-1])
                    
                    if obj_id not in obj_id_dict:
                        obj_id_dict[obj_id] = []
                    
                    obj_id_dict[obj_id].append([obj_name, rel, tar_name, tar_id])


                for obj_id,_ in obj_id_dict.items():
                    edges = obj_id_dict[obj_id]
                    for edge in edges:
                        obj_name, rel, tar_name, tar_id = edge
                    
                        # if obj_id == 235:
                        #     pdb.set_trace()

                        if obj_id not in self.edge_belief:
                            new_edge = {'from_id': obj_id, 'to_id': tar_id, 'relation_type': rel.upper(), 'prob': 1.}
                            self.sampled_graph['edges'].append(new_edge)

                        else:
                            if tar_id not in self.edge_belief[obj_id]['INSIDE'][0]:
                                ## tar_id is a room
                                assert tar_id in self.room_ids
                                prob = softmax(self.room_node[obj_id][1])


                                ## if very certain about one object, do not update
                                if prob.max()<1.0-1e-4:
                                    self.room_node[obj_id][1][:] = self.low_prob
                                    self.room_node[obj_id][1][self.room_index_belief_dict[tar_id]] = 1.
                                    self.edge_belief[obj_id]['INSIDE'][1][:] = self.low_prob
                                    self.edge_belief[obj_id]['INSIDE'][1][0] = 1.


                                if obj_id not in ids_to_update:
                                    ids_to_update.append(obj_id)
                                
                                # if obj_id not in self.ids_to_update_mess:
                                #     self.ids_to_update_mess.append(obj_id)
                            else:
                                assert tar_id not in self.room_ids
                                prob = softmax(self.edge_belief[obj_id]['INSIDE'][1])
                                ## if very certain about one object, do not update
                                if prob.max()<1.0-1e-4:
                                    index_inside = self.container_index_belief_dict[tar_id]

                                    self.edge_belief[obj_id]['INSIDE'][1][:] = self.low_prob
                                    self.edge_belief[obj_id]['INSIDE'][1][index_inside] = 1.
                                
                                if obj_id not in ids_to_update:
                                    ids_to_update.append(obj_id)

                                # if obj_id not in self.ids_to_update_mess:
                                #     self.ids_to_update_mess.append(obj_id)


                            ## check if this object is in the sampled graph
                            obj_edge_certain = [edge for edge in self.sampled_graph['edges'] if edge['from_id']==obj_id if edge['prob']>1-1e-4 and edge['relation_type'] in ['INSIDE', 'ON']]
                            obj_edge_uncertain = [edge for edge in self.sampled_graph['edges'] if edge['from_id']==obj_id if edge['prob']<1-1e-4]

                            if len(obj_edge_certain)==0:
                                if tar_id not in self.edge_belief[obj_id]['INSIDE'][0]:
                                    ## tar_id is a room
                                    assert tar_id in self.room_ids
                                    node_room_cands =  self.room_node[obj_id]
                                    sampled_node = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                                    prob = softmax(node_room_cands[1])[node_room_cands[0].index(sampled_node)]

                                else:
                                    assert tar_id not in self.room_ids
                                    edge_belief_inside = self.edge_belief[obj_id]['INSIDE']
                                    sampled_node = np.random.choice(edge_belief_inside[0], p=scipy.special.softmax(edge_belief_inside[1]))
                                    prob = softmax(edge_belief_inside[1])[edge_belief_inside[0].index(sampled_node)]
                        
                                new_edge = {'from_id': obj_id, 'to_id': sampled_node, 'relation_type': rel.upper(), 'prob': prob}
                                self.sampled_graph['edges'].append(new_edge)

                                ## remove uncertain edges
                                edge_keep = []
                                for edge in self.sampled_graph['edges']:
                                    if edge not in obj_edge_uncertain:
                                        edge_keep.append(edge)
                                self.sampled_graph['edges'] = edge_keep

                            else:
                                pass
                                
        return ids_to_update         


    def update_from_gt_graph(self, gt_graph):
       
        # Update the states of nodes that we can see in the belief. Note that this does not change the sampled graph
        id2node = {}
        for x in gt_graph['nodes']:
            id2node[x['id']] = x
       
        inside = {}
        grabbed_object = []

        ## objects in current observation (current room)
        for x in gt_graph['edges']:
            if x['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                grabbed_object.append(x['to_id'])

            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'], id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])

                    raise Exception

                inside[x['from_id']] = x['to_id']
                

        visible_ids = [x['id'] for x in gt_graph['nodes']]
        edge_tuples = [(x['from_id'], x['to_id']) for x in gt_graph['edges']]

        ## update node states
        for x in gt_graph['nodes']:
            try:
                dict_state = self.node_to_state_belief[x['id']]
                for state in x['states']:
                    pred_name = self.bin_var_dict[state][0][0]
                    dict_state[pred_name] = self.bin_var_dict[state][1]
            except:
                pass
        
        char_node = self.agent_id

        visible_room = inside[char_node]

        
        deleted_edges = []
        id_updated = []

        # Keep track of things with impossible belief
        # Objects and rooms we are just seeing
        ids_known_info = [self.room_index_belief_dict[visible_room], []]

        

        for id_node in self.edge_belief.keys():
            id_updated.append(id_node)
            if id_node in grabbed_object:
                continue

            if id_node in visible_ids:
                # TODO: what happens when object grabbed
                assert(id_node in inside.keys())
                inside_obj = inside[id_node]
                # If object is inside a room, for sure it is not insde another object

                # Some objects have the relationship inside but they are not part of the belief because
                # they are visible anyways like bookshelf. In that case we consider them to just be
                # inside the room
                if inside_obj not in self.room_ids and inside_obj not in self.container_index_belief_dict:
                    inside_obj = inside[inside_obj]

                if inside_obj in self.room_ids:
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][0] = 1.
                    self.room_node[id_node][1][:] = self.low_prob
                    self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = 1.

                    # if id_node in id2node:
                    #     print(783, id_node, id2node[id_node]['class_name'])
                    # else:
                    #     print(783, id_node)

                else:
                    # If object is inside an object, for sure it is not insde another object
                    try:
                        index_inside = self.container_index_belief_dict[inside_obj]
                    except:
                        ipdb.set_trace()
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][index_inside] = 1.
                
            else:
                # If not visible. for sure not in this room
                self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob
                if (self.room_node[id_node][1] > self.low_prob).sum() == 0:
                    if id_node in self.edge_belief:
                        # If not in any room, needs to be inside something
                        self.edge_belief[id_node]['INSIDE'][1][0] = self.low_prob

            
            for id_node in self.container_ids:
                if id_node in visible_ids and id_node in self.container_ids and 'OPEN' in id2node[id_node]['states']:
                    for id_node_child in self.edge_belief.keys():
                        if id_node_child not in inside.keys() or inside[id_node_child] != id_node:
                            ids_known_info[1].append(self.container_index_belief_dict[id_node])
                            self.edge_belief[id_node_child]['INSIDE'][1][self.container_index_belief_dict[id_node]] = self.low_prob
                    
        
        # Some furniture has no edges, only has info about inside rooms
        # We need to udpate its location
        for id_node in self.room_node.keys():
            if id_node not in id_updated:
                if id_node in visible_ids:
                    inside_obj = inside[id_node]
                    if inside_obj == visible_room:
                        self.room_node[id_node][1][:]  = self.low_prob
                        self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = 1.

                        # if id_node in id2node:
                        #     print(817, id_node, id2node[id_node]['class_name'])
                        # else:
                        #     print(817, id_node)

                    else:
                        assert('Error: A grabbable object is inside something else than a room')
                else:
                    # Either the node goes inside somehting in the room... or ti should not be
                    # in this room
                    self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob

                    # if id_node in id2node:
                    #     print(825, id_node, id2node[id_node]['class_name'])
                    # else:
                    #     print(825, id_node)


        mask_house = np.ones(len(self.room_nodes))
        mask_obj = np.ones(len(self.container_ids))
        mask_house[ids_known_info[0]] = 0

        assert (len(self.room_nodes) > 0)
        if len(ids_known_info[1]):
            mask_obj[np.array(ids_known_info[1])] = 0

        mask_obj = (mask_obj == 1)
        mask_house = (mask_house == 1)
        # Check for impossible beliefs
        for id_node in self.room_node.keys():
            if id_node in self.edge_belief.keys():
                # the object should be in a room or inside something
                if np.max(self.edge_belief[id_node]['INSIDE'][1]) == self.low_prob:
                    # Sample locations except for marked ones
                    self.edge_belief[id_node]['INSIDE'][1] = self.first_belief[id_node]['INSIDE'][1]
                    # Sample rooms except marked
                    try:
                        self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]
                    except:
                        pdb.set_trace()
            else:
                if np.max(self.room_node[id_node][1]) == self.low_prob:
                    self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]








if __name__ == '__main__':
    graph_init = '../../example_graph/example_graph.json' 
    with open(graph_init, 'r') as f:
        graph = json.load(f)['init_graph']
    Belief(graph)
