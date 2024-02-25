import pdb
import copy
import random
import numpy as np

def inside_not_trans(graph):
    id2node = {node['id']: node for node in graph['nodes']}
    parents = {}
    grabbed_objs = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':

            if edge['from_id'] not in parents:
                parents[edge['from_id']] = [edge['to_id']]
            else:
                parents[edge['from_id']] += [edge['to_id']]
        elif edge['relation_type'].startswith('HOLDS'):
            grabbed_objs.append(edge['to_id'])

    edges = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
            if len(parents[edge['from_id']]) == 1:
                edges.append(edge)
            else:
                if edge['from_id'] > 1000:
                    pdb.set_trace()
        else:
            edges.append(edge)
    graph['edges'] = edges

    # # add missed edges
    # missed_edges = []
    # for obj_id, action in self.obj2action.items():
    #     elements = action.split(' ')
    #     if elements[0] == '[putback]':
    #         surface_id = int(elements[-1][1:-1])
    #         found = False
    #         for edge in edges:
    #             if edge['relation_type'] == 'ON' and edge['from_id'] == obj_id and edge['to_id'] == surface_id:
    #                 found = True
    #                 break
    #         if not found:
    #             missed_edges.append({'from_id': obj_id, 'relation_type': 'ON', 'to_id': surface_id})
    # graph['edges'] += missed_edges

    parent_for_node = {}

    char_close = {1: [], 2: []}
    for char_id in range(1, 3):
        for edge in graph['edges']:
            if edge['relation_type'] == 'CLOSE':
                if edge['from_id'] == char_id and edge['to_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['to_id'])
                elif edge['to_id'] == char_id and edge['from_id'] not in char_close[char_id]:
                    char_close[char_id].append(edge['from_id'])
    ## Check that each node has at most one parent
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
                print('{} has > 1 parent'.format(edge['from_id']))
                pdb.set_trace()
                raise Exception
            parent_for_node[edge['from_id']] = edge['to_id']
            # add close edge between objects in a container and the character
            if id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave',
                                                        'dishwasher', 'stove']:
                for char_id in range(1, 3):
                    if edge['to_id'] in char_close[char_id] and edge['from_id'] not in char_close[char_id]:
                        graph['edges'].append({
                            'from_id': edge['from_id'],
                            'relation_type': 'CLOSE',
                            'to_id': char_id
                        })
                        graph['edges'].append({
                            'from_id': char_id,
                            'relation_type': 'CLOSE',
                            'to_id': edge['from_id']
                        })

    ## Check that all nodes except rooms have one parent
    nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
    nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
    nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
    if len(nodes_without_parent) > 0:

        for nd in nodes_without_parent:
            print(id2node[nd])
        
        # pdb.set_trace()
        raise Exception

    return graph




def convert_action(num_agents, action_dict):
    

    # if num_agents==1:
    #     script = action_dict[0]
    #     current_script = ['<char{}> {}'.format(agent_id, script)]
    #     script_list = [x + '|' + y if len(x) > 0 else y for x, y in zip(script_list, current_script)]
    #     return script_list


    agent_do = [item for item, action in action_dict.items() if action is not None]
    # Make sure only one agent interact with the same object
    if len(action_dict.keys()) > 1:
        # if None not in list(action_dict.values()) and sum(['walk' in x for x in action_dict.values()]) < 2:
        # if None not in list(action_dict.values()) and sum(['grab' in x for x in action_dict.values()]) == 2:
        # if None not in list(action_dict.values()) and sum(['walk' in x for x in action_dict.values()]) == 0:
        if None not in list(action_dict.values()) and sum(['walk' in x for x in action_dict.values()]) < 2:
            # continue
            objects_interaction = [x.split('(')[1].split(')')[0] for x in action_dict.values()]
            if len(set(objects_interaction)) == 1:
                # agent_do = [random.choice([0,1])]
                agent_do = [0] # always select the first agent


    script_list = ['']

    for agent_id in agent_do:
        script = action_dict[agent_id]
        if script is None:
            continue
        current_script = ['<char{}> {}'.format(agent_id, script)]

        script_list = [x + '|' + y if len(x) > 0 else y for x, y in zip(script_list, current_script)]

    # if self.follow:
    #script_list = [x.replace('[walk]', '[walktowards]') for x in script_list]
    # script_all = script_list

    return script_list



def convert_action_v1(action_dict):
    script_lists = []
    # assert len(action_dict)==2
    
    # Make sure only one agent interact with the same object
    if None not in list(action_dict.values()) and sum(['walk' in x for x in action_dict.values()]) < 2:
        objects_interaction = [x.split('(')[1].split(')')[0] for x in action_dict.values()]
        
        # if len(set(objects_interaction)) == 1:
        #     agent_do = random.choice([0,1])
        #     if agent_do==0:
        #         current_script = ['<char{}> {}'.format(agent_do, action_dict[agent_do])]
        #         # current_script = [x.replace('[walk]', '[walkto]') for x in current_script]
        #         script_lists.append(current_script)
        #         script_lists.append(None)
        #     else:
        #         script_lists.append(None)
        #         current_script = ['<char{}> {}'.format(agent_do, action_dict[agent_do])]
        #         # current_script = [x.replace('[walk]', '[walkto]') for x in current_script]
        #         script_lists.append(current_script)
        # else:
        #     for agent_id in action_dict:
        #         current_script = ['<char{}> {}'.format(agent_id, action_dict[agent_id])]
        #         # current_script = [x.replace('[walk]', '[walkto]') for x in current_script]
        #         script_lists.append(current_script)

        for agent_id in action_dict:
            current_script = ['<char{}> {}'.format(agent_id, action_dict[agent_id])]
            # current_script = [x.replace('[walk]', '[walkto]') for x in current_script]
            script_lists.append(current_script)

    else:
        for agent_id in action_dict:
            if action_dict[agent_id] is not None:
                current_script = ['<char{}> {}'.format(agent_id, action_dict[agent_id])]
                # current_script = [x.replace('[walk]', '[walkto]') for x in current_script]
                script_lists.append(current_script)
            else:
                script_lists.append(None)

    return script_lists
    


# def separate_new_ids_graph(graph, max_id):
#     new_graph = copy.deepcopy(graph)
#     for node in new_graph['nodes']:
#         if node['id'] > max_id:
#             node['id'] = node['id'] - max_id + 1000
#     for edge in new_graph['edges']:
#         if edge['from_id'] > max_id:
#             edge['from_id'] = edge['from_id'] - max_id + 1000
#         if edge['to_id'] > max_id:
#             edge['to_id'] = edge['to_id'] - max_id + 1000
#     return new_graph



def separate_new_ids_graph(graph, max_id):
    new_graph = copy.deepcopy(graph)
    change_ids = {}
    delete = []
    objects_issues = ['cutleryfork', 'wineglass', 'juice', 'waterglass', 'pudding', 'apple', 'poundcake', 'cupcake', 'plate']
    max_id_graph = max([node['id'] for node in graph['nodes']])
    
    for node in new_graph['nodes']:
        if node['id'] > max_id or node['class_name'] in ['apple', 'lime', 'plum', 'powersocket', 'pillow', 'dishbowl', 'wallpictureframe', 'radio']:
            if node['id'] > max_id:
                new_id = node['id'] - max_id + 1000
            else:
                delete.append(node['id'])
                new_id = None
                continue
                # new_id = max_id_graph + node['id'] + 1000
            change_ids[node['id']] = new_id
            node['id'] = new_id

        if node['class_name'] in objects_issues:
            node['obj_transform']['position'][1] += 0.03
            node['bounding_box']['center'][1] += 0.03

            num1 = random.random() - 0.5
            num2 = random.random() - 0.5
            num = [num1, num2]
            node['obj_transform']['position'][0] += (num[0] * 0.05)
            node['bounding_box']['center'][0] += (num[0] * 0.05)

            node['obj_transform']['position'][2] += (num[1] * 0.05)
            node['bounding_box']['center'][2] += (num[1] * 0.05)
    for edge in new_graph['edges']:
        if edge['from_id'] > max_id:
            edge['from_id'] = edge['from_id'] - max_id + 1000
        if edge['to_id'] > max_id:
            edge['to_id'] = edge['to_id'] - max_id + 1000
    new_graph['nodes'] = [node for node in new_graph['nodes'] if node['id'] not in delete]
    new_graph['edges'] = [edge for edge in new_graph['edges'] if edge['from_id'] not in delete and edge['to_id'] not in delete]
    return new_graph

    
def check_progress(state, task_goal):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    reward = 0.
    id2node = {node['id']: node for node in state['nodes']}

    if len(task_goal)>0:
        goal_spec = {goal_k: [goal_c, True, 2] for goal_k, goal_c in task_goal.items()}
    else:
        return None, None

    for key, value in goal_spec.items():

        elements = key.split('_')
        unsatisfied[key] = value[0] if elements[0].lower() not in ['offon', 'offinside'] else 0
        satisfied[key] = []
        for edge in state['edges']:

            if 'prob' in edge:
                if edge['prob'] < 1-1e-4:
                    continue
            # else:
            #     print('no prob')
            #     continue
                

            if elements[0].lower() in 'close':
                if edge['relation_type'].lower().startswith('close') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            if elements[0].lower() in ['on', 'inside']:
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0].lower() == 'offon':
                if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[0].lower() == 'offinside':
                if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[0].lower() == 'holds':
                if edge['relation_type'].lower().startswith('holds') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0].lower() == 'sit':
                if edge['relation_type'].lower().startswith('sit') and edge['to_id'] == int(elements[2]) and edge['from_id'] == int(elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
        

        if elements[0].lower() == 'turnon':
            if 'ON' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1

        if elements[0].lower() == 'turnoff':
            if 'OFF' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1

        if elements[0].lower() == 'open':
            if 'OPEN' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1
        
        if elements[0].lower() == 'closed':
            if 'CLOSED' in id2node[int(elements[1])]['states']:
                predicate = '{}_{}_{}'.format(elements[0], elements[1], 1)
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1

    return satisfied, unsatisfied



def check_progress_action_put(cur_graph, obj_name, tar_name, subgoal):
    obj_nodes = [node for node in cur_graph['nodes'] if node['class_name'] == obj_name]
    tar_nodes = [node for node in cur_graph['nodes'] if node['class_name'] == tar_name]
    obj_ids = [node['id'] for node in obj_nodes]
    tar_ids = [node['id'] for node in tar_nodes]

    if 'on ' in subgoal:
        obj_tar_edges = [edge for edge in cur_graph['edges'] if edge['from_id'] in obj_ids and edge['to_id'] in tar_ids and edge['relation_type']=='ON']
    elif 'inside ' in subgoal:
        obj_tar_edges = [edge for edge in cur_graph['edges'] if edge['from_id'] in obj_ids and edge['to_id'] in tar_ids and edge['relation_type']=='INSIDE']
    else:
        pdb.set_trace()
    return obj_tar_edges

def check_progress_action_open(cur_graph, obj_name, subgoal):
    obj_nodes = [node for node in cur_graph['nodes'] if node['class_name'] == obj_name]
    obj_nodes_open = [node for node in obj_nodes if 'OPEN' in node['states']]
    return obj_nodes_open

def check_progress_action_grab(cur_graph, obj_name, subgoal):
    obj_nodes = [node for node in cur_graph['nodes'] if node['class_name'] == obj_name]
    obj_ids = [node['id'] for node in obj_nodes]
    agent_hold_edge = [edge for edge in cur_graph['edges'] if (edge['from_id'] == 1 or edge['to_id'] == 1) and 'HOLD' in edge['relation_type']]
    agent_hold_obj_edge = [edge for edge in agent_hold_edge if edge['from_id'] in obj_ids or edge['to_id'] in obj_ids]
    return agent_hold_obj_edge


def check_progress_language(init_graph, cur_graph, language):
    satisfied = []
    unsatisfied = []
    done = False
    reward = 0.

    count = 1
    subgoals = language.split(',')
    for subgoal in subgoals:
        if 'put ' in subgoal:
            subgoal_split = subgoal.split(' ')
            obj_name = subgoal_split[1]
            tar_name = subgoal_split[4]

            init_obj_tar_edges = check_progress_action_put(init_graph, obj_name, tar_name, subgoal)
            cur_obj_tar_edges = check_progress_action_put(cur_graph, obj_name, tar_name, subgoal)

            obj_tar_edges = []
            for edge in cur_obj_tar_edges:
                overlap_edge = [edge2 for edge2 in init_obj_tar_edges if (edge['from_id']==edge2['from_id'] and edge['to_id']==edge2['to_id'] and edge['relation_type']==edge2['relation_type'])]
                if len(overlap_edge)==0:
                    obj_tar_edges.append(edge)
            
            reward += np.min([len(obj_tar_edges), count])
            if len(obj_tar_edges)==count:
                satisfied.append(subgoal)
            else:
                unsatisfied.append(subgoal)

        elif 'open ' in subgoal:
            subgoal_split = subgoal.split(' ')
            obj_name = subgoal_split[1]

            init_obj_nodes_open = check_progress_action_open(init_graph, obj_name, subgoal)
            cur_obj_nodes_open = check_progress_action_open(cur_graph, obj_name, subgoal)

            obj_nodes_open = []
            for node in cur_obj_nodes_open:
                overlap_node = [node2 for node2 in init_obj_nodes_open if node['id']==node2['id']]
                if len(overlap_node)==0:
                    obj_nodes_open.append(node)
            
            reward += np.min([len(obj_nodes_open), count])
            if len(obj_nodes_open)==count:
                satisfied.append(subgoal)
            else:
                unsatisfied.append(subgoal)

        elif 'grab ' in subgoal:
            subgoal_split = subgoal.split(' ')
            obj_name = subgoal_split[1]

            init_agent_hold_edge = check_progress_action_grab(init_graph, obj_name, subgoal)
            cur_agent_hold_edge = check_progress_action_grab(cur_graph, obj_name, subgoal)

            agent_hold_edge = []
            for edge in cur_agent_hold_edge:
                overlap_edge = [edge2 for edge2 in init_agent_hold_edge if (edge['from_id']==edge2['from_id'] and edge['to_id']==edge2['to_id'] and edge['relation_type']==edge2['relation_type'])]
                if len(overlap_edge)==0:
                    agent_hold_edge.append(edge)
            
            reward += np.min([len(agent_hold_edge), count])
            if len(agent_hold_edge)==count:
                satisfied.append(subgoal)
            else:
                unsatisfied.append(subgoal)
        
        elif 'move to' in subgoal:
            subgoal_split = subgoal.split(' ')
            obj_name = subgoal_split[2]

            obj_nodes = [node for node in cur_graph['nodes'] if node['class_name'] == obj_name]
            obj_ids = [node['id'] for node in obj_nodes]

            agent_edge = [edge for edge in cur_graph['edges'] if edge['from_id']==1 or edge['to_id']==1]
            agent_room_edge = [edge for edge in agent_edge if edge['from_id'] in obj_ids or edge['to_id'] in obj_ids]

            reward += np.min([len(agent_room_edge), count])
            if len(agent_room_edge)==count:
                satisfied.append(subgoal)
            else:
                unsatisfied.append(subgoal)

        else:
            pdb.set_trace()

    if len(satisfied)==len(subgoals):
        done = True

    return satisfied, unsatisfied, done, reward












