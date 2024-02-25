import pickle as pkl
import json
from tqdm import tqdm
import glob
import pdb
import argparse
import os
import pickle 
import numpy as np
import shutil
import re


def check_env_bug_step(correct_graph_flag, script_list, obj_name, obj_id, graph, agent_i, fix_edge=False, opponent_agent_action=None):

    if 'walk' in script_list:
        if obj_name not in ['livingroom', 'bedroom', 'kitchen', 'bathroom']:
            agent_edges = [edge for edge in graph['edges'] if edge['from_id']==agent_i+1 or edge['to_id']==agent_i+1]
            agent_obj_edges = [edge for edge in agent_edges if edge['from_id']==obj_id or edge['to_id']==obj_id]
            agent_obj_edges = [edge for edge in agent_obj_edges if edge['relation_type']=='CLOSE']

            if len(agent_obj_edges)<2:
                print('agent walk object error, no close edge %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add edge (%d %s %d)' % (agent_i+1, 'close', obj_id) )
                    graph['edges'].append({'from_id': agent_i+1, 'to_id': obj_id, 'relation_type': 'CLOSE'})
                    graph['edges'].append({'from_id': obj_id, 'to_id': agent_i+1, 'relation_type': 'CLOSE'})
        else:
            agent_obj_edges = [edge for edge in graph['edges'] if edge['from_id']==agent_i+1 and edge['to_id']==obj_id and edge['relation_type']=='INSIDE']
            agent_obj_edges_all = [edge for edge in graph['edges'] if edge['from_id']==agent_i+1 and edge['relation_type']=='INSIDE']

            if len(agent_obj_edges)==0:
                print('agent walk room error, no inside edge %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add edge (%d %s %d)' % (agent_i+1, 'inside', obj_id) )
                    graph['edges'].append({'from_id': agent_i+1, 'to_id': obj_id, 'relation_type': 'INSIDE'})

                    if len(agent_obj_edges_all)>0:
                        for tem in agent_obj_edges_all:
                            del graph['edges'][graph['edges'].index(tem)]



    
    elif 'grab' in script_list:
        agent_edges = [edge for edge in graph['edges'] if edge['from_id']==agent_i+1 or edge['to_id']==agent_i+1]
        agent_obj_edges = [edge for edge in agent_edges if edge['from_id']==obj_id or edge['to_id']==obj_id]
        agent_obj_edges = [edge for edge in agent_obj_edges if 'HOLD' in edge['relation_type']]

        # if both agents grab the same object, already fixed in the script, no need to check here
        if opponent_agent_action is not None:
            if len(agent_obj_edges)==0 and str(obj_id) not in opponent_agent_action:
                print('agent grab object error, no hold edge %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add edge (%d %s %d)' % (agent_i+1, 'hold', obj_id) )
                    graph['edges'].append({'from_id': agent_i+1, 'to_id': obj_id, 'relation_type': 'HOLDS_RH'})

        else:
            if len(agent_obj_edges)==0:
                print('agent grab object error, no hold edge %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add edge (%d %s %d)' % (agent_i+1, 'hold', obj_id) )
                    graph['edges'].append({'from_id': agent_i+1, 'to_id': obj_id, 'relation_type': 'HOLDS_RH'})
            
        

    elif 'open' in script_list:
        obj_node_ids = [i for i, node in enumerate(graph['nodes']) if node['id']==obj_id]
        
        for node_id in obj_node_ids:
            if 'OPEN' not in graph['nodes'][node_id]['states']:
                print('agent open object error, no open state %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add node state: %d %s' % (obj_id, 'open') )
                    close_id = graph['nodes'][node_id]['states'].index('CLOSED')
                    graph['nodes'][node_id]['states'][close_id] = 'OPEN'

    elif 'close' in script_list:
        obj_node_ids = [i for i, node in enumerate(graph['nodes']) if node['id']==obj_id]
        
        for node_id in obj_node_ids:
            if 'CLOSED' not in graph['nodes'][node_id]['states']:
                print('agent close object error, no open state %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add node state: %d %s' % (obj_id, 'close') )
                    close_id = graph['nodes'][node_id]['states'].index('OPEN')
                    graph['nodes'][node_id]['states'][close_id] = 'CLOSED'

    elif 'switchon' in script_list:
        obj_node_ids = [i for i, node in enumerate(graph['nodes']) if node['id']==obj_id]
        
        for node_id in obj_node_ids:
            if 'ON' not in graph['nodes'][node_id]['states']:
                print('agent open object error, no open state %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add node state: %d %s' % (obj_id, 'open') )
                    close_id = graph['nodes'][node_id]['states'].index('OFF')
                    graph['nodes'][node_id]['states'][close_id] = 'ON'

    elif 'switchoff' in script_list:
        obj_node_ids = [i for i, node in enumerate(graph['nodes']) if node['id']==obj_id]
        
        for node_id in obj_node_ids:
            if 'OFF' not in graph['nodes'][node_id]['states']:
                print('agent close object error, no open state %s' % script_list)
                correct_graph_flag = False

                if fix_edge:
                    print('add node state: %d %s' % (obj_id, 'close') )
                    close_id = graph['nodes'][node_id]['states'].index('ON')
                    graph['nodes'][node_id]['states'][close_id] = 'OFF'                    

    elif 'putback' in script_list:
        if 'char' in script_list:
            obj_name2 = re.findall(r"\<([A-Za-z0-9_]+)\>", script_list)[2]
            obj_id2 = int(re.findall(r"\(([A-Za-z0-9_]+)\)", script_list)[1])
        else:
            obj_name2 = re.findall(r"\<([A-Za-z0-9_]+)\>", script_list)[1]
            obj_id2 = int(re.findall(r"\(([A-Za-z0-9_]+)\)", script_list)[1])

        obj1_edges = [edge for edge in graph['edges'] if edge['from_id']==obj_id or edge['to_id']==obj_id]
        obj1_obj2_edges = [edge for edge in obj1_edges if edge['from_id']==obj_id2 or edge['to_id']==obj_id2]
        obj1_obj2_edges_on = [edge for edge in obj1_obj2_edges if edge['relation_type']=='ON']

        if len(obj1_obj2_edges)==0:
            print('agent putback object error, no edge %s' % script_list)
            correct_graph_flag = False

            if fix_edge:
                print('add edge "%s(%d) %s %s(%d)"' % (obj_name, obj_id, 'on', obj_name2, obj_id2) )
                graph['edges'].append({'from_id': obj_id, 'to_id': obj_id2, 'relation_type': 'ON'})


    elif 'putin' in script_list:
        if 'char' in script_list:
            obj_name2 = re.findall(r"\<([A-Za-z0-9_]+)\>", script_list)[2]
            obj_id2 = int(re.findall(r"\(([A-Za-z0-9_]+)\)", script_list)[1])
        else:
            obj_name2 = re.findall(r"\<([A-Za-z0-9_]+)\>", script_list)[1]
            obj_id2 = int(re.findall(r"\(([A-Za-z0-9_]+)\)", script_list)[1])

        obj1_edges = [edge for edge in graph['edges'] if edge['from_id']==obj_id or edge['to_id']==obj_id]
        obj1_obj2_edges = [edge for edge in obj1_edges if edge['from_id']==obj_id2 or edge['to_id']==obj_id2]
        obj1_obj2_edges_inside = [edge for edge in obj1_obj2_edges if edge['relation_type']=='INSIDE']

        if len(obj1_obj2_edges)==0:
            print('agent putin object error, no edge %s' % script_list)
            correct_graph_flag = False

            if fix_edge:
                print('add edge "%s(%d) %s %s(%d)"' % (obj_name, obj_id, 'inside', obj_name2, obj_id2) )
                graph['edges'].append({'from_id': obj_id, 'to_id': obj_id2, 'relation_type': 'INSIDE'})



    else:
        pdb.set_trace()

    return correct_graph_flag, graph



def check_env_bug(agent_action, graph, agent_i, opponent_agent_action=None, logging=None):

    correct_graph_flag = True
    
    if agent_action is not None:
        obj_name = re.findall(r"\<([A-Za-z0-9_]+)\>", agent_action)[0]
        obj_id = int(re.findall(r"\(([A-Za-z0-9_]+)\)", agent_action)[0])
                
        correct_graph_flag, graph = check_env_bug_step(correct_graph_flag, agent_action, obj_name, obj_id, graph, agent_i, opponent_agent_action=opponent_agent_action)

    return correct_graph_flag




def action_repeat(actions, logging=None, agent_id=1):
    repeat = [1]
    rooms = ['kitchen', 'bedroom', 'livingroom', 'bathroom']
    
    ## interacted objects are not rooms
    for i in range(2, len(actions)):
        if actions[i] is None:
            obj_name = None
        else:
            obj_name = actions[i].split('<')[1].split('>')[0]
        
        if actions[i] is not None and obj_name in rooms:
            repeat.append(1)
            continue

        if actions[i] == actions[i-1] and actions[i] is not None:
            repeat[-1] += 1
        else:
            repeat.append(1)

    repeat_rooms = False
    for i in range(len(actions)-3):
        if actions[i] == actions[i+2] and actions[i+1] == actions[i+3] and actions[i] is not None and actions[i+1] is not None:
            print(actions[i], actions[i+1], actions[i+2], actions[i+3])
            repeat_rooms = True


    if np.max(repeat)>1 or repeat_rooms:
        for tem in actions: print(tem)
        print(repeat)
        logging.info('agent '+str(agent_id)+' repeat actions')
        return True
    else:
        return False

def check_action(data, n_agent=2, logging=None):
    if n_agent==1:
        a1_actions = [tem[0] for tem in data['actions']]
        a1_next_obs = [tem[0] for tem in data['partial_obs']]
        assert len(a1_actions)+1 == len(a1_next_obs)

        for i, _ in enumerate(a1_actions):
            if a1_actions[i] is not None:
                result = check_env_bug(a1_actions[i], a1_next_obs[i+1], agent_i=0)
                if not result:
                    return False

    elif n_agent==2:    
        a1_actions = [tem[0] for tem in data['actions']]
        a2_actions = [tem[1] for tem in data['actions']]
        a1_next_obs = [tem[0] for tem in data['partial_obs']]
        a2_next_obs = [tem[1] for tem in data['partial_obs']]
    
        assert len(a1_actions)+1 == len(a1_next_obs)
        assert len(a2_actions)+1 == len(a2_next_obs)
        
        for i, _ in enumerate(a1_actions):
            if a1_actions[i] is not None:
                result = check_env_bug(a1_actions[i], a1_next_obs[i+1], agent_i=0, opponent_agent_action=a2_actions[i])
                if not result:
                    return False

        for i, _ in enumerate(a2_actions):
            if a2_actions[i] is not None:
                result = check_env_bug(a2_actions[i], a2_next_obs[i+1], agent_i=1, opponent_agent_action=a1_actions[i])
                if not result:
                    return False

    return True



def check_action_logic(data, n_agent=2, logging=None):
    env_task_goal = {k.split('_')[1]:v for k,v in data['env_task_goal'][0][0].items() if v>0}


    if n_agent==1:
        a1_actions = [tem[0] for tem in data['actions']]    
        a1_cur_task_obj_count = {}
        a2_cur_task_obj_count = {}
        
    if n_agent==2:
        a1_actions = [tem[0] for tem in data['actions']]
        a2_actions = [tem[1] for tem in data['actions']]
        a1_message = [tem[0] for tem in data['message']]
        a2_message = [tem[1] for tem in data['message']]

        a1_cur_task_obj_count = {}
        a2_cur_task_obj_count = {}

    check_done_result = check_done(data, a1_cur_task_obj_count, a2_cur_task_obj_count, env_task_goal)

    return True and check_done_result



def check_done(data, a1_cur_task_obj_count, a2_cur_task_obj_count, env_task_goal, logging=None):
    try:
        if len(a1_cur_task_obj_count)>0 and len(a2_cur_task_obj_count)>0:
            for k,v in a1_cur_task_obj_count.items():
                if k in a2_cur_task_obj_count:
                    assert env_task_goal[k]>1
                    a1_cur_task_obj_count[k] += a2_cur_task_obj_count[k]

        for k,v in a1_cur_task_obj_count.items():
            assert env_task_goal[k]==v
    except:
        logging.info('not finish goal')
        return False
    return True


def check_cur_task(data, n_agent=2, logging=None):
    if n_agent==1:
        a1_cur_task = [tem[0] for tem in data['cur_task']]
        tasks = []
        for i, _ in enumerate(a1_cur_task):
            if a1_cur_task[i] not in tasks:
                tasks.append(a1_cur_task[i])

    elif n_agent==2:
        a1_cur_task = [tem[0] for tem in data['cur_task']]
        a2_cur_task = [tem[1] for tem in data['cur_task']]
            
        tasks = []
        for i, _ in enumerate(a1_cur_task):
            if a1_cur_task[i] not in tasks:
                tasks.append(a1_cur_task[i])
            if a2_cur_task[i] not in tasks:
                tasks.append(a2_cur_task[i])

    tasks = [tem for tem in tasks if tem is not None]

    ## check the task order
    env_task_goal = [k+'_'+str(v) for k,v in data['env_task_goal'][0][0].items() if v>0]

    try:
        assert len(tasks)==len(env_task_goal)
        for i, _ in enumerate(tasks): assert tasks[i] in env_task_goal
        return True
    except:
        logging.info('task error')
        logging.info(str(tasks))
        logging.info(str(env_task_goal))
        return False
    
    return tasks

    
def check_message(data, logging=None):
    a1_message = [tem[0] for tem in data['message']]
    a2_message = [tem[1] for tem in data['message']]

    a1_cur_task = [tem[0] for tem in data['cur_task']]
    a2_cur_task = [tem[1] for tem in data['cur_task']]
    
    for i, _ in enumerate(a1_message):
        try:
            if 'cur_task' in a1_message[i]:
                # logging.info(a1_message[i])
                a1_message_cur_task = [k+'_'+str(v) for k,v in a1_message[i]['cur_task'].items()]
                assert len(a1_message_cur_task)==1
                assert a2_cur_task[i+1]==a1_message_cur_task[0] ## a2 next task == a1 current message
                if i>0:
                    assert 'others' in a2_message[i-1] and 'done' in a2_message[i-1]['others'] ## a2 last task done

            elif 'finish' in a1_message[i]:
                # logging.info(a1_message[i])
                if 'I am done, I am helping you.' in a1_message[i]['finish']: # a1 help a2
                    pass
                else: # a1 ask a2 to help
                    if len(a2_cur_task)>i+1:
                        assert a2_cur_task[i+1]==a1_message[i]['finish'].split(':')[1][1:] ## a2 next task == a1 current message
                        
                    if i>0:
                        assert 'others' in a2_message[i-1] and 'done' in a2_message[i-1]['others'] ## a2 last task done
            else:
                assert 'none' in a1_message[i]
        except:
            logging.info('message error')
            for tem in a1_message: logging.info(tem)
            for tem in a2_message: logging.info(tem)
            return False

    return True




def check_data_two_agent(data, logging):
    cur_task_result = check_cur_task(data, logging=logging)
    message_result = check_message(data, logging=logging)

    action_logic_result = check_action_logic(data, logging=logging)
    action_result = check_action(data, logging=logging)

    a1_actions = [tem[0] for tem in data['actions']]
    a2_actions = [tem[1] for tem in data['actions']]
    a1_repeat_action = action_repeat(a1_actions, logging=logging, agent_id=1)
    a2_repeat_action = action_repeat(a2_actions, logging=logging, agent_id=2)
    
    output_message = []
    if not cur_task_result:
        output_message.append('cur_task_result')
    if not message_result:
        output_message.append('message_result')
    if not action_logic_result:
        output_message.append('action_logic_result')
    if not action_result:
        output_message.append('unity_action_result')
    if a1_repeat_action:
        output_message.append('a1_repeat_action')
    if a2_repeat_action:
        output_message.append('a2_repeat_action')

    if cur_task_result and message_result and action_logic_result and action_result and not a1_repeat_action and not a2_repeat_action:
        return True, output_message
    else:
        return False, output_message



def check_data_single_agent(data, logging):
    cur_task_result = check_cur_task(data, n_agent=1, logging=logging)
    action_logic_result = check_action_logic(data, n_agent=1, logging=logging)
    action_result = check_action(data, n_agent=1, logging=logging)

    a1_actions = [tem[0] for tem in data['actions']]
    a1_repeat_action = action_repeat(a1_actions, logging=logging, agent_id=1)
    

    output_message = []
    if not cur_task_result:
        output_message.append('cur_task_result')
    if not action_logic_result:
        output_message.append('action_logic_result')
    if not action_result:
        output_message.append('unity_action_result')
    if a1_repeat_action:
        output_message.append('a1_repeat_action')
    

    if cur_task_result and action_logic_result and action_result and not a1_repeat_action:
        return True, output_message
    else:
        return False, output_message
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check action')
    parser.add_argument('--data-dir', type=str, default='gen_action_message/result/init_env_2000_per_apt_only_goal', help='result folder')
    args = parser.parse_args()

    examples = glob.glob(os.path.join(args.data_dir, '*.p'))
    examples.sort()

    for i, example in enumerate(examples):
        logging.info(i, len(examples), example)
        data = pickle.load( open(example, 'rb') )
        check_result = check_data(data)

        assert check_result
        if not check_result:
            os.remove(example)
            logging.info('check_result', check_result)




