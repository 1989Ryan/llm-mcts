from openai import OpenAI

client = OpenAI(api_key="Your Key")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
from typing import List
import os
import random
import pickle
from mcts.virtualhome.expert_data import get_action_list_valid
import time

MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
LAMBDA = 0.5

class LLM_Model:
    def __init__(self, device, model='gpt-3.5-turbo-0125'):
        self.device = device
        self.model = model
        self.get_goal_sample_params = \
            {
                "max_tokens": 32,
                "temperature": 0.5,
                "top_p": 0.9,
                "n": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": ['\n']
            }
        
        self.sampling_params = \
            {
                "max_tokens": 32,
                "temperature": 0.5,
                "top_p": 0.9,
                "n": 50,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": ['\n']
            }
        
        self.prompt_furniture_begin = """Generate one most possible position of the objects in the scene. The objects should be placed INSIDE a room.

Rooms in the scene: living room, kitchen, bedroom, bathroom

You need to strictly follow the format in the following examples, and only generate one instance:\n
Question: What is the most possible position of desktop?
Answer: INSIDE bedroom\n
Question: What is the most possible position of snack?
Answer: INSIDE kitchen, INSIDE living room\n
Question: What is the most possible position of strawberry?
Answer: INSIDE kitchen, INSIDE living room\n
Question: What is the most possible position of soap?
Answer: INSIDE bathroom\n
Now, answer the following questions:\n
"""
        self.lang_prompt = """You need to interpret the natural language goal into a formal goal representation
For example,
Goal: put 1 toothbrush inside the bathroomcabinet.
Formal goal: (INSIDE, toothbrush, bathroomcabinet, 1)
Goal: put 1 toothbrush inside the bathroomcabinet, put 1 apple on the kitchentable.
Formal goal: (INSIDE, toothbrush, bathroomcabinet, 1)-(ON, apple, kitchentable, 1)
Goal: put 1 toothbrush inside the bathroomcabinet, put 1 apple on the kitchentable, put 1 chicken inside the fridge.
Formal goal: (INSIDE, toothbrush, bathroomcabinet, 1)-(ON, apple, kitchentable, 1)-(INSIDE, chicken, fridge, 1)
Now, interpret the next following goals:
"""
        self.prompt_begin = """Generate one most possible position of the objects in the scene. The objects should be placed ON the surfaces (for example, table) or INSIDE a container (for example, a fridge)

Containers in the scene: bathroomcabinet, kitchencabinet, bathroomcounter, fridge, oven, dishwasher, microwave, stove, bathroomcabinet
Surfaces in the scene: bed, bookshelf, cabinet, coffeetable, cuttingboard, floor, fryingpan, kitchencounter, kitchentable, nightstand, bathroomcounter, sofa, stove

You need to strictly follow the format in the following examples, and only generate one instance:\n
Question: What is the most possible position of watermelon?
Answer: INSIDE fridge\n
Question: What is the most possible position of snack?
Answer: INSIDE kitchencabinet\n
Question: What is the most possible position of strawberry?
Answer: ON kitchencounter\n
Question: What is the most possible position of soap?
Answer: ON bathroomcounter\n
Now, answer the following questions:\n
"""
        self.translation_lm = SentenceTransformer('stsb-roberta-large').to(self.device)
        self.object_info = json.load(open('./data/object_info.json', 'r'))
        self.container_list = self.object_info['objects_inside']
        self.surface_list = self.object_info['objects_surface']
        # self.grabable_list = self.object_info['objects_grab']
        self.grabable_list = self.object_info['objects_grab']
        self.container_list_embedding = self.translation_lm.encode(self.container_list, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory
        self.surface_list_embedding = self.translation_lm.encode(self.surface_list, batch_size=8,
                convert_to_tensor=True, device=self.device)
        self.grabable_list_embedding = self.translation_lm.encode(self.grabable_list, batch_size=8,
                convert_to_tensor=True, device=self.device)
        self.position_list =  self.container_list + self.surface_list
        self.position_list_embedding = torch.concat((self.container_list_embedding, self.surface_list_embedding), dim=0)
        self.room_embedding = self.translation_lm.encode(['living room', 'kitchen', 'bedroom', 'bathroom'], batch_size=8,
                convert_to_tensor=True, device=self.device)
        self.room_list = ['livingroom', 'kitchen', 'bedroom', 'bathroom']

    def trans_observation_to_string(self, observation: int):
        '''translate observation  (int) into a string'''
        obs_list = [] 
        for i in range(5):
            obs_list.append(int(observation/10**i)%10)
        # obs_list = obs_list[::-1]
        obs_str = ""
        for i in range(5):
            if obs_list[i] < 5:
                if i == 0:
                    obs_str += self.item_list[i] + " are "\
                        + self.state_list[obs_list[i]] + ", "
                else:
                    obs_str += self.item_list[i] + " is "\
                        + self.state_list[obs_list[i]] + ", "
        return obs_str[:-2] + "."
        

    def init_translation_trf(self):
        # initialize Translation LM
        self.translation_lm = SentenceTransformer('stsb-roberta-base').to(self.device)

        # create action embeddings using Translated LM
        # with open('available_actions.json', 'r') as f:
        #     self.action_list = json.load(f)
        
        
        
        self.action_list_embedding = self.translation_lm.encode(self.action_list, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory

        # create example task embeddings using Translated LM
        # with open('available_examples.json', 'r') as f:
        #     available_examples = json.load(f)
        # example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
        # self.example_task_embedding = self.translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
    def find_top_k_most_similar(self, query_embedding, corpus_embedding, k):
        # helper function for finding similar sentence in a corpus given a query
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve top k high-ranked index and similarity score
        most_similar_idxs = np.argsort(cos_scores)[-k:]
        matching_scores = cos_scores[most_similar_idxs]
        return most_similar_idxs, matching_scores
    
    def find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx = np.argmax(cos_scores)
        return most_similar_idx
    
    def plan(self, task):
        best_overall_score = -np.inf
        samples = self.query_llm(task)
        for sample in samples:
            most_similar_idx, matching_score = self.find_most_similar(sample, self.action_list_embedding)
            # rank each sample by its similarity score and likelihood score
            overall_score = matching_score 
            translated_action = self.action_list[most_similar_idx]
            # heuristic for penalizing generating the same action as the last action
            if self.previous_action is not None:
                if translated_action == self.previous_action:
                    overall_score -= 0.5
            # find the translated action with highest overall score
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_action = translated_action

        # terminate early when either the following is true:
        # 1. top P*100% of samples are all 0-length (ranked by log prob)
        # 2. overall score is below CUTOFF_THRESHOLD
        # else: autoregressive generation based on previously translated action
        top_samples_ids = np.argsort(matching_score)[-int(P * len(samples)):]
        are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
        below_threshold = best_overall_score < CUTOFF_THRESHOLD
        if are_zero_length:
            print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
            return
        # elif below_threshold:
        #     print(f'\n[Terminating early because best overall score is lower than CUTOFF_THRESHOLD ({best_overall_score} < {CUTOFF_THRESHOLD})]')
        #     return
        else:
            self.previous_action = best_action
            formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ') # 'open_fridge' -> 'Open fridge'
            # self.prompt += f'\nStep {step}: {formatted_action}'
            # print(f'Step {step}: {formatted_action}')
        return self.action_dict[best_action]

    def ground_actions(self, task, instruction, observation, valid_action_embedding, valid_actions):
        try_times = 0
        while 1:
            try:
                samples = self.query_llm(task, instruction, observation)
                break
            except:
                try_times += 1
                time.sleep(5)
            if try_times >=10:
                return None, None, None
        print(samples)
        actions = []
        num_of_steps = []
        is_done = []
        for sample in samples:
            most_similar_idx = self.find_most_similar(sample[0], valid_action_embedding)
            if "done" in sample[-1] or "Done" in sample[-1]:
                is_done.append(True)
            else:
                is_done.append(False)
            num_of_steps.append(len(sample)) 
            translated_action = valid_actions[most_similar_idx]
            actions.append(translated_action)
        # print(actions)
        # print(valid_actions)
        return actions, num_of_steps, is_done

    @staticmethod
    def construct_prompt(actions, observations, lang_goal, similar_obs_idx):
        # construct the prompt with the format of 
        # goal: <goal>
        # completed actions: <action1>, <action2>, ... <action<similar_obs_idx>>
        # visible objects: <observations>[<similar_obs_idx>]
        # next action: <action<similar_obs_idx>+1>, <action<similar_obs_idx>+2>, ...
        goal = lang_goal
        completed_actions = ", ".join(actions[:similar_obs_idx]) + '.'
        next_actions = ", ".join(actions[similar_obs_idx:])
        prompt = f"{goal}\nCompleted actions: {completed_actions}\nNext actions: {next_actions}, done."
        # prompt = f"{goal}\nCompleted actions: {completed_actions}\n{observations}\nNext actions: {next_actions}"
        return prompt

    def find_similar_task_and_observation(self, query_task, curr_observation, k):
        # find the most similar task and observation in the corpus given the current task and observation
        # calculate cosine similarity against each candidate sentence in the corpus
        task_embedding = self.translation_lm.encode(query_task, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        most_similar_task_idxs, _ = self.find_top_k_most_similar(task_embedding, self.task_embedding, k)
        # retrieve high-ranked index and similarity score
        # find the most similar observation given the most similar task
        most_similar_obs_idxs = [self.find_most_similar(curr_observation, self.observation_embedding[most_similar_task_idx].to(self.device)) for most_similar_task_idx in most_similar_task_idxs]
        return most_similar_task_idxs, most_similar_obs_idxs

    def get_prompt_examples(self, task, observation, k):
        # get the prompt examples given the current task and observation
        most_similar_task_idxs, most_similar_obs_idxs = self.find_similar_task_and_observation([task], observation, k)
        prompt_examples = [self.get_prompt(most_similar_task_idx, most_similar_obs_idx) 
                           for most_similar_task_idx, most_similar_obs_idx in zip(most_similar_task_idxs, most_similar_obs_idxs)]
        prompt_examples = "\n\n".join(prompt_examples) + "\n\nNow, please finish the following task.\n\n"
        
        return self.prompt_furniture_begin + prompt_examples
        # return self.prompt_begin + prompt_examples
    
    def get_prompt(self, most_similar_task_idx, most_similar_obs_idx):
        task = self.task_lang_list[most_similar_task_idx]
        observation = self.obs_lang_list[most_similar_task_idx][most_similar_obs_idx]
        action = self.act_lang_list[most_similar_task_idx]
        # print(observation)
        prompt = self.construct_prompt(action, observation, task, most_similar_obs_idx)
        return prompt
    
    def interpret_goal(self, goal_language, container_name2id):
        # interpret goal language into a list of actions
        # goal_language: string
        # return: list of actions
        subgoals = self.query_llm_goal(goal_language)
        formal_goal = {}
        for subgoal in subgoals:
            print(subgoal)
            relation = subgoal[0]
            target_object = subgoal[1]
            container = subgoal[2]
            num = subgoal[3]
            if relation == 'INSIDE':
                if container in self.room_list:
                    formal_goal[f'inside_{target_object}_{container_name2id[container]}'] = [num, True, 2]
                else:
                    formal_goal[f'inside_{target_object}_{container_name2id[container]}'] = [num, True, 2]
            elif relation == 'ON':
                formal_goal[f'on_{target_object}_{container_name2id[container]}'] = [num, True, 2]

        return formal_goal                      


        
    def query_llm_goal(self, ins):
        try_times = 0
        while 1:
            try: 
                response = client.chat.completions.create(model=self.model,
                # model = "gpt-4",
                messages=[{
                    "role": "system",
                    # "content": self.prompt_furniture_begin,
                    "content": self.lang_prompt,
                },
                {
                    "role": "user",
                    "content": f"Goal: {ins}.\nFormal goal: ",
                }
                ],
                **self.get_goal_sample_params)
                break
            except:
                try_times += 1
                time.sleep(5)
            if try_times >= 10:
                return None
        # print(task)
        # print(response['choices'][0]['message']['content'])
        generated_samples = response.choices[0].message.content.split("-") 
        # print(generated_samples)
        # generated_samples = [response['choices'][i]['text'].split("\n")[0] for i in range(self.sampling_params['n'])]
            # calculate mean log prob across tokens
        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) 
                        #   for i in range(self.sampling_params['n'])]
        subgoals = []
        for subgoal in generated_samples:
            subgoal = subgoal.replace('(', '').replace(')', '')
            subgoal = subgoal.split(',')
            subgoal[0] = subgoal[0].strip()
            subgoal[1] = subgoal[1].strip()
            subgoal[2] = subgoal[2].strip()
            subgoal[3] = int(subgoal[3].strip())
            subgoals.append([subgoal[0], subgoal[1], subgoal[2], subgoal[3]])
        return subgoals

    
    def query_llm(self, task, ins, curr_obs, k=3):
        # response = openai.chatcompletion.create(
        #     model="gpt-3.5-turbo",
        #     prompt=self.prompt + task,
        #     **self.sampling_params,
        # )
        # prompt = self.get_prompt_examples(ins, curr_obs, k)
        # print(prompt + task)
        print(task)
        try_times = 0
        while 1:
            try: 
                response = client.chat.completions.create(model=self.model,
                # model = "gpt-4",
                messages=[{
                    "role": "system",
                    # "content": self.prompt_furniture_begin,
                    "content": self.prompt_begin,
                },
                {
                    "role": "user",
                    "content": task,
                }
                ],
                **self.sampling_params)
                break
            except:
                try_times += 1
                time.sleep(5)
            if try_times >= 10:
                return None
        # print(task)
        # print(response['choices'][0]['message']['content'])
        generated_samples = [response.choices[i].message.content.split(",") \
            for i in range(self.sampling_params['n'])]
        # print(generated_samples)
        # generated_samples = [response['choices'][i]['text'].split("\n")[0] for i in range(self.sampling_params['n'])]
            # calculate mean log prob across tokens
        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) 
                        #   for i in range(self.sampling_params['n'])]
        samples = []
        for generated_sample in generated_samples:
            for sample in generated_sample:
                sample_ = sample.split(' ')
                if '' in sample_:
                    sample_.remove('')
                if len(sample_) != 2:
                    samples.append([sample_[0], sample_[1] + ' ' + sample_[2]])
                samples.append(sample_) 
        return samples

    def construct_house_model(self):
        # find the possible positions of the objects in the house
        object_positions = {}
        # for object in self.grabable_list:
        self.furniture_list = self.container_list + [ele for ele in self.surface_list if ele not in self.container_list]
        self.all_list = self.furniture_list + self.grabable_list
        for object in self.grabable_list:
            samples = self.query_llm(f"Question: What is the most possible position of {object}?\nAnswer: ", "", "", 1)
            object_positions[object] = samples
            # print(object_positions[object])
        # print(object_positions)   

        # save samples
        json.dump(object_positions, open("./data/object_all_positions.json", "w"))
        object_positions = {}
        for object in self.furniture_list:
            samples = self.query_llm(f"Question: What is the most possible position of {object}?\nAnswer: ", "", "", 1)
            object_positions[object] = samples
            # print(object_positions[object])
        # print(object_positions)   
        # save samples
        json.dump(object_positions, open("./data/furniture_all_positions.json", "w"))

    def call(self, request, response):
        history = '\n'
        step = 0
        for action in request.history:
            
            if step >= len(request.observation_history):
                break
            if step == 0:
                step += 1
                continue
            history += "You "
            history += self.action_list[action] 
            history += '.\n'
            history += f"You observe {self.trans_observation_to_string(request.observation_history[step])}\n"
            step += 1
        if request.observation != -1:
            observation = self.trans_observation_to_string(request.observation)
        else:
            observation = ""
            response.action = 7
            return response
        task = f"Task description: {self.instruction}{history}You observe {observation}\nYou"
        # self.get_logger().info(
        #     f'Incoming request\nCompleted plan: {history}\nVisible object: {observation}')

        action_next = self.plan(task) 
        if action_next is None:
            response.action = 7
            return response
        # print(f'Next action: {action_next}')
        response.action = int(action_next)
        return response

    def calculate_emperical_prob(self, history, observation, valid_action_list, instruction, done_reward, step_reward, discount_factor):
    # def act(self, history, observation, valid_action_list, instruction):
        # calculate the emperical probability of each action given the history and observation
        # history: list of actions
        # observation: string
        # valid_actions: list of valid actions
        # return: list of probabilities
        if not isinstance(observation, str):
            observation = self.get_observation(observation)
        if len(history) >0:
            hist_text = get_action_list_valid(history, len(history)) 
        else:
            hist_text = ""
        valid_actions_lang = get_action_list_valid(valid_action_list, len(valid_action_list))
        # print(valid_actions_lang)
        valid_action_embedding = self.translation_lm.encode(valid_actions_lang, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\n{observation}\nNext plan:"
        samples, num_steps, is_done = self.ground_actions(task, instruction, observation, valid_action_embedding, valid_action_list)
        pred_value = self.trivial_pred_value_function(history, num_steps, is_done, done_reward, step_reward, discount_factor)
        action_count = {}
        for action in samples:
            if action not in action_count:
                action_count[action] = 0
            action_count[action] += 1
        emperical_prob = []
        for action in valid_action_list:
            if action in action_count:
                emperical_prob.append(LAMBDA * action_count[action] / len(samples) + (1-LAMBDA) /len(valid_action_list))
            else:
                emperical_prob.append((1-LAMBDA) / len(valid_action_list))
        return emperical_prob, pred_value

    @staticmethod
    def get_observation(obs_graph):
        # print(obs_graph)
        list_obs_item = [node['class_name'] for node in obs_graph[0]['nodes']]
        list_obs_item = list(set(list_obs_item))
        observation = 'Visible objects are: ' + ', '.join(list_obs_item)
        return observation

    def act(self, history, observation, valid_action_list, instruction):
        # calculate the emperical probability of each action given the history and observation
        # history: list of actions
        # observation: string
        # valid_actions: list of valid actions
        # return: list of probabilities
        if not isinstance(observation, str):
            observation = self.get_observation(observation)
        if len(history) >0:
            hist_text = get_action_list_valid(history, len(history)) 
        else:
            hist_text = ""
        valid_actions_lang = get_action_list_valid(valid_action_list, len(valid_action_list))
        valid_action_embedding = self.translation_lm.encode(valid_actions_lang, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\n{observation}\nNext plan:"
        samples, _, _ = self.ground_actions(task, instruction, observation, valid_action_embedding, valid_action_list)
        return random.choice(samples)

    def trivial_pred_value_function(self, history, num_of_steps, is_done, done_reward, step_reward, discount_factor):
        # return a list of probabilities
        # num_of_steps: list of number of steps
        # is_done: list of boolean
        # return: list of probabilities
        pred_vale_functions = []
        for i in range(len(num_of_steps)):
            value = 0
            for j in range(num_of_steps[i]):
                value += step_reward * discount_factor ** j
            if is_done[i]:
                value += done_reward * discount_factor ** num_of_steps[i]
                pred_vale_functions.append(value)
            pred_vale_functions.append(value)
        ave_value = sum(pred_vale_functions)/len(pred_vale_functions)
        # print(f'Predicted value: {ave_value}')
        return ave_value
    
    def rollout(self, history, observation, valid_actions, instruction, discount_factor):
        '''return number of remaining steps'''
        valid_action_list = list(valid_actions.keys())
        valid_action_embedding = self.translation_lm.encode(valid_action_list, convert_to_tensor=True, device=self.device)
        task = f"Task description: {instruction}{history}You observe {observation}\nYou"
        samples = self.ground_actions(task, valid_action_embedding, valid_action_list)
        pass
    
    def post_process(self, filepath, goal_path):
        # process the data after querying the llm
        data = json.load(open(filepath, 'r'))
        data_new = {}
        for key, values in data.items():
            for ele in values:
                # print(ele)
                if len(ele) != 2:
                    continue
                rel = ele[0]
                obj = ele[1]
                if rel not in ['INSIDE', 'ON']:
                    continue
                else:
                    similar_obj_idx = self.find_most_similar(obj, self.room_embedding) 
                    # similar_obj_idx = self.find_most_similar(obj, self.position_list_embedding) 
                    position_obj = self.room_list[similar_obj_idx]
                    if key not in data_new:
                        data_new[key] = {}
                    if (rel, position_obj) not in data_new[key]:
                        data_new[key][(rel, position_obj)] = 0
     
                    data_new[key][(rel, position_obj)] += 1
        data_save = {}
        for key, value in data_new.items():
            data_save[key] = [(ele[0], ele[1], value[ele]) for ele in value]
        json.dump(data_save, open(goal_path, 'w'))
        print(data_save)

if __name__ == '__main__':
    llm_agent = LLM_Model("cuda:0")
    llm_agent.construct_house_model()
    llm_agent.post_process('./data/object_all_positions.json', './data/obj_commonsense.json')
    llm_agent.post_process('./data/furniture_all_positions.json', './data/fur_commonsense.json')
