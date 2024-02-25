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
import backoff
import openai as openai_api

MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
LAMBDA = 0.5

@backoff.on_exception(backoff.expo, openai_api.OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

class LLMPolicy:
    def __init__(self, device, model='gpt-3.5-turbo-0125'):
        self.device = device
        self.model = model
        self.sampling_params = \
            {
                "max_tokens": 10,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 5,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": [',', '.', '\n']
            }
        
        self.prompt_begin = """Generate a high-level plan for completing a household task using the allowed actions and visible objects.
Allowed actions: walk to <object or room>, grab <object>, open <container>, close <container>, put <object> on <surface>, put <object> in <container>. Before taking objects from container, you need to open it first. 
Rooms in the house: bedroom, bathroom, living room, kitchen. You can only grab one object at all time. 
You need to find the target object and perform the action in the correct order. 
Do not generate repeated or looped actions. You must interact with objects that are observed. You must generate the defined actions. 
\nExample tasks:"""
        self.translation_lm = SentenceTransformer('stsb-roberta-large').to(self.device)

        self.task_embedding = torch.load('expert_actions/task_goal_embd_list.pt').to(self.device)
        self.observation_embedding = torch.load('expert_actions/obs_langs_embd_list.pt')
        self.act_lang_list = pickle.load(open('expert_actions/act_langs_list.pik', 'rb'))
        self.task_lang_list = pickle.load(open('expert_actions/task_langs_list.pik', 'rb'))
        self.obs_lang_list = pickle.load(open('expert_actions/obs_langs_list.pik', 'rb'))
        self.prompt_buffer = {}


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
        return most_similar_idxs[::-1], matching_scores[::-1]
    
    def find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx = np.argmax(cos_scores)
        return most_similar_idx
    
    def _find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        cos_scores = cos_scores - np.mean(cos_scores) 
        return cos_scores
    
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

    def ground_actions_softmax(self, task, instruction, observation, valid_action_embedding, valid_actions):
        samples = self.query_llm(task, instruction, observation)
        actions_dis = np.zeros(len(valid_actions))
        num_of_steps = []
        is_done = []
        for sample in samples:
            cos_sim = self._find_most_similar(sample[0], valid_action_embedding)
            if "done" in sample[-1] or "Done" in sample[-1]:
                is_done.append(True)
            else:
                is_done.append(False)
            num_of_steps.append(len(sample)) 
            # use softmax to get the distribution using cos_sim
            softmax = np.exp(100 * cos_sim) / np.sum(np.exp(100 * cos_sim), axis=0)
            actions_dis += (softmax / len(samples))
        # print(valid_actions)
        return actions_dis, valid_actions, num_of_steps, is_done
    
    def ground_actions(self, task, instruction, observation, valid_action_embedding, valid_actions):
        try_times = 0
        print(observation)
        while 1:
            try:
                samples = self.query_llm(task, instruction, observation)
                break
            except:
                try_times += 1
                time.sleep(5)
            if try_times >=10:
                return None, None, None
        # print(samples)
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
        if similar_obs_idx != 0:
            completed_actions = ", ".join(actions[:similar_obs_idx - 1]) + '.'
            next_actions = ", ".join(actions[similar_obs_idx - 1:])
        else:
            completed_actions = ", ".join(actions[:similar_obs_idx]) + '.'
            next_actions = ", ".join(actions[similar_obs_idx:])
        # prompt = f"{goal}\nCompleted actions: {completed_actions}\nNext plan: {next_actions}, done."
        prompt = f"{goal}\n{observations}\nPrevious actions: {completed_actions}\nNext actions: {next_actions}"
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
        prompt_examples = "\n\n".join(prompt_examples) + "\n\nNow, complete the following task.\n"
        return self.prompt_begin + prompt_examples
    
    def get_prompt(self, most_similar_task_idx, most_similar_obs_idx):
        task = self.task_lang_list[most_similar_task_idx]
        observation = self.obs_lang_list[most_similar_task_idx][most_similar_obs_idx]
        action = self.act_lang_list[most_similar_task_idx]
        # print(observation)
        prompt = self.construct_prompt(action, observation, task, most_similar_obs_idx)
        return prompt


    def query_llm(self, task, ins, curr_obs, k=3):
        prompt = self.get_prompt_examples(ins, curr_obs, k)
        if prompt + task in self.prompt_buffer:
            return self.prompt_buffer[prompt + task]
        else:
            response = completions_with_backoff(model=self.model,
            timeout=5,
            messages=[{
                "role": "system",
                "content": prompt + task,
            }],
            **self.sampling_params)
            generated_samples = [response.choices[i].message.content.split(", ") \
                for i in range(self.sampling_params['n'])]
            self.prompt_buffer[prompt + task] = generated_samples
            return generated_samples

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

    def _calculate_emperical_prob(self, history, observation, valid_action_list, instruction, done_reward, step_reward, discount_factor):
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
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        task = f"{instruction}\n{observation}\nPrevious actions: {', '.join(hist_text)}.\nNext actions:"
        action_dis, valid_action_lists, num_steps, is_done = self.ground_actions_softmax(task, instruction, observation, valid_action_embedding, valid_action_list)
        pred_value = self.trivial_pred_value_function(history, num_steps, is_done, done_reward, step_reward, discount_factor)
        emperical_prob = LAMBDA * action_dis + (1-LAMBDA) /len(valid_action_list)
        # for action in valid_action_list:
        #     if action in action_count:
        #         emperical_prob.append(LAMBDA * action_count[action] / len(samples) + (1-LAMBDA) /len(valid_action_list))
        #     else:
        #         emperical_prob.append((1-LAMBDA) / len(valid_action_list))
        return emperical_prob # , pred_value

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
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\n{observation}\nNext plan:"
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
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\n{observation}\nNext plan:"
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