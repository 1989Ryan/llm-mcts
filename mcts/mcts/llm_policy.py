from openai import OpenAI

client = OpenAI(api_key="Your key")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
from typing import List
import os
import random

MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
LAMBDA = 0.5

class LLMPolicy:
    def __init__(self, device):
        self.device = device
        self.sampling_params = \
            {
                "max_tokens": 8,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 5,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": '\n'
            }
        with open(os.path.expanduser('~') + \
            "/llm-pomdp/mcts/mcts/prompt.txt", "r") as f:
            self.prompt = f.read()
        # print(f'Prompt:\n{self.prompt}.')
        self.action_dict = {
		    "pick the apple" : 0, "pick the t-shirt" : 1, 'pick the cup' : 2, "move to the bathroom" : 3,
            "move to the bedroom": 4, "move to the kitchen" : 5, "move to the living room" : 6, 
            "drop the picked item" : 7, "drop the apple": 7, "drop the t-shirt": 7, "drop the cup": 7
        }
        self.previous_action = None
        self.action_list = ["pick the apple", "pick the t-shirt", "pick the cup", "move to the bathroom", "move to the bedroom",
                            "move to the kitchen", "move to the living room", "drop the picked item", "drop the apple", "drop the t-shirt", "drop the cup"]
        self.state_list = ["in the bedroom", "in the living room", "in the bathroom", "in the kitchen", "picked by you",]
        self.item_list = ["you", "human", "apple", "t-shirt", "cup", ]
        self.init_translation_trf()
        print('Action Policy Service is ready.')


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

    
    def find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        return most_similar_idx, matching_score 
    
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

    def ground_actions(self, task, valid_action_embedding, valid_actions):
        samples = self.query_llm(task)
        # print(samples)
        actions = []
        num_of_steps = []
        is_done = []
        for sample in samples:
            most_similar_idx, _ = self.find_most_similar(sample[0], valid_action_embedding)
            if "done" in sample[-1] or "Done" in sample[-1]:
                is_done.append(True)
            else:
                is_done.append(False)
            num_of_steps.append(len(sample)) 
            translated_action = valid_actions[most_similar_idx]
            actions.append(translated_action)
        # print(actions)
        return actions, num_of_steps, is_done


    def query_llm(self, task):
        # response = openai.chatcompletion.create(
        #     model="gpt-3.5-turbo",
        #     prompt=self.prompt + task,
        #     **self.sampling_params,
        # )
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        # model = "gpt-4",
        messages=[{
            "role": "user",
            "content": self.prompt + task,
        }],
        **self.sampling_params)
        # print(task)
        # print(response['choices'][0]['message']['content'])
        generated_samples = [response.choices[i].message.content.split(", ") \
            for i in range(self.sampling_params['n'])]
        # print(generated_samples)
        # generated_samples = [response['choices'][i]['text'].split("\n")[0] for i in range(self.sampling_params['n'])]
            # calculate mean log prob across tokens
        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) 
                        #   for i in range(self.sampling_params['n'])]
        # print(generated_samples)
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

    def calculate_emperical_prob(self, history, observation, valid_actions, instruction, done_reward, step_reward, discount_factor):
        # calculate the emperical probability of each action given the history and observation
        # history: list of actions
        # observation: string
        # valid_actions: list of valid actions
        # return: list of probabilities
        hist_text = ", ".join(history)
        valid_action_list = list(valid_actions.keys())
        valid_action_text = ", ".join(valid_action_list)
        valid_action_embedding = self.translation_lm.encode(valid_action_list, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        task = f"Task description: {instruction}\nCompleted actions: {hist_text}\nYou see {observation}\nThe available actions are {valid_action_text}.\nYour next plan:"
        samples, num_steps, is_done = self.ground_actions(task, valid_action_embedding, valid_action_list)
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

    def act(self, history, observation, valid_actions, instruction):
        # calculate the emperical probability of each action given the history and observation
        # history: list of actions
        # observation: string
        # valid_actions: list of valid actions
        # return: list of probabilities
        hist_text = ", ".join(history)
        valid_action_list = list(valid_actions.keys())
        valid_action_text = ", ".join(valid_action_list)
        valid_action_embedding = self.translation_lm.encode(valid_action_list, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        task = f"Task description: {instruction}\nCompleted actions: {hist_text}\nYou see {observation}\nThe available actions are {valid_action_text}.\nYour next plan:"
        samples, _, _ = self.ground_actions(task, valid_action_embedding, valid_action_list)
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