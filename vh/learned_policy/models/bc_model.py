import numpy as np
import pdb
import torch
import torch.nn as nn
from copy import deepcopy

from models import base_nets
from utils_bc.utils_llm import MODEL_CLASSES, LLM_HIDDEN_SIZE


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class BC_MODEL(nn.Module):
    def __init__(self, args):
        super(BC_MODEL, self).__init__()

        self.base = GoalAttentionModel(args, recurrent=True, hidden_size=args.hidden_size)
        self.train()

    def forward(self, data):
        value = self.base(data)
        return value

class GoalAttentionModel(nn.Module):
    def __init__(self, args, recurrent=False, hidden_size=128):
        super(GoalAttentionModel, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.args = args
        self.hidden_size = hidden_size

        self.max_node_length = args.data_info['max_node_length']

        self.graph_node_class_names = args.graph_node_class_names
        self.vocabulary_node_class_name_word_index_dict = args.vocabulary_node_class_name_word_index_dict
        self.vocabulary_node_class_name_index_word_dict = args.vocabulary_node_class_name_index_word_dict

        self.graph_node_states = args.graph_node_states
        self.vocabulary_node_state_word_index_dict = args.vocabulary_node_state_word_index_dict
        self.vocabulary_node_state_index_word_dict = args.vocabulary_node_state_index_word_dict

        self.action_names = args.action_names
        
        ## pretrained language model
        self.llm_hidden_size = LLM_HIDDEN_SIZE[self.args.model_type]
    
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        if self.args.language_model_type_pretrain=='train_scratch':
            print('large language model from scratch %s' % args.model_name_or_path)
            model_tem = model_class.from_pretrained(args.model_name_or_path)
            self.large_language_model = model_class(model_tem.config)

            model_tem = model_class.from_pretrained(args.model_name_or_path)
            model_tem = model_class(model_tem.config)
            self.large_language_model_token_encoder_wte = deepcopy(model_tem.transformer.wte)
        else:
            print('large language model from pretrained model %s' % args.model_name_or_path)
            self.large_language_model = model_class.from_pretrained(args.model_name_or_path)       

            model_tem = model_class.from_pretrained(args.model_name_or_path)
            self.large_language_model_token_encoder_wte = deepcopy(model_tem.transformer.wte)

        
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        model_tem = model_class.from_pretrained(args.model_name_or_path)
        self.large_language_model_token_encoder_wte = deepcopy(model_tem.transformer.wte)

        ## encoders
        self.single_object_encoding_name_token = base_nets.ObjNameCoordStateEncodeNameTokenMix(self.args, self.large_language_model_token_encoder_wte, output_dim=self.llm_hidden_size, hidden_dim=hidden_size, num_node_name_classes=len(self.graph_node_class_names), num_node_states=len(self.graph_node_states))
        self.large_language_model_token_encoder_wte_goal = deepcopy(model_tem.transformer.wte)
        self.large_language_model_token_encoder_wte_history = deepcopy(model_tem.transformer.wte)

        ## object / action decoders
        self.action_decoder_hidden = nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        self.verb_decoder = nn.Sequential(nn.ReLU(), nn.Linear(self.llm_hidden_size, len(self.action_names)))
        self.object_attention = base_nets.SimpleAttention(self.llm_hidden_size, hidden_size, key=False, query=False, memory=False)

        self.train()


    def forward(self, inputs):

        input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_node_state_gpt2_token, input_obs_node_state_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token, input_obs_char_obj_rel_gpt2_token_mask, \
                history_action_gpt2_token, history_action_gpt2_token_mask, goal_gpt2_token, goal_gpt2_token_mask = inputs
        
        B = input_obs_node_gpt2_token.shape[0]
        
        ## encode goal
        goal_embedding = self.large_language_model_token_encoder_wte_goal(goal_gpt2_token.long())
        goal_embedding = goal_embedding.view(B, -1, self.llm_hidden_size)
        goal_embedding_mask = goal_gpt2_token_mask.view(B, -1)

        ## encode observation
        input_node_embedding = self.single_object_encoding_name_token(input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_node_state_gpt2_token, input_obs_node_state_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token, input_obs_char_obj_rel_gpt2_token_mask)
        input_node_mask = input_obs_node_gpt2_token_mask.sum(2)>0
    
        ## encode history
        history_embedding = self.large_language_model_token_encoder_wte_history(history_action_gpt2_token.long())
        history_embedding = history_embedding.view(B, -1, self.llm_hidden_size)
        history_embedding_mask = history_action_gpt2_token_mask.view(B, -1)

        ## joint embedding
        joint_embedding = torch.cat([goal_embedding, history_embedding, input_node_embedding], dim=1)
        joint_mask = torch.cat([goal_embedding_mask, history_embedding_mask, input_node_mask], dim=1)
        
        ## pre-trained language model
        pretrained_language_output = self.large_language_model(inputs_embeds=joint_embedding, attention_mask=joint_mask, output_hidden_states=True)
        
        language_ouput_embedding = pretrained_language_output['hidden_states'][-1]
        joint_mask = joint_mask.unsqueeze(-1)
        language_ouput_embedding = language_ouput_embedding * joint_mask
        context_embedding = language_ouput_embedding.sum(1) / (1e-9 + joint_mask.sum(1))
        
        obs_node_embedding = language_ouput_embedding[:,-self.max_node_length:,:]
        action_hidden = self.action_decoder_hidden(context_embedding)

        ## predict verb / object
        verb = self.verb_decoder(action_hidden)
        obj = self.object_attention(obs_node_embedding, action_hidden.unsqueeze(dim=1), mask=1-input_node_mask.float())
        
        return verb, obj
        






