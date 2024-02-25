from torch import nn
import pdb
import torch
from copy import deepcopy
import torch.nn.functional as F


class ObjNameCoordStateEncodeNameTokenMix(nn.Module):
    def __init__(self, args, large_language_model_token_encoder_wte, output_dim=128, hidden_dim=128, num_node_name_classes=102, num_node_states=5):
        super(ObjNameCoordStateEncodeNameTokenMix, self).__init__()
        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.args = args

        self.large_language_model_token_encoder_wte = deepcopy(large_language_model_token_encoder_wte)
        
        self.class_fc = nn.Linear(output_dim, int(hidden_dim / 2))
        self.state_embedding = nn.Linear(num_node_states, int(hidden_dim / 2))
        self.coord_embedding = nn.Sequential(nn.Linear(6, int(hidden_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(hidden_dim / 2), int(hidden_dim / 2)))
        inp_dim = int(3*hidden_dim/2)
        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(inp_dim, output_dim))

        
    def forward(self, input_obs_node_gpt2_token, input_obs_node_gpt2_token_mask, input_obs_node_state_gpt2_token, input_obs_node_state_gpt2_token_mask, input_obs_char_obj_rel_gpt2_token, input_obs_char_obj_rel_gpt2_token_mask):
        obs_node_class_name_feat_tem = self.large_language_model_token_encoder_wte(input_obs_node_gpt2_token.long())
        obs_node_class_name_feat_tem = (obs_node_class_name_feat_tem * input_obs_node_gpt2_token_mask[:,:,:,None]).sum(2) / (1e-9 + input_obs_node_gpt2_token_mask.sum(2)[:,:,None])
        class_embedding = self.class_fc(obs_node_class_name_feat_tem)

        state_embedding = self.state_embedding(input_obs_node_state_gpt2_token)
        coord_embedding = self.coord_embedding(input_obs_char_obj_rel_gpt2_token)
        
        inp = torch.cat([class_embedding, coord_embedding, state_embedding], dim=2)

        return self.combine(inp)




class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=True, query=False, memory=False):
        super().__init__()
        self.key = key
        self.query = query
        self.memory = memory
        if self.key:
            self.make_key = nn.Linear(n_features, n_features)
        if self.query:
            self.make_query = nn.Linear(n_features, n_features)
        if self.memory:
            self.make_memory = nn.Linear(n_features, n_features)
        self.n_out = n_hidden


    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        if self.memory:
            memory = self.make_memory(features)
        else:
            memory = features

        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        scores = (key * query).sum(dim=2)
        
        if mask is not None:
            mask_values = (torch.min(scores, -1)[0].view(-1, 1).expand_as(scores)) * mask
            scores = scores * (1-mask) + mask_values
        
        return scores


