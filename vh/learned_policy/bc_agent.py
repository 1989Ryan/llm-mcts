import numpy as np
import torch.optim as optim
import torch.nn as nn
import pdb
import torch
import random
from utils_bc.utils import accuracy, RecordLossTop1
from utils_bc.utils_llm import MODEL_CLASSES, LLM_HIDDEN_SIZE
import wandb
from utils_bc.utils import save_model

class BC_Agent(object):
    def __init__(self, args, model, action_criterion, obj_criterion, logging, gpu):
      
        self.args = args
        self.n_agent = args.n_agent
        self.model = model
        self.gpu = gpu


        if self.args.language_model_type_pretrain=='fine_tune_pretrain':
            self.model = nn.DataParallel(self.model, device_ids=[6, 7])
            self.model.to("cuda:6")
            print('large language model from pretrained model %s, fine tune pretrain llm' % args.model_name_or_path)
            self.optimizer = optim.AdamW(self.model.parameters(), **{'lr': args.lr, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0})

        else:
            pdb.set_trace()

        self.action_criterion = action_criterion
        self.obj_criterion = obj_criterion
        self.logging = logging
        self.llm_hidden_size = LLM_HIDDEN_SIZE[self.args.model_type]



    def run(self, trainloader, epoch, mode='train'):
        self.logger = wandb.init(project="vh-bc")
        recoder = RecordLossTop1(self.args)
        steps = 0
        verbose = False

        if mode == 'train':
            self.model.train()
            self.action_criterion.train()
            self.obj_criterion.train()
        else:
            self.model.eval()
            self.action_criterion.eval()
            self.obj_criterion.eval()
            
        for e in range(epoch):
            for i, data in enumerate(trainloader):
                
                for k, _ in enumerate(data):
                    data[k] = torch.tensor(data[k]).float().to("cuda:6")
                B = data[1].shape[0]
                

                input_obs_node, input_obs_node_mask, input_obs_node_state, input_obs_node_state_mask, input_obs_node_coords, input_obs_node_coords_mask, \
                        history_action_index, history_action_index_mask, goal_index, goal_index_mask, \
                        output_action = data


                batch_max_steps = 1

                loss_all = 0
                action_loss_all = 0
                obj_loss_all = 0

                top1_all = 0
                action_top1_all = 0
                obj_top1_all = 0

                for step_i in range(batch_max_steps):
                    input_data = [input_obs_node[:,step_i], input_obs_node_mask[:,step_i], input_obs_node_state[:,step_i], input_obs_node_state_mask[:,step_i], input_obs_node_coords[:,step_i], input_obs_node_coords_mask[:,step_i], \
                                    history_action_index[:,step_i], history_action_index_mask[:,step_i], goal_index[:,step_i], goal_index_mask[:,step_i]]


                    if mode == 'train':
                        self.optimizer.zero_grad()
                    

                    action, obj = self.model(input_data)

                    ## loss
                    action_loss = self.action_criterion(action, output_action[:,step_i][:, 0].long())
                    obj_loss = self.obj_criterion(obj, output_action[:,step_i][:,1].long())
                    
                    loss = (action_loss + obj_loss)/2
                    loss_all += loss
                    action_loss_all += action_loss
                    obj_loss_all += obj_loss
                    
                    ## accuracy
                    action_top1 = accuracy(action, output_action[:,step_i][:, 0].long())[0]
                    obj_top1 = accuracy(obj, output_action[:,step_i][:, 1].long())[0]

                    top1 = (action_top1 + obj_top1)/2
                    top1_all += top1
                    action_top1_all += action_top1
                    obj_top1_all += obj_top1

                loss_all = loss_all/batch_max_steps
                action_loss_all = action_loss_all/batch_max_steps
                obj_loss_all = obj_loss_all/batch_max_steps
                top1_all = top1_all/batch_max_steps
                action_top1_all = action_top1_all/batch_max_steps
                obj_top1_all = obj_top1_all/batch_max_steps

                if mode == 'train':
                    loss_all.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()


                recoder.update(B, loss=loss_all, action_loss=action_loss_all, obj_loss=obj_loss_all, 
                                  top1=top1_all, action_top1=action_top1_all, obj_top1=obj_top1_all)

                
                if i % 10 == 0:
                    print(
                        "{} {} \n \
                        Epoch {}/{} Updates {}/{} \n \
                        loss: {:.3f} action_loss: {:.3f} obj_loss: {:.3f} \n \
                        top1: {:.3f} action_top1: {:.3f} obj_top1: {:.3f}"
                            .format(mode.capitalize(), self.args.save_dir, e, self.args.train_epoch, i, len(trainloader),
                                    recoder.losses.avg, recoder.action_losses.avg, recoder.obj_losses.avg,
                                    recoder.top1s.avg, recoder.action_top1s.avg, recoder.obj_top1s.avg))

                    self.logger.log({"loss": recoder.losses.avg, "action_loss": recoder.action_losses.avg, "obj_loss": recoder.obj_losses.avg,
                                "top1": recoder.top1s.avg, "action_top1": recoder.action_top1s.avg, "obj_top1": recoder.obj_top1s.avg})
                    
                if self.args.debug:
                    break
            save_model(self.args, self, e, 0, is_best=False)

        output = [recoder.losses.avg, recoder.action_losses.avg, recoder.obj_losses.avg, 
                    recoder.top1s.avg, recoder.action_top1s.avg, recoder.obj_top1s.avg]
        
        return output





    def get_action(self, data, lstm_hidden=None):

        self.model.eval()
        self.action_criterion.eval()
        self.obj_criterion.eval()

        with torch.no_grad():
            for k, _ in enumerate(data):
                data[k] = torch.tensor(data[k]).float().cuda(non_blocking=True)
            B = data[1].shape[0]    

            action, obj = self.model(data)
            return [action], [obj]














