import glob
import os

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np
import imageio
import cv2
import random
import matplotlib
import pickle


def save_model(args, agent, j, best_top1, is_best=False):
    saved_model_path = '/'.join(args.save_dir.split('/')[:-1] + ['saved_model_latest.p'] )
    torch.save([
        agent.model.state_dict(),
        agent.optimizer.state_dict(),
        j,
        best_top1,
    ], saved_model_path)
    print('saved model to %s' % saved_model_path)
    if j % 50 == 49:
        saved_model_path = '/'.join(args.save_dir.split('/')[:-1] + ['saved_model_%d_epoch.p' % j] )
        torch.save([
            agent.model.state_dict(),
            agent.optimizer.state_dict(),
            j,
            best_top1,
        ], saved_model_path)
        print('saved model to %s' % saved_model_path)


def load_pretrained_model(args, agent, gpu, logging):
    best_top1 = 0
    start_epoch = 0

    if args.pretrained_model_dir:
        if os.path.exists(args.pretrained_model_dir):
            logging.info('-----------------------------------------------------------------------------------')
            logging.info('loading pretrained model %s' % args.pretrained_model_dir)
            logging.info('-----------------------------------------------------------------------------------')
            
            if args.pretrained_model_dir.endswith('.p'):
                
                checkpoint = torch.load(args.pretrained_model_dir, map_location='cpu')
                agent.model.load_state_dict(checkpoint[0])
                agent.optimizer.load_state_dict(checkpoint[1])
                start_epoch = checkpoint[2]
                best_top1 = checkpoint[3]
            else:
                checkpoint = torch.load(args.pretrained_model_dir, map_location='cpu')
                agent.model.load_state_dict(checkpoint)

        else:
            logging.info('there is no pretrained model %s' % args.pretrained_model_dir)
            
    return agent, best_top1, start_epoch



def accuracy(output, target, topk=(1,), reduce=True):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    if not reduce:
        res = []
        for k in topk:
            res.append(correct[:k])
    else:
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res


class RecordLossTop1(object):
    def __init__(self, args):
        self.args = args
        self.n_agent = args.n_agent
        
        self.losses = AverageMeter()
        self.action_losses = AverageMeter()
        self.obj_losses = AverageMeter()
        
        self.top1s = AverageMeter()
        self.action_top1s = AverageMeter()
        self.obj_top1s = AverageMeter()
        

    def update(self, B, loss=None, action_loss=None, obj_loss=None, top1=None, action_top1=None, obj_top1=None):
    
        self.action_losses.update(action_loss.item(), B)
        self.action_top1s.update(action_top1.item(), B)

        self.obj_losses.update(obj_loss.item(), B)
        self.obj_top1s.update(obj_top1.item(), B)

        self.losses.update(loss.item(), B)
        self.top1s.update(top1.item(), B)

        



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def mask_topk(logits, k=5, dim=-1):
    top_values, top_indices = torch.topk(logits, 3, dim=dim)
    kth_best = top_values[:, -1].view([-1, 1])
    kth_best = kth_best.repeat([1, logits.shape[dim]]).float()
    ignore = torch.lt(logits, kth_best)
    logits = logits.masked_fill(ignore, -99999)
    return logits

def sample_topk(logits,k=5,dim=-1):
    dist = torch.distributions.Multinomial(logits=mask_topk(logits,k=k,dim=dim), total_count=1)
    toks = torch.argmax(dist.sample(), dim=dim)
    return toks
