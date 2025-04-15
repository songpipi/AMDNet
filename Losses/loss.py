import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools as it
from easydict import EasyDict as edict
from Models.gmmformer.model_components_new import clip_nce


class mask_div(nn.Module):
    def __init__(self, lambda_):
        torch.nn.Module.__init__(self)
        self.lambda_ = lambda_

    def forward(self, gauss_weight):
        num_props = gauss_weight.size(1)
        gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
        target = torch.eye(num_props).unsqueeze(0).cuda() * self.lambda_
        source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
        div_loss = torch.norm(target - source, dim=(1, 2))**2
        return div_loss.mean()




class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.div_criterion = mask_div(lambda_=0.15)

    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']
        
        [rank_loss, clip_scale_scores, clip_scale_scores_, label_dict, query, gauss_weight, clip_scores] = input_list

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_) 
        mask_div_loss = self.cfg['loss_factor'][1] * self.div_criterion(gauss_weight)
        match_loss = rank_loss * 25

        if rank_loss is None or str(rank_loss) == "nan":
            raise ValueError("rank_loss is None or nan")


        loss =  clip_nce_loss + mask_div_loss + match_loss 

        return loss
