import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components_new import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, GMMBlock, NeXtVLAD

import ipdb



class GMMFormer_Net(nn.Module):
    def __init__(self, config):
        super(GMMFormer_Net, self).__init__()
        self.config = config

        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, event_size=config.event_size,map_size=config.map_size,batch_size=config.batch_size))

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size


    def forward(self, batch):

        clip_video_feat = batch['clip_video_features'] # [bs, 32, 512] # bs is the video number in the batch
        query_feat = batch['text_feat'] # [640, 512] # 640 is the query number in the batch
        query_mask = batch['text_mask'] # None
        query_labels = batch['text_labels'] # [640] video index in the batch

        frame_video_feat = batch['frame_video_features'] # [bs, 83, 512]
        frame_video_mask = batch['videos_mask'] # [bs, 83]

        vid_proposal_feat, gmm_masks, proposal_fea, video = self.encode_context(clip_video_feat) # [bs, 384], [bs, 32, 384],
       

        clip_scale_scores, clip_scale_scores_, clip_scores = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, proposal_fea, return_query_feats=True)
        # all shape are [640, bs]

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        video_query = self.encode_query(query_feat, query_mask)


        if proposal_fea is not None:

            moment_fea = proposal_fea  # moment_fea: 128*4*384
            v_bsz = moment_fea.size(0)
            modularied_query = F.normalize(video_query, dim=-1) # query:128*384
            q_bsz = video_query.size(0)
            shared_num = int(q_bsz / v_bsz)
            context_feat = F.normalize(moment_fea, dim=-1)
        
            clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)  # torch.Size([128, 4, 640])
            pos_score, max_indices = torch.max(clip_level_query_context_scores,dim=1)

            video_feat = F.normalize(video, dim=-1)
        
            query = modularied_query  # 640*512
            ref_query_context_scores = torch.matmul(video_feat, query.t()).permute(1, 0) 

            "loss"
            margin_1 = 0.02
            indices_1d = query_labels
            pos_moment_score = pos_score[torch.arange(q_bsz), indices_1d]
            query_context_scores = ref_query_context_scores[torch.arange(q_bsz), indices_1d]
            ref_loss = (margin_1 + query_context_scores - pos_moment_score).clamp(min=0).sum()  
            rank_loss = ref_loss / len(pos_moment_score)


        return [rank_loss, clip_scale_scores, clip_scale_scores_, label_dict, video_query, gmm_masks, clip_scores]


    def encode_query(self, query_feat, query_mask):
        video_query = self.query_input_proj(query_feat)

        return video_query

        
    def encode_context(self, clip_video_feat):

        encoded_clip_feat, gmm_masks, proposal_fea, video  = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)

        return encoded_clip_feat, gmm_masks, proposal_fea, video 

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        # return feat
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def get_modularized_frames(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        ### for test

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0) # [bs, 32, 640] 

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1) # [bs, 640] 


        return query_context_scores, clip_level_query_context_scores


    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):
        ### for train

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)




        return output_query_context_scores, query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, proposal_fea=None, return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask) # [640,384] sentence level
        
        # get clip-level retrieval scores
        clip_scale_scores, clip_scores = self.get_clip_scale_scores(       # [640,128]
            video_query, video_proposal_feat)

       
        if return_query_feats: 
            clip_scale_scores_, query_context_scores = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            
            return clip_scale_scores, clip_scale_scores_, query_context_scores # training
        else:

            return clip_scale_scores, clip_scores # test


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
