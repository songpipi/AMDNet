import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
        else:
            return denominator - nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator



class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)



class GMMBlock(nn.Module):
    def __init__(self, config):
        super(GMMBlock, self).__init__()
        self.attn0 = BertAttention(config)
        self.attn1 = BertAttention(config, mask=True)
        self.map_size = config.map_size
        self.event_size = config.event_size
        self.batch_size = config.batch_size
        self.clip2video = avgpooling(config.hidden_size, config.map_size)
        self.fc_gauss = nn.Linear(config.hidden_size, config.event_size*2)
    
    def generate_v_gauss_weight(self, props_len, center, width):

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        out = weight/weight.max(dim=-1, keepdim=True)[0] # num_mask, 32
        return  out.view(-1,self.event_size,self.map_size ) # b, num_mask, 32


    def soft_moment_mask(self, props_len, center, width):
        # ..... not complete
        center = torch.sigmoid(center) # num_mask
        width = torch.sigmoid(width) # num_mask  
        weight = torch.linspace(0, 1, props_len)
        out = torch.sigmoid()

        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        out = weight/weight.max(dim=-1, keepdim=True)[0] # num_mask, 32

        return  out.view(1,self.event_size,props_len) # 1, num_mask, 32



    def forward(self, input_tensor, attention_mask=None):
        # input_tensor
        video = self.clip2video(input_tensor)
        bsz = input_tensor.size(0) # 128
        modular_video = torch.sigmoid(self.fc_gauss(video).view(bsz*self.event_size, 2)) # (bs*4, 2)
        gauss_param = modular_video # (128*4,2) 512,2
        v_gauss_center = gauss_param[:, 0] # (b*K) 512
        v_gauss_width = gauss_param[:, 1] # 512  128*4

        v_gmm_masks = self.generate_v_gauss_weight(self.map_size, v_gauss_center, v_gauss_width) # bs, num_mask, 32 (128*4*32)
        proposal_fea = torch.matmul(v_gmm_masks,input_tensor) # 128*4*384


        _, center, width = self.generate_gauss_weight(self.map_size, self.gauss_center, self.gauss_width) # 1, num_mask, 32
        o0 = self.attn0(input_tensor, attention_mask).unsqueeze(-1)
        o1 = self.attn1(o0.squeeze(-1), attention_mask, v_gmm_masks).unsqueeze(-1)

        oo = torch.cat([o0, o1], dim=-1)
        out = torch.mean(oo, dim=-1).squeeze()
        return out, v_gmm_masks, proposal_fea, video

class avgpooling(nn.Module):
    def __init__(self, hidden_dim, map_size):
        super(avgpooling, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.fc_out1 = nn.Linear(hidden_dim, hidden_dim)
        self.pool = nn.AvgPool1d(kernel_size=map_size, stride=map_size) 

    def forward(self, query):

        bert_encoding = query.permute(0, 2, 1) # use avg_pool as the whole sentence feature
        x_pooled = self.pool(bert_encoding).permute(0, 2, 1)
        query = x_pooled.squeeze(1)  # [N, C]
        query = self.layernorm(query)
        sentence = self.fc_out1(query)

        return sentence


class BertAttention(nn.Module):
    def __init__(self, config, mask=False):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, mask=mask)
        self.output = BertSelfOutput(config)
        self.mask = mask

    def forward(self, input_tensor, attention_mask=None, v_gmm_masks=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask, v_gmm_masks)
        # if not self.mask:
        self_output = self.output(self_output, input_tensor)
        return self_output


class BertSelfAttention(nn.Module):
    def __init__(self, config, mask=False):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.mask= mask
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)



    def forward(self, query_states, key_states, value_states, attention_mask=None, v_gmm_masks=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)


        attention_scores = attention_scores_ori # B, num_head, 32, 32
        if self.mask:
            bsz = attention_scores_ori.shape[0]
            gmm_mask = v_gmm_masks.unsqueeze(2).repeat(1, 1, attention_scores.shape[-1], 1)
            attention_scores = attention_scores_ori * gmm_mask 

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

