import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import math
import dgl.function as fn
from functools import partial
import math
class BN1D(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)
    def forward(self, x):
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if len(x.shape) == 2:
            return self.bn(x.unsqueeze(-1)).squeeze(-1)
        
class SEBlock(nn.Module):
    '''
    用于实现注意力机制中的 Squeeze-and-Excitation（SE）模块
    包括两个关键步骤：Squeeze 和 Excitation
    SE 模块最后将这些学习到的通道注意力权重乘以原始的特征图，以产生最终的加权特征表示
    '''
    def __init__(self, channels):
        super().__init__()
        self.conv_se1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.conv_se2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)       #输出的最低维度变为1
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv_se2(self.act(self.conv_se1(self.avg_pool(x))))) * x   #特征缩放

class SEBasicBlock(nn.Module):
    '''
    一个基本的残差块，可以避免梯度消失，构建更深层次的网络
    '''
    def __init__(self, in_c, out_c, temporal_length, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv1d(kernel_size=3, in_channels=in_c, out_channels=out_c, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.conv2 = nn.Conv1d(kernel_size=3, in_channels=out_c, out_channels=out_c, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        
        self.stride = stride
        self.downsample = None
        self.se = SEBlock(in_c)
        if stride != 1:
            if temporal_length == 16 or temporal_length == 8 or temporal_length == 4 or temporal_length == 2:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
            else:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=1, count_include_pad=False), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
        elif in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))     
        if self.in_c == self.out_c:
            out = self.se(out)
            out = out + identity
        return out


class TemporalBlock(nn.Module):
    def __init__(self, inplanes, temporal_length, args):
        super(TemporalBlock, self).__init__()                                
        self.layers = [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=1)]
        self.layers += [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=2)]
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        out = x
        for _ in range(len(self.layers)):
            out = self.layers[_](out)
        return out

class H_AgentTemporalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        kernel_size_dic = {0:16, 1:8, 2:4, 3:2, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}
        self.temporal_layers = torch.nn.ModuleList([TemporalBlock(args.hidden_dim, kernel_size_dic[_], args) for _ in range(2)])
    def forward(self, feat):
        feat = self.temporal_layers[0](feat)
        for _ in range(1, len(self.temporal_layers)):
            feat = self.temporal_layers[_](feat)
        feat = feat[..., -1] ## Last timestep
        return feat

class F_AgentTemporalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        kernel_size_dic = {0:25, 1:13, 2:7, 3:4, 4:2, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}
        self.temporal_layers = torch.nn.ModuleList([TemporalBlock(args.hidden_dim, kernel_size_dic[_], args) for _ in range(2)])
    def forward(self, feat):
        feat = self.temporal_layers[0](feat)
        for _ in range(1, len(self.temporal_layers)):
            feat = self.temporal_layers[_](feat)
        feat = feat[..., -1]
        return feat

class AttentionTemporalEncoder(nn.Module):
    def __init__(self, args):
        super(AttentionTemporalEncoder, self).__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.hidden_dim = args.hidden_dim
        self.query_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value_linear = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.out_linear = nn.Linear(args.hidden_dim, args.hidden_dim)

    def forward(self, x, mask=None):
        # x: [seq_len, batch_size, embed_dim]
        Q = self.query_linear(x.transpose(1, 2)).view(x.shape[0], x.shape[-1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        K = self.key_linear(x.transpose(1, 2)).view(x.shape[0], x.shape[-1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        V = self.value_linear(x.transpose(1, 2)).view(x.shape[0], x.shape[-1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # mask 的形状应该是 [batch_size, 1, 1, seq_len]，广播到 attention_scores
            mask = mask.unsqueeze(1).unsqueeze(2)  # 扩展维度以便与注意力分数相加
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # 使用 softmax 对注意力分数进行归一化
        attention_scores = torch.clamp(attention_scores, min=-50, max=50)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[-1], self.hidden_dim)
        output = self.out_linear(attention_output)  # [batch_size, seq_len, embed_dim
        output, _ = output.max(dim=1)
        return output

class BN1D(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)          #归一化处理,input:(N,C,L),C is the number of features.
    def forward(self, x):
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if len(x.shape) == 2:
            is_single_input = (x.shape[0] == 1 and self.training == True)
            if is_single_input:
                self.bn.eval()
            res = self.bn(x.unsqueeze(-1)).squeeze(-1)
            if is_single_input:
                self.bn.train()
            return res
        
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, norm=None, dropout=0.0, prenorm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out) # position-wise
        self.act = nn.ReLU(inplace=True)
        self.prenorm = prenorm
        if norm is None:
            self.norm = nn.Identity()
        elif prenorm:
            self.norm = norm(d_in)
        else:
            self.norm = norm(d_hid)
        self.dropout = nn.Identity()
    def forward(self, x):
        if self.prenorm:
            output = self.dropout(self.w_2(self.act(self.w_1(self.norm(x)))))
        else:
            output = self.dropout(self.w_2(self.act(self.norm(self.w_1(x)))))
        return output


class PointNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc2 = MLP(d_in=args.hidden_dim//8, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc3 = MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//4, norm=BN1D)
    def forward(self, x):
        out = self.fc1(x)
        out = torch.cat([out, self.fc2(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        out = torch.cat([out, self.fc3(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        return out

class Agent2embedding(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        ## -4 - minus coor and mask, args.hidden_dim//4 - coordinate feature
        self.fea_MLP = MLP(input_dim-4+args.hidden_dim//4, args.hidden_dim//2, args.hidden_dim, norm=BN1D)
    def forward(self, input_dic, shared_coor_encoder):
        #0-2 x, y, z, 3-4 vx, vy, 5-6 cos, sin, 7-9 witdth, length, height 10 mask
        x = input_dic["graph_lis"].ndata["a_n_fea"]["agent"]
        #z, vx, vy, cos, sin, width, length, height, mask
        coor_fea = shared_coor_encoder(x[..., :3])
        fea = torch.cat([coor_fea, x[..., 3:-1]], dim=-1)
        fea = self.fea_MLP(fea)            
        return fea.transpose(1, 2)

class Road2embedding(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        ## -4 - minus coor and mask, args.hidden_dim//4 - coordinate feature
        self.fea_MLP = MLP(input_dim-4+args.hidden_dim//4, args.hidden_dim//2, args.hidden_dim, norm=BN1D)
    def forward(self, input_dic, shared_coor_encoder):
        #0-2 x, y, z, 3-4 vx, vy, 5-6 cos, sin, 7-9 witdth, length, height 10 mask
        x_fea = torch.empty((0, 16, self.input_dim-1))
        x = input_dic["graph_lis"].ndata["a_n_fea"]["road"]
        #z, vx, vy, cos, sin, width, length, height, mask
        if x.shape[0] > 0:
            coor_fea = shared_coor_encoder(x[..., :3])
            x_fea = torch.cat([coor_fea, x[..., 3:]], dim=-1)
            x_fea = self.fea_MLP(x_fea)             
        return x_fea.transpose(1, 2)

## Centerline (Lane) Embedding
class Lane2embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        ## Encoder Coordinate Information
        self.pointnet = PointNet(args)
        self.point_out_fc = nn.Linear(args.hidden_dim//2, args.hidden_dim)
        self.type_emb =  torch.nn.Embedding(num_embeddings=4, embedding_dim=args.hidden_dim//4)
        self.stop_fc = nn.Linear(args.hidden_dim//4, args.hidden_dim//4)
        self.signal_fc = nn.Linear(args.hidden_dim//4, args.hidden_dim//4)
        self.signal_emb = torch.nn.Embedding(num_embeddings=9, embedding_dim=args.hidden_dim//2)
        self.signal_gru = torch.nn.GRU(input_size=args.hidden_dim//2, hidden_size=args.hidden_dim//2, num_layers=1, batch_first=True)  
        
        ## [coor, type, stop_signal]
        self.lane_n_out_fc = MLP(d_in=args.hidden_dim+args.hidden_dim//4+self.hidden_dim//4*3+self.hidden_dim//2, d_hid=args.hidden_dim*4, d_out=args.hidden_dim, norm=BN1D)
        
        self.boudary_emb = torch.nn.Embedding(num_embeddings=11, embedding_dim=args.hidden_dim//4)
        ## [boundary, n_fea, rel_pos_fea]
        self.lane_e_out_fc = MLP(d_in=args.hidden_dim+args.hidden_dim//4+args.hidden_dim//4, d_hid=args.hidden_dim*4, d_out=args.hidden_dim, norm=BN1D)

    def forward(self, input_dic, shared_coor_encoder, shared_rel_encoder):
        coor_fea = input_dic["graph_lis"].ndata["l_n_coor_fea"]["lane"]
        coor_fea = shared_coor_encoder(coor_fea)
        coor_fea = self.pointnet(coor_fea)
        coor_fea = self.act(self.point_out_fc(coor_fea.max(dim=1)[0]))
        type_fea = self.type_emb(input_dic["graph_lis"].ndata["l_n_type_fea"]["lane"])

        lane_n_num = coor_fea.shape[0]
        ## If there is no stop sign/traffic signal controlling the lane, the cooresponding features are all zeros 
        stop_signal_fea = torch.zeros((lane_n_num, self.hidden_dim//4*3+self.hidden_dim//2), device="cuda:"+str(input_dic["gpu"]))  
        if "lane_n_stop_sign_fea_lis" in input_dic:
            stop_sign_fea = input_dic["lane_n_stop_sign_fea_lis"]
            stop_sign_fea = shared_coor_encoder(stop_sign_fea)
            stop_sign_fea = self.act(self.stop_fc(stop_sign_fea))
            stop_signal_fea[input_dic["lane_n_stop_sign_index_lis"]][..., :self.hidden_dim//4] += stop_sign_fea
        if "lane_n_signal_fea_lis" in input_dic:
            signal_fea = input_dic["lane_n_signal_fea_lis"]
            signal_coor_fea = self.act(self.signal_fc(shared_coor_encoder(signal_fea[..., :3])))
            signal_dynamic = self.signal_emb(signal_fea[..., 3:].long())
            signal_dynamic, _ = self.signal_gru(signal_dynamic)
            signal_fea = torch.cat([signal_coor_fea, self.act(signal_dynamic[:, -1, :])], dim=-1)
            stop_signal_fea[input_dic["lane_n_signal_index_lis"]][..., self.hidden_dim//4:self.hidden_dim//4*2+self.hidden_dim//2] += signal_fea
        
        output_n_fea = self.lane_n_out_fc(torch.cat([coor_fea, type_fea, stop_signal_fea], dim=-1))
        
        ## Lane Edge Feature Encoding
        lane_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["l_e_fea"][_]) for _ in [("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]])  
        lane_e_rel_pos = torch.cat([input_dic["graph_lis"].edata["l_e_fea"][_] for _ in [("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]], dim=0)
        lane_e_rel_pos_fea = shared_rel_encoder(lane_e_rel_pos)
        lane_e_num = lane_e_rel_pos_fea.shape[0]
        lane_src_indices = torch.cat([input_dic["graph_lis"].edges(etype=_)[0]  for _ in ["l2a", "left", "right", "prev", "follow"]], dim=0)
        lane_e_src_n_fea = output_n_fea[lane_src_indices]
        
        output_e_fea = torch.cat([torch.zeros((lane_e_num, self.hidden_dim//4), device="cuda:"+str(input_dic["gpu"])), lane_e_rel_pos_fea, lane_e_src_n_fea,], dim=-1)
        
        boudary_emb = torch.cat([input_dic["graph_lis"].edata["boundary_type"][_] for _ in [("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]], dim=0)
        boudary_emb = self.boudary_emb(boudary_emb)
        output_e_fea[lane_e_num_lis_by_etype[1]:, :self.hidden_dim//4] += boudary_emb
        output_e_fea = self.lane_e_out_fc(output_e_fea)
        
        input_dic["graph_lis"].ndata["l_n_hidden"] = {"lane":output_n_fea}
        for _index, _ in enumerate([("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]):
            input_dic["graph_lis"].edata["l_e_hidden"] = {_:output_e_fea[lane_e_num_lis_by_etype[_index]:lane_e_num_lis_by_etype[_index+1]]}
        return None

class Polygon2embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.pointnet = PointNet(args)
        self.type_emb = torch.nn.Embedding(num_embeddings=14, embedding_dim=args.hidden_dim//2)
        self.out_fc = MLP(args.hidden_dim//2+args.hidden_dim//2, args.hidden_dim*4, args.hidden_dim, norm=BN1D)
    
    def forward(self, input_dic, shared_coor_encoder):
        coor_fea = input_dic["graph_lis"].edata['g2a_e_fea'][("polygon", "g2a", "agent")]
        coor_fea = shared_coor_encoder(coor_fea)
        coor_fea = self.pointnet(coor_fea).max(dim=1)[0]
        type_fea = input_dic["graph_lis"].edata['g2a_e_type'][("polygon", "g2a", "agent")]
        type_fea = self.type_emb(type_fea)
        fea = torch.cat([coor_fea, type_fea], axis=-1)
        fea = self.out_fc(fea)
        input_dic["graph_lis"].edata["g_e_hidden"] = {("polygon", "g2a", "agent"):fea}


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, args):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(args.dropout)
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))     #bmm:批次矩阵乘法
        n = float(v.shape[1])
        attn = attn / self.temperature * math.log(n+1, 32)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout, args):
        super().__init__()
        self.d_in = d_in
        self.out_dim = d_in
        self.norm = nn.LayerNorm(d_in)
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.w3 = nn.Linear(d_in, d_hid)
        self.act = nn.SiLU()
        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    def forward(self, x):
        residual = x
        output = self.norm(x)
        output = self.w2(self.act(self.w1(output)) * self.w3(output))
        output = self.dropout(output) + residual
        return output

class LaneHetGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        self.etype_num = 5
        self.norm = nn.LayerNorm
        self.node_mlp = nn.ModuleDict({
            _:MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, norm=nn.LayerNorm, prenorm=True)
            for _ in ["left", "right", "prev", "follow"]})
        
        self.node_mlp["a2l"] = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, norm=nn.LayerNorm, prenorm=True)
        for _ in range(3)])

        self.node_fc = nn.Linear(self.etype_num*args.hidden_dim, args.hidden_dim)
        self.node_ffn = PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, args)

        self.etype_dic = {}
        for etype in ["left", "right", "prev", "follow", "a2l"][:self.etype_num]:
            self.etype_dic[etype] = (partial(self.message_func, etype=etype), partial(self.reduce_func, etype=etype))           #partial的功能：把函数的某些参数固定住，返回一个新函数
        
        self.edge_MLP = MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, norm=nn.LayerNorm, dropout=args.dropout, prenorm=True)
        
        self.agent_edge_MLP = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, norm=nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])

    def forward(self, input_dic):
        with input_dic["graph_lis"].local_scope():
            self.gpu = input_dic["gpu"]
            self.a_e_type_dict = input_dic["a_e_type_dict"]

            input_dic["graph_lis"].multi_update_all(etype_dict=self.etype_dic, cross_reducer="stack")  ## Stack all features of all types of in-edges  节点信息更新
            output_lane_n_fea = input_dic["graph_lis"].ndata["l_n_hidden_out"]["lane"]
            
            output_lane_n_fea = self.node_fc(output_lane_n_fea.view(output_lane_n_fea.shape[0], -1)) + input_dic["graph_lis"].ndata["l_n_hidden"]["lane"]
            output_lane_n_fea = self.node_ffn(output_lane_n_fea)
            
            output_e_fea = [self.edge_MLP(torch.cat([input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype=_)[0]], input_dic["graph_lis"].edges[_].data["l_e_hidden"], input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype=_)[1]]], dim=-1))+input_dic["graph_lis"].edges[_].data["l_e_hidden"] for _ in ["left", "right", "prev", "follow"]]
            #a2l时只考虑agne的历史时域特征
            if self.use_planning:
                agent2lane_e_fea = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="a2l")[0]][..., 0], input_dic["graph_lis"].edges["a2l"].data["a_e_hidden"], input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype="a2l")[1]]], dim=-1)
            else:
                agent2lane_e_fea = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="a2l")[0]], input_dic["graph_lis"].edges["a2l"].data["a_e_hidden"], input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype="a2l")[1]]], dim=-1)
            output_agent2lane_e_fea = torch.zeros_like(input_dic["graph_lis"].edges["a2l"].data["a_e_hidden"])
            for agent_type_index in range(3):
                if len(self.a_e_type_dict["a2l"][agent_type_index]) != 0:
                    output_agent2lane_e_fea[self.a_e_type_dict["a2l"][agent_type_index]] = self.agent_edge_MLP[agent_type_index](agent2lane_e_fea[self.a_e_type_dict["a2l"][agent_type_index]]) +input_dic["graph_lis"].edges["a2l"].data["a_e_hidden"][self.a_e_type_dict["a2l"][agent_type_index]]
            output_e_fea.append(output_agent2lane_e_fea)
        return output_lane_n_fea, output_e_fea

    def message_func(self, edges, etype):
        '''
        定义消息传递函数
        '''
        if etype != "a2l":
            return {"l_e_hidden_"+etype:self.node_mlp[etype](edges.data["l_e_hidden"])}
        else:
            tmp_out = torch.zeros_like((edges.data["a_e_hidden"]))
            tmp_input = edges.data["a_e_hidden"]
            for agent_type_index in range(3):
                if len(self.a_e_type_dict[etype][agent_type_index]) != 0:
                    tmp_out[self.a_e_type_dict[etype][agent_type_index]] = self.node_mlp[etype][agent_type_index](tmp_input[self.a_e_type_dict[etype][agent_type_index]])
            return {"l_e_hidden_"+etype:tmp_out}

    def reduce_func(self, nodes, etype):
        '''
        定义节点聚合函数
        '''
        return {"l_n_hidden_out":nodes.mailbox["l_e_hidden_"+etype].max(dim=1)[0]}     #nodes.mailbox存储了从相邻节点收集来的信息，第一维是每种类型的节点数量


class AgentHetGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_road_obs = args.use_road_obs
        self.use_map = args.use_map
        self.use_planning = args.use_planning
        self.norm = nn.LayerNorm
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = args.hidden_dim
        self.n_head = self.d_model//args.head_dim
        d_k = self.head_dim
        d_v = self.head_dim
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)

        if self.use_planning:
            if self.use_road_obs:
                self.wqs = nn.ModuleList([nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 5, bias=False),
                )
                    for _ in range(3)
                ])
            else:
                self.wqs = nn.ModuleList([nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 4, bias=False),
                )
                    for _ in range(3)
                ])
            if self.use_map:
                self.wkvs = nn.ModuleDict({
                    _: nn.Sequential(
                        self.norm(self.d_model),
                        nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                    )
                    for _ in ["l2a", "g2a"]
                })
            self.wkvs["planning"] = nn.ModuleList([
                nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                )
                for _ in range(3)
            ])
            if self.use_map:
                self.attn_fcs = nn.ModuleList([nn.ModuleDict({
                    _: nn.Sequential(
                        nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                        nn.ReLU(inplace=True)
                    )
                    for _ in ["l2a", "g2a", "other", "planning"]
                })
                    for agent_type_index in range(3)]
                )
                self.out_fc = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim * 5, self.hidden_dim, bias=True),
                    nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                )
                    for _ in range(3)])
                if self.use_road_obs:
                    self.out_fc_view = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.hidden_dim * 6, self.hidden_dim, bias=True),
                        nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                    )
                        for _ in range(3)])
            else:
                self.attn_fcs = nn.ModuleList([nn.ModuleDict({
                    _: nn.Sequential(
                        nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                        nn.ReLU(inplace=True)
                    )
                    for _ in ["other", "planning"]
                })
                    for agent_type_index in range(3)]
                )

                self.out_fc = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim * 3, self.hidden_dim, bias=True),
                    nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                )
                    for _ in range(3)])
                if self.use_road_obs:
                    self.out_fc_view = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.hidden_dim * 4, self.hidden_dim, bias=True),
                        nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                    )
                        for _ in range(3)])

        else:
            if self.use_road_obs:
                self.wqs = nn.ModuleList([nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 4, bias=False),
                )
                    for _ in range(3)
                ])
            else:
                self.wqs = nn.ModuleList([nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 3, bias=False),
                )
                    for _ in range(3)
                ])
            if self.use_map:
                self.wkvs = nn.ModuleDict({
                    _: nn.Sequential(
                        self.norm(self.d_model),
                        nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                    )
                    for _ in ["l2a", "g2a"]
                })
            else:
                self.wkvs = nn.ModuleList({nn.ModuleList([
                    nn.Sequential(
                        self.norm(self.d_model),
                        nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                    )
                    for _ in range(3)
                ])for _ in ["other"]
                })
            if self.use_map:
                self.attn_fcs = nn.ModuleList([nn.ModuleDict({
                    _: nn.Sequential(
                        nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                        nn.ReLU(inplace=True)
                    )
                    for _ in ["l2a", "g2a", "other"]
                })
                    for agent_type_index in range(3)]
                )
                self.out_fc = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim, bias=True),
                    nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                )
                    for _ in range(3)])
                if self.use_road_obs:
                    self.out_fc_view = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.hidden_dim * 5, self.hidden_dim, bias=True),
                        nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                    )
                        for _ in range(3)])
            else:
                self.attn_fcs = nn.ModuleList([nn.ModuleDict({
                    _: nn.Sequential(
                        nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                        nn.ReLU(inplace=True)
                    )
                    for _ in ["other"]
                })
                    for agent_type_index in range(3)]
                )

                self.out_fc = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=True),
                    nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                )
                    for _ in range(3)])
                if self.use_road_obs:
                    self.out_fc_view = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.hidden_dim * 3, self.hidden_dim, bias=True),
                        nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
                    )
                        for _ in range(3)])

        
        if self.use_map:
            self.wkvs["other"] = nn.ModuleList([
                nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                    )
            for _ in range(3)
            ])
        if self.use_road_obs:
            self.wkvs["view"] = nn.ModuleList([
                nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                )
                for _ in range(3)
            ])
            for agent_type_index in range(3):
                self.attn_fcs[agent_type_index].update({"view": nn.Sequential(
                        nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                        nn.ReLU(inplace=True))})

        self.self_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True)
            )
        for _ in range(3)]
        )

                
        self.out_ffn = nn.ModuleList([PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, nn.LayerNorm)
        for _ in range(3)
        ])
        self.etype_dic = {}                                         
        self.etype2hidden_name = {}
        self.etype2src_name = {}
        self.edge_MLP = {}
        if self.use_road_obs:
            self.etype_dic["view"] = (partial(self.message_func, etype="view"), partial(self.reduce_func, etype="view"))
            self.etype2hidden_name = {"view": "a_e_hidden"}
            self.etype2src_name = {"view": "road"}
            self.edge_MLP["view"] = nn.ModuleList([MLP(args.hidden_dim * 3, args.hidden_dim * 4, args.hidden_dim,
                                                           nn.LayerNorm, dropout=args.dropout, prenorm=True) for _ in range(3)])
        if self.use_map:
            if self.use_planning:
                for etype in ["other", "l2a", "g2a", "planning"]:
                    self.etype_dic[etype] = (partial(self.message_func, etype=etype), partial(self.reduce_func, etype=etype))
                self.etype2hidden_name.update({"other": "a_e_hidden", "l2a": "l_e_hidden", "g2a": "g_e_hidden", "planning": "a_e_hidden"})
                self.etype2src_name.update({"other": "agent", "l2a": "lane", "g2a": "polygon", "planning": "agent"})
            else:
                for etype in ["other", "l2a", "g2a"]:
                    self.etype_dic[etype] = (partial(self.message_func, etype=etype), partial(self.reduce_func, etype=etype))
                self.etype2hidden_name.update({"other": "a_e_hidden", "l2a": "l_e_hidden", "g2a": "g_e_hidden"})
                self.etype2src_name.update({"other": "agent", "l2a": "lane", "g2a": "polygon"})

            self.edge_MLP.update({
                "l2a":MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True),
                "g2a":MLP(args.hidden_dim*2, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True),
            })
        else:
            if self.use_planning:
                for etype in ["other", "planning"]:
                    self.etype_dic[etype] = (
                    partial(self.message_func, etype=etype), partial(self.reduce_func, etype=etype))
                self.etype2hidden_name.update({"other": "a_e_hidden", "planning": "a_e_hidden"})
                self.etype2src_name.update({"other": "agent", "planning": "agent"})
            else:
                for etype in ["other"]:
                    self.etype_dic[etype] = (
                    partial(self.message_func, etype=etype), partial(self.reduce_func, etype=etype))
                self.etype2hidden_name.update({"other": "a_e_hidden"})
                self.etype2src_name.update({"other": "agent"})
            self.edge_MLP = {}
        self.edge_MLP["self"] = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])        
        self.edge_MLP["other"] = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])
        if self.use_planning:
            self.edge_MLP["planning"] = nn.ModuleList([MLP(args.hidden_dim * 3, args.hidden_dim * 4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
            for _ in range(3)
            ])
        self.edge_MLP = nn.ModuleDict(self.edge_MLP)

    def forward(self, input_dic):
        self.a_e_type_dict = input_dic["a_e_type_dict"]
        self.a_n_type_lis = input_dic["a_n_type_lis"] 
        with input_dic["graph_lis"].local_scope():            
            self.device = "cuda:"+str(input_dic["gpu"])
            
            if self.use_planning:
                input_q = input_dic["graph_lis"].ndata["a_n_hidden"]["agent"]
                if self.use_road_obs:
                    q = torch.zeros((input_dic["graph_lis"].ndata["a_n_hidden"]["agent"].shape[0], self.n_head * self.head_dim * 5), device=self.device)
                    for agent_type_index in range(3):
                        if len(self.a_n_type_lis[agent_type_index]) != 0: 
                            q[self.a_n_type_lis[agent_type_index]] = self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]][..., 0])
                    other_q, lane_q, polygon_q, planning_q, view_q = q.view(input_q.shape[0], self.n_head, self.head_dim * 5).split(dim=-1, split_size=self.head_dim)
                    input_dic["graph_lis"].nodes["agent"].data["view_q"] = view_q
                else:
                    q = torch.zeros((input_dic["graph_lis"].ndata["a_n_hidden"]["agent"].shape[0], self.n_head * self.head_dim * 4), device=self.device)
                    for agent_type_index in range(3):
                        if len(self.a_n_type_lis[agent_type_index]) != 0:          
                            q[self.a_n_type_lis[agent_type_index]] = self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]][..., 0])
                    other_q, lane_q, polygon_q, planning_q = q.view(input_q.shape[0], self.n_head, self.head_dim*4).split(dim=-1, split_size=self.head_dim)
                input_dic["graph_lis"].nodes["agent"].data["planning_q"] = planning_q
            else:
                input_q = input_dic["graph_lis"].ndata["a_n_hidden"]["agent"]
                if self.use_road_obs:
                    q = torch.zeros((input_dic["graph_lis"].ndata["a_n_hidden"]["agent"].shape[0], self.n_head * self.head_dim * 4), device=self.device)
                    for agent_type_index in range(3):
                        if len(self.a_n_type_lis[agent_type_index]) != 0: 
                            q[self.a_n_type_lis[agent_type_index]] = self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]])
                    other_q, lane_q, polygon_q, view_q = q.view(input_q.shape[0], self.n_head, self.head_dim * 4).split(dim=-1, split_size=self.head_dim)
                    input_dic["graph_lis"].nodes["agent"].data["view_q"] = view_q
                else:
                    q = torch.zeros((input_dic["graph_lis"].ndata["a_n_hidden"]["agent"].shape[0], self.n_head * self.head_dim * 3), device=self.device)
                    for agent_type_index in range(3):
                        if len(self.a_n_type_lis[agent_type_index]) != 0:  
                            q[self.a_n_type_lis[agent_type_index]] = self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]])
                    other_q, lane_q, polygon_q = q.view(input_q.shape[0], self.n_head, self.head_dim * 3).split(dim=-1, split_size=self.head_dim)

            input_dic["graph_lis"].nodes["agent"].data["other_q"] = other_q
            if self.use_map:
                input_dic["graph_lis"].nodes["agent"].data["l2a_q"] = lane_q
                input_dic["graph_lis"].nodes["agent"].data["g2a_q"] = polygon_q
            input_dic["graph_lis"].multi_update_all(etype_dict=self.etype_dic, cross_reducer="stack")
            all_out_fea = input_dic["graph_lis"].ndata["a_n_hidden_out"]["agent"]
            index = 0
            if self.use_road_obs:
                if input_dic["graph_lis"].edges["view"].data["a_e_hidden"].shape[0] != 0:
                    view_out_tmp_in = all_out_fea[:, index, :]
                    index += 1
            if self.use_map:
                if input_dic["graph_lis"].edges["other"].data["a_e_hidden"].shape[0] != 0:
                    other_out_tmp_in = all_out_fea[:, 0, :]
                    index += 1
                else:
                    other_out_tmp_in = torch.zeros((all_out_fea.shape[0], self.hidden_dim), device=self.device)
                if input_dic["graph_lis"].edges["l2a"].data["l_e_hidden"].shape[0] != 0:
                    lane_out_fea_tmp_in = all_out_fea[:, index, :]
                    index += 1
                else:
                    lane_out_fea_tmp_in = torch.zeros((all_out_fea.shape[0], self.hidden_dim), device=self.device)
                polygon_out_fea_tmp_in = all_out_fea[:, index, :] if input_dic["graph_lis"].edges["g2a"].data["g_e_hidden"].shape[0] != 0 else torch.zeros((all_out_fea.shape[0], self.hidden_dim), device=self.device)
            else:
                if input_dic["graph_lis"].edges["other"].data["a_e_hidden"].shape[0] != 0:
                    other_out_tmp_in = all_out_fea[:, 0, :]
                    index += 1
                else:
                    other_out_tmp_in = torch.zeros((all_out_fea.shape[0], self.hidden_dim), device=self.device)
            if self.use_planning:
                planning_out_tmp_in = all_out_fea[:, -1, :]
            input_self_fea = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
            all_out_n_fea = torch.zeros_like(input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"])
            for agent_type_index in range(3):
                if len(self.a_n_type_lis[agent_type_index]) != 0:
                    other_out_fea = self.attn_fcs[agent_type_index]["other"](other_out_tmp_in[self.a_n_type_lis[agent_type_index]])
                    if self.use_road_obs:
                        view_out_fea = self.attn_fcs[agent_type_index]["view"](view_out_tmp_in[self.a_n_type_lis[agent_type_index]])
                    if self.use_map:
                        lane_out_fea = self.attn_fcs[agent_type_index]["l2a"](lane_out_fea_tmp_in[self.a_n_type_lis[agent_type_index]])
                        polygon_out_fea = self.attn_fcs[agent_type_index]["g2a"](polygon_out_fea_tmp_in[self.a_n_type_lis[agent_type_index]])
                    if self.use_planning:
                        planning_out_fea = self.attn_fcs[agent_type_index]["planning"](planning_out_tmp_in[self.a_n_type_lis[agent_type_index]])
                    self_fea = self.self_fc[agent_type_index](input_self_fea[self.a_n_type_lis[agent_type_index]])
                    if self.use_map:
                        if self.use_planning:
                            out_his_n_fea = torch.stack([self_fea, other_out_fea, lane_out_fea, polygon_out_fea, planning_out_fea], dim=1)
                            if self.use_road_obs:
                                out_his_n_fea = torch.cat([out_his_n_fea, view_out_fea.unsqueeze(1)], dim=1)
                                out_his_n_fea = self.out_fc_view[agent_type_index](out_his_n_fea.view(out_his_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]][..., 0]
                            else:
                                out_his_n_fea = self.out_fc[agent_type_index](out_his_n_fea.view(out_his_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]][..., 0]
                            out_his_n_fea = self.out_ffn[agent_type_index](out_his_n_fea)
                            out_planning_n_fea = planning_out_fea + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]][..., 1]
                            out_planning_n_fea = self.out_ffn[agent_type_index](out_planning_n_fea)
                            out_n_fea = torch.stack([out_his_n_fea, out_planning_n_fea], dim=-1)
                        else:
                            out_n_fea = torch.stack([self_fea, other_out_fea, lane_out_fea, polygon_out_fea], dim=1)
                            if self.use_road_obs:
                                out_n_fea = torch.cat([out_n_fea, view_out_fea.unsqueeze(1)], dim=1)
                                out_n_fea = self.out_fc_view[agent_type_index](out_n_fea.view(out_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]]
                            else:
                                out_n_fea = self.out_fc[agent_type_index](out_n_fea.view(out_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]]
                            out_n_fea = self.out_ffn[agent_type_index](out_n_fea)
                    else:
                        if self.use_planning:
                            out_his_n_fea = torch.stack([self_fea, other_out_fea, planning_out_fea], dim=1)
                            out_his_n_fea = self.out_fc[agent_type_index](out_his_n_fea.view(out_his_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]][..., 0]
                            out_his_n_fea = self.out_ffn[agent_type_index](out_his_n_fea)
                            out_planning_n_fea = planning_out_fea + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]][..., 1]
                            out_planning_n_fea = self.out_ffn[agent_type_index](out_planning_n_fea)
                            out_n_fea = torch.stack([out_his_n_fea, out_planning_n_fea], dim=-1)
                        else:
                            out_n_fea = torch.stack([self_fea, other_out_fea], dim=1)
                            out_n_fea = self.out_fc[agent_type_index](out_n_fea.view(out_n_fea.shape[0], -1)) + input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][self.a_n_type_lis[agent_type_index]]
                            out_n_fea = self.out_ffn[agent_type_index](out_n_fea)

                    all_out_n_fea[self.a_n_type_lis[agent_type_index]] = out_n_fea

            out_e_fea_lis = []
            if self.use_planning:
                self_e_fea_tmp_in = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
                other_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="other")[0], :, 0], input_dic["graph_lis"].edges["other"].data["a_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="other")[1], :, 0]], dim=-1)
                planning_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="planning")[0], :, 1], input_dic["graph_lis"].edges["planning"].data["a_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="planning")[1], :, 0]], dim=-1)
                if self.use_road_obs:
                    view_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["road"][input_dic["graph_lis"].edges(etype="view")[0], :], input_dic["graph_lis"].edges["view"].data["a_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="view")[1], :, 0]], dim=-1)
                    view_e_fea_tmp_out = torch.zeros((view_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)

                self_e_fea_tmp_out = torch.zeros_like(self_e_fea_tmp_in)
                other_e_fea_tmp_out = torch.zeros((other_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
                planning_e_fea_tmp_out = torch.zeros((planning_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
                for agent_type_index in range(3):
                    if len(self.a_e_type_dict["self"][agent_type_index]) != 0:
                        self_e_fea_tmp_out[self.a_e_type_dict["self"][agent_type_index]] = self.edge_MLP["self"][agent_type_index](self_e_fea_tmp_in[self.a_e_type_dict["self"][agent_type_index]])
                    if len(self.a_e_type_dict["other"][agent_type_index]) != 0:      
                        other_e_fea_tmp_out[self.a_e_type_dict["other"][agent_type_index]] = self.edge_MLP["other"][agent_type_index](other_e_fea_tmp_in[self.a_e_type_dict["other"][agent_type_index]])
                    if len(self.a_e_type_dict["planning"][agent_type_index]) != 0:
                        planning_e_fea_tmp_out[self.a_e_type_dict["planning"][agent_type_index]] = self.edge_MLP["planning"][agent_type_index](planning_e_fea_tmp_in[self.a_e_type_dict["planning"][agent_type_index]])
                    if self.use_road_obs:
                        if len(self.a_e_type_dict["view"][agent_type_index]) != 0:
                            view_e_fea_tmp_out[self.a_e_type_dict["view"][agent_type_index]] = self.edge_MLP["view"][agent_type_index](view_e_fea_tmp_in[self.a_e_type_dict["view"][agent_type_index]])
                out_e_fea_lis.append(self_e_fea_tmp_out + input_dic["graph_lis"].edges["self"].data["a_e_hidden"])
                out_e_fea_lis.append(other_e_fea_tmp_out + input_dic["graph_lis"].edges["other"].data["a_e_hidden"])
                out_e_fea_lis.append(planning_e_fea_tmp_out + input_dic["graph_lis"].edges["planning"].data["a_e_hidden"])

                if self.use_map:
                    l2a_out_e_fea = torch.cat([input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype="l2a")[0]], input_dic["graph_lis"].edges["l2a"].data["l_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="l2a")[1], :, 0]], dim=-1)
                    l2a_out_e_fea = self.edge_MLP["l2a"](l2a_out_e_fea)
                    out_e_fea_lis.append(l2a_out_e_fea + input_dic["graph_lis"].edges["l2a"].data["l_e_hidden"])

                    g2a_out_e_fea = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="g2a")[1], :, 0], input_dic["graph_lis"].edges["g2a"].data["g_e_hidden"]], dim=-1)
                    g2a_out_e_fea = self.edge_MLP["g2a"](g2a_out_e_fea)
                    out_e_fea_lis.append(g2a_out_e_fea + input_dic["graph_lis"].edges["g2a"].data["g_e_hidden"])
                if self.use_road_obs:
                    out_e_fea_lis.append(view_e_fea_tmp_out + input_dic["graph_lis"].edges["view"].data["a_e_hidden"])
            else:
                self_e_fea_tmp_in = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
                other_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="other")[0]], input_dic["graph_lis"].edges["other"].data["a_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="other")[1]]], dim=-1)
                if self.use_road_obs:
                    view_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["road"][input_dic["graph_lis"].edges(etype="view")[0], :], input_dic["graph_lis"].edges["view"].data["a_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="view")[1], :]], dim=-1)
                    view_e_fea_tmp_out = torch.zeros((view_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)

                self_e_fea_tmp_out = torch.zeros_like(self_e_fea_tmp_in)
                other_e_fea_tmp_out = torch.zeros((other_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
                for agent_type_index in range(3):
                    if len(self.a_e_type_dict["self"][agent_type_index]) != 0:
                        self_e_fea_tmp_out[self.a_e_type_dict["self"][agent_type_index]] = self.edge_MLP["self"][agent_type_index](self_e_fea_tmp_in[self.a_e_type_dict["self"][agent_type_index]])
                    if len(self.a_e_type_dict["other"][agent_type_index]) != 0:  
                        other_e_fea_tmp_out[self.a_e_type_dict["other"][agent_type_index]] = self.edge_MLP["other"][agent_type_index](other_e_fea_tmp_in[self.a_e_type_dict["other"][agent_type_index]])
                out_e_fea_lis.append(self_e_fea_tmp_out + input_dic["graph_lis"].edges["self"].data["a_e_hidden"])
                out_e_fea_lis.append(other_e_fea_tmp_out + input_dic["graph_lis"].edges["other"].data["a_e_hidden"])

                if self.use_map:
                    l2a_out_e_fea = torch.cat([input_dic["graph_lis"].ndata["l_n_hidden"]["lane"][input_dic["graph_lis"].edges(etype="l2a")[0]], input_dic["graph_lis"].edges["l2a"].data["l_e_hidden"], input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="l2a")[1]]], dim=-1)
                    l2a_out_e_fea = self.edge_MLP["l2a"](l2a_out_e_fea)
                    out_e_fea_lis.append(l2a_out_e_fea + input_dic["graph_lis"].edges["l2a"].data["l_e_hidden"])

                    g2a_out_e_fea = torch.cat([input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][input_dic["graph_lis"].edges(etype="g2a")[1]], input_dic["graph_lis"].edges["g2a"].data["g_e_hidden"]], dim=-1)
                    g2a_out_e_fea = self.edge_MLP["g2a"](g2a_out_e_fea)
                    out_e_fea_lis.append(g2a_out_e_fea + input_dic["graph_lis"].edges["g2a"].data["g_e_hidden"])
                if self.use_road_obs:
                    out_e_fea_lis.append(view_e_fea_tmp_out + input_dic["graph_lis"].edges["view"].data["a_e_hidden"])
        return all_out_n_fea, out_e_fea_lis

    def message_func(self, edges, etype):
        if self.use_map:
            if etype == "l2a" or etype == "g2a":
                k, v = self.wkvs[etype](edges.data[self.etype2hidden_name[etype]]).view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
            elif etype == "other" or etype == "view":
                tmp_input = edges.data[self.etype2hidden_name[etype]]
                tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
                for agent_type_index in range(3):
                    if len(self.a_e_type_dict[etype][agent_type_index]) != 0:
                        tmp_output[self.a_e_type_dict[etype][agent_type_index]] = self.wkvs[etype][agent_type_index](tmp_input[self.a_e_type_dict[etype][agent_type_index]])
                k, v = tmp_output.view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
            else:
                tmp_input = edges.data[self.etype2hidden_name[etype]]
                tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
                for agent_type_index in range(3):
                    if len(self.a_e_type_dict["planning"][agent_type_index]) != 0:
                        tmp_output[self.a_e_type_dict["planning"][agent_type_index]] = self.wkvs[etype][agent_type_index](tmp_input[self.a_e_type_dict["planning"][agent_type_index]])
                k, v = tmp_output.view(-1, self.n_head, self.head_dim * 2).split(dim=-1, split_size=self.head_dim)
        else:
            if etype != "other" and etype != "planning":
                k, v = self.wkvs[etype](edges.data[self.etype2hidden_name[etype]]).view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
            else:
                tmp_input = edges.data[self.etype2hidden_name[etype]]
                tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
                for agent_type_index in range(3):
                    if len(self.a_e_type_dict["other"][agent_type_index]) != 0:
                        tmp_output[self.a_e_type_dict["other"][agent_type_index]] = self.wkvs[etype][agent_type_index](tmp_input[self.a_e_type_dict["other"][agent_type_index]])
                k, v = tmp_output.view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
        return {etype+"_k": k, etype+"_v": v}

    def reduce_func(self, nodes, etype):
        node_num, neighbor_num, n_head, hidden = nodes.mailbox[etype+"_k"].shape
        q = nodes.data[etype+"_q"].view(node_num*self.n_head, -1).unsqueeze(1)
        k = nodes.mailbox[etype+"_k"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        v = nodes.mailbox[etype+"_v"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        output, attn = self.attention(q, k, v, mask=None)
        return {"a_n_hidden_out":output.view(node_num, -1)}



class HDGT_encoder(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.use_road_obs = args.use_road_obs
        self.use_map = args.use_map
        self.use_planning = args.use_planning
        self.hidden_dim = args.hidden_dim
        self.shared_coor_encoder = MLP(d_in=3, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None) ## Encode x,y,z   
        self.shared_rel_encoder = MLP(d_in=5, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None) ## Encode Delta (x, y, z, cos(psi), sin(psi))
        
        self.agent_emb = Agent2embedding(input_dim, args)
        if self.use_road_obs:
            self.road_agent_emb = Road2embedding(input_dim, args)
        if self.use_planning:
            self.his_temporal_encoders = torch.nn.ModuleList([H_AgentTemporalEncoder(args) for _ in range(3)]) ## different temporal encoder for different agent type
            self.fut_temporal_encoders = torch.nn.ModuleList([F_AgentTemporalEncoder(args) for _ in range(3)])
            self.attention_temporal_encoder = torch.nn.ModuleList([AttentionTemporalEncoder(args) for _ in range(3)])
            self.pool_after_merge = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.temporal_encoders = torch.nn.ModuleList([H_AgentTemporalEncoder(args) for _ in range(3)])
            self.attention_temporal_encoder = torch.nn.ModuleList([AttentionTemporalEncoder(args) for _ in range(3)])
            ## Input: [Node Feature, Rel_Pos]
        self.agent_e_fea_MLPs = torch.nn.ModuleList([MLP(args.hidden_dim+args.hidden_dim//4, args.hidden_dim*4, args.hidden_dim, norm=BN1D) for _ in range(3)]) ## Inti Agent Edge Feature
        
        self.lane_emb = Lane2embedding(args)
        self.polygon_emb = Polygon2embedding(args)
        
        self.num_of_gnn_layer = args.num_of_gnn_layer
        self.lane_gnns = torch.nn.ModuleList([LaneHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])
        self.agent_gnns = torch.nn.ModuleList([AgentHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])

    def forward(self, input_dic):
        ## Init Agent Node
        if self.use_planning:
            agent_n_emb = self.agent_emb(input_dic, self.shared_coor_encoder)
            if self.use_road_obs:
                road_agent_n_emb = self.road_agent_emb(input_dic, self.shared_coor_encoder)
                road_n_type_indices = [torch.where((input_dic["graph_lis"].ndata["a_n_type"]["road"]) == _)[0] for _ in range(3)]
                road_n_fea = torch.zeros((road_agent_n_emb.shape[0], road_agent_n_emb.shape[1]), device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    if len(road_n_type_indices[_]) == 0:
                        continue
                    road_n_fea[road_n_type_indices[_]] = self.attention_temporal_encoder[_](road_agent_n_emb[road_n_type_indices[_]], input_dic["road_mask"][road_n_type_indices[_]])
            plan_index = input_dic["fut_agent_index"]
            plan_index = plan_index.long()
            plan_agent_n_emb = agent_n_emb[plan_index, :, 16:]
            agent_n_emb = agent_n_emb[..., :16]
            plan_agent_n_type = input_dic["graph_lis"].ndata["a_n_type"]["agent"][plan_index]
            plan_agent_type_dic = [torch.where(plan_agent_n_type == _) for _ in range(3)]
            agent_n_type_indices = [torch.where((input_dic["graph_lis"].ndata["a_n_type"]["agent"]) == _) for _ in range(3)]
            agent_n_his_fea = torch.zeros((agent_n_emb.shape[0], agent_n_emb.shape[1]), device="cuda:"+str(input_dic["gpu"]))
            agent_n_fut_fea = torch.zeros((plan_index.shape[0], agent_n_emb.shape[1]),device="cuda:" + str(input_dic["gpu"]))
            for _ in range(3):
                agent_n_his_fea[agent_n_type_indices[_]] = self.his_temporal_encoders[_](agent_n_emb[agent_n_type_indices[_]])
                agent_n_fut_fea[plan_agent_type_dic[_]] = self.fut_temporal_encoders[_](plan_agent_n_emb[plan_agent_type_dic[_]])
            agent_n_his_fea = agent_n_his_fea[..., np.newaxis]
            agent_n_fea = torch.cat([agent_n_his_fea, torch.zeros((agent_n_his_fea.shape[0], agent_n_his_fea.shape[1], 1), device="cuda:"+str(input_dic["gpu"]))], dim=-1)
            agent_n_fea[plan_index, :, 1] = agent_n_fut_fea
        else:
            agent_n_emb = self.agent_emb(input_dic, self.shared_coor_encoder)
            if self.use_road_obs:
                road_agent_n_emb = self.road_agent_emb(input_dic, self.shared_coor_encoder)
                road_n_type_indices = [torch.where((input_dic["graph_lis"].ndata["a_n_type"]["road"]) == _)[0] for _ in range(3)]
                road_n_fea = torch.zeros((road_agent_n_emb.shape[0], road_agent_n_emb.shape[1]), device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    if len(road_n_type_indices[_]) == 0:
                        continue
                    road_n_fea[road_n_type_indices[_]] = self.attention_temporal_encoder[_](road_agent_n_emb[road_n_type_indices[_]], input_dic["road_mask"][road_n_type_indices[_]])
            agent_n_type_indices = [torch.where((input_dic["graph_lis"].ndata["a_n_type"]["agent"]) == _) for _ in range(3)]
            agent_n_fea = torch.zeros((agent_n_emb.shape[0], agent_n_emb.shape[1]), device="cuda:" + str(input_dic["gpu"]))
            for _ in range(3):
                agent_n_fea[agent_n_type_indices[_]] = self.temporal_encoders[_](agent_n_emb[agent_n_type_indices[_]])


        ## Init Agent-Related Edge
        if self.use_map:
            if self.use_planning:
                #处理历史时域的边特征
                agent_his_e_type_lis = torch.cat([input_dic["graph_lis"].edata["a_e_type"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]], dim=0)
                agent_his_e_fea_rel_pos = torch.cat([input_dic["graph_lis"].edata["a_e_fea"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]], dim=0)
                agent_his_e_type_indices = [torch.where(agent_his_e_type_lis == _) for _ in range(3)]
                agent_his_e_srcn_fea = torch.cat([agent_n_fea[input_dic["graph_lis"].edges(etype=_)[0], :, 0] for _ in ["self", "other", "a2l"]], dim=0)
                agent_his_e_fea_relpos = self.shared_rel_encoder(agent_his_e_fea_rel_pos)  
                agent_his_e_fea = torch.zeros((agent_his_e_fea_relpos.shape[0], self.hidden_dim),device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    agent_his_e_fea[agent_his_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_his_e_fea_relpos[agent_his_e_type_indices[_]], agent_his_e_srcn_fea[agent_his_e_type_indices[_]]], dim=-1))
                #处理未来时域的边特征
                agent_plan_e_type_lis = input_dic["graph_lis"].edata["a_e_type"][('agent', 'planning', 'agent')]
                agent_plan_e_fea_rel_pos = input_dic["graph_lis"].edata["a_e_fea"][('agent', 'planning', 'agent')]
                agent_plan_e_type_indices = [torch.where(agent_plan_e_type_lis == _) for _ in range(3)]
                agent_plan_e_srcn_fea =agent_n_fea[input_dic["graph_lis"].edges(etype="planning")[0], :, 1]
                agent_plan_e_fea_relpos = self.shared_rel_encoder(agent_plan_e_fea_rel_pos) 
                agent_plan_e_fea = torch.zeros((agent_plan_e_fea_relpos.shape[0], self.hidden_dim), device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    agent_plan_e_fea[agent_plan_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_plan_e_fea_relpos[agent_plan_e_type_indices[_]], agent_plan_e_srcn_fea[agent_plan_e_type_indices[_]]], dim=-1))
                input_dic["graph_lis"].ndata["a_n_hidden"] = {"agent": agent_n_fea}
                agent_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["a_e_type"][_]) for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]])
                for _index, _ in enumerate([('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]):
                    input_dic["graph_lis"].edata["a_e_hidden"] = {_: agent_his_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index + 1]]}  # default
                input_dic["graph_lis"].edata["a_e_hidden"] = {('agent', 'planning', 'agent'): agent_plan_e_fea}
                if self.use_road_obs:
                    road_view_e_type_lis = input_dic["graph_lis"].edata["a_e_type"][("road", "view", "agent")]
                    road_view_e_fea_rel_pos = input_dic["graph_lis"].edata["a_e_fea"][("road", "view", "agent")]
                    road_view_e_type_indices = [torch.where(road_view_e_type_lis == _)[0] for _ in range(3)]
                    road_view_e_srcn_fea = road_n_fea[input_dic["graph_lis"].edges(etype="view")[0], ...]
                    road_view_e_fea_relpos = self.shared_rel_encoder(road_view_e_fea_rel_pos)  
                    road_view_e_fea = torch.zeros((road_view_e_fea_relpos.shape[0], self.hidden_dim), device="cuda:" + str(input_dic["gpu"]))
                    for _ in range(3):
                        road_view_e_fea[road_view_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([road_view_e_fea_relpos[road_view_e_type_indices[_]],
                             road_view_e_srcn_fea[road_view_e_type_indices[_]]], dim=-1))
                    input_dic["graph_lis"].ndata["a_n_hidden"] = {"road": road_n_fea}
                    input_dic["graph_lis"].edata["a_e_hidden"] = {("road", "view", "agent"): road_view_e_fea}
            else:
                agent_e_type_lis = torch.cat([input_dic["graph_lis"].edata["a_e_type"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]], dim=0)
                agent_e_fea_rel_pos = torch.cat([input_dic["graph_lis"].edata["a_e_fea"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]], dim=0)
                agent_e_type_indices = [torch.where(agent_e_type_lis == _) for _ in range(3)]
                agent_e_src_n_fea = torch.cat([agent_n_fea[input_dic["graph_lis"].edges(etype=_)[0],...]  for _ in ["self", "other", "a2l"]], dim=0)    #各edge起点agent的特征
                agent_e_fea_rel_pos = self.shared_rel_encoder(agent_e_fea_rel_pos)     
                agent_e_fea = torch.zeros((agent_e_fea_rel_pos.shape[0], self.hidden_dim), device="cuda:"+str(input_dic["gpu"]))
                for _ in range(3):
                    agent_e_fea[agent_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_e_fea_rel_pos[agent_e_type_indices[_]], agent_e_src_n_fea[agent_e_type_indices[_]]], dim=-1))
                input_dic["graph_lis"].ndata["a_n_hidden"] = {"agent": agent_n_fea}
                agent_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["a_e_type"][_]) for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]])
                for _index, _ in enumerate([('agent', 'self', 'agent'), ('agent', 'other', 'agent'), ('agent', 'a2l', 'lane')]):
                    input_dic["graph_lis"].edata["a_e_hidden"] = {_:agent_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index+1]]}  #default
                if self.use_road_obs:
                    road_view_e_type_lis = input_dic["graph_lis"].edata["a_e_type"][("road", "view", "agent")]
                    road_view_e_fea_rel_pos = input_dic["graph_lis"].edata["a_e_fea"][("road", "view", "agent")]
                    road_view_e_type_indices = [torch.where(road_view_e_type_lis == _)[0] for _ in range(3)]
                    road_view_e_srcn_fea = road_n_fea[input_dic["graph_lis"].edges(etype="view")[0], ...]
                    road_view_e_fea_relpos = self.shared_rel_encoder(road_view_e_fea_rel_pos) 
                    road_view_e_fea = torch.zeros((road_view_e_fea_relpos.shape[0], self.hidden_dim), device="cuda:" + str(input_dic["gpu"]))
                    for _ in range(3):
                        road_view_e_fea[road_view_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([road_view_e_fea_relpos[road_view_e_type_indices[_]],
                             road_view_e_srcn_fea[road_view_e_type_indices[_]]], dim=-1))
                    input_dic["graph_lis"].ndata["a_n_hidden"] = {"road": road_n_fea}
                    input_dic["graph_lis"].edata["a_e_hidden"] = {("road", "view", "agent"): road_view_e_fea}

            self.lane_emb(input_dic, self.shared_coor_encoder, self.shared_rel_encoder)
            self.polygon_emb(input_dic, self.shared_coor_encoder)
        else:
            if self.use_planning:
                #处理历史时域的边特征
                agent_his_e_type_lis = torch.cat([input_dic["graph_lis"].edata["a_e_type"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]], dim=0)
                agent_his_e_fea_rel_pos = torch.cat([input_dic["graph_lis"].edata["a_e_fea"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]], dim=0)
                agent_his_e_type_indices = [torch.where(agent_his_e_type_lis == _) for _ in range(3)]
                agent_his_e_srcn_fea = torch.cat([agent_n_fea[input_dic["graph_lis"].edges(etype=_)[0], :, 0] for _ in ["self", "other"]], dim=0)
                agent_his_e_fea_relpos = self.shared_rel_encoder(agent_his_e_fea_rel_pos)  
                agent_his_e_fea = torch.zeros((agent_his_e_fea_relpos.shape[0], self.hidden_dim),device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    agent_his_e_fea[agent_his_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_his_e_fea_relpos[agent_his_e_type_indices[_]], agent_his_e_srcn_fea[agent_his_e_type_indices[_]]], dim=-1))
                #处理未来时域的边特征
                agent_plan_e_type_lis = input_dic["graph_lis"].edata["a_e_type"][('agent', 'planning', 'agent')]
                agent_plan_e_fea_rel_pos = input_dic["graph_lis"].edata["a_e_fea"][('agent', 'planning', 'agent')]
                agent_plan_e_type_indices = [torch.where(agent_plan_e_type_lis == _) for _ in range(3)]
                agent_plan_e_srcn_fea =agent_n_fea[input_dic["graph_lis"].edges(etype="planning")[0], :, 1]
                agent_plan_e_fea_relpos = self.shared_rel_encoder(agent_plan_e_fea_rel_pos)  
                agent_plan_e_fea = torch.zeros((agent_plan_e_fea_relpos.shape[0], self.hidden_dim), device="cuda:" + str(input_dic["gpu"]))
                for _ in range(3):
                    agent_plan_e_fea[agent_plan_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_plan_e_fea_relpos[agent_plan_e_type_indices[_]], agent_plan_e_srcn_fea[agent_plan_e_type_indices[_]]], dim=-1))
                input_dic["graph_lis"].ndata["a_n_hidden"] = {"agent": agent_n_fea}
                agent_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["a_e_type"][_]) for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]])
                for _index, _ in enumerate([('agent', 'self', 'agent'), ('agent', 'other', 'agent')]):
                    input_dic["graph_lis"].edata["a_e_hidden"] = {_: agent_his_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index + 1]]}  # default
                input_dic["graph_lis"].edata["a_e_hidden"] = {('agent', 'planning', 'agent'): agent_plan_e_fea}
            else:
                agent_e_type_lis = torch.cat([input_dic["graph_lis"].edata["a_e_type"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]], dim=0)
                agent_e_fea_rel_pos = torch.cat([input_dic["graph_lis"].edata["a_e_fea"][_] for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]], dim=0)
                agent_e_type_indices = [torch.where(agent_e_type_lis == _) for _ in range(3)]
                agent_e_src_n_fea = torch.cat([agent_n_fea[input_dic["graph_lis"].edges(etype=_)[0],...]  for _ in ["self", "other", "a2l"]], dim=0)    #各edge起点agent的特征
                agent_e_fea_rel_pos = self.shared_rel_encoder(agent_e_fea_rel_pos)       
                agent_e_fea = torch.zeros((agent_e_fea_rel_pos.shape[0], self.hidden_dim), device="cuda:"+str(input_dic["gpu"]))
                for _ in range(3):
                    agent_e_fea[agent_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_e_fea_rel_pos[agent_e_type_indices[_]], agent_e_src_n_fea[agent_e_type_indices[_]]], dim=-1))
                input_dic["graph_lis"].ndata["a_n_hidden"] = {"agent": agent_n_fea}
                agent_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["a_e_type"][_]) for _ in [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]])
                for _index, _ in enumerate([('agent', 'self', 'agent'), ('agent', 'other', 'agent')]):
                    input_dic["graph_lis"].edata["a_e_hidden"] = {_:agent_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index+1]]}  #default
        
        for i in range(self.num_of_gnn_layer):
            if self.use_map:
                output_lane_n_fea, output_in_lane_e_fea = self.lane_gnns[i](input_dic)
            output_agent_n_fea, output_in_agent_e_fea = self.agent_gnns[i](input_dic)
            if self.use_map:
                input_dic["graph_lis"].nodes["lane"].data["l_n_hidden"] = output_lane_n_fea
                for _index, _ in enumerate(["left", "right", "prev", "follow"]):
                    input_dic["graph_lis"].edges[_].data["l_e_hidden"] = output_in_lane_e_fea[_index]
                input_dic["graph_lis"].edges["a2l"].data["a_e_hidden"] = output_in_lane_e_fea[-1]
            
            input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"] = output_agent_n_fea
            if self.use_map:
                if self.use_planning:
                    for _index, _ in enumerate([("self", "a_e_hidden"), ("other", "a_e_hideen"), ("planning", "a_e_hideen"), ("l2a", "l_e_hidden"), ("g2a", "g_e_hidden")]):
                        input_dic["graph_lis"].edges[_[0]].data[_[1]] = output_in_agent_e_fea[_index]
                    if len(output_in_agent_e_fea) == 6:
                        input_dic["graph_lis"].edges["view"].data["a_e_hidden"] = output_in_agent_e_fea[-1]
                else:
                    for _index, _ in enumerate([("self", "a_e_hidden"), ("other", "a_e_hideen"), ("l2a", "l_e_hidden"), ("g2a", "g_e_hidden")]):
                        input_dic["graph_lis"].edges[_[0]].data[_[1]] = output_in_agent_e_fea[_index]
                    if len(output_in_agent_e_fea) == 5:
                        input_dic["graph_lis"].edges["view"].data["a_e_hidden"] = output_in_agent_e_fea[-1]
            else:
                if self.use_planning:
                    for _index, _ in enumerate([("self", "a_e_hidden"), ("other", "a_e_hideen"), ("planning", "a_e_hideen")]):
                        input_dic["graph_lis"].edges[_[0]].data[_[1]] = output_in_agent_e_fea[_index]
                    if len(output_in_agent_e_fea) == 4:
                        input_dic["graph_lis"].edges["view"].data["a_e_hidden"] = output_in_agent_e_fea[-1]
                else:
                    for _index, _ in enumerate([("self", "a_e_hidden"), ("other", "a_e_hideen")]):
                        input_dic["graph_lis"].edges[_[0]].data[_[1]] = output_in_agent_e_fea[_index]
                    if len(output_in_agent_e_fea) == 3:
                        input_dic["graph_lis"].edges["view"].data["a_e_hidden"] = output_in_agent_e_fea[-1]
        return input_dic["graph_lis"]



class HDGT_model(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.use_planning = args.use_planning
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.num_prediction = args.num_prediction
        
        self.encoder = HDGT_encoder(input_dim, args)
        self.decoder = torch.nn.ModuleList([RefineDecoder(args) for _ in range(3)])
        
    def forward(self, input_dic):
        output_het_graph = self.encoder(input_dic)
    
        neighbor_size_lis = input_dic["neighbor_size_lis"]        #每个场景中的agent数量    
        if self.use_planning:
            all_agent_raw_traj = input_dic["graph_lis"].nodes["agent"].data["a_n_fea"][:, :16, :2].clone()
            agent_node_fea = output_het_graph.nodes["agent"].data["a_n_hidden"][..., 0]
        else:
            all_agent_raw_traj = input_dic["graph_lis"].nodes["agent"].data["a_n_fea"][..., :2].clone()
            agent_node_fea = output_het_graph.nodes["agent"].data["a_n_hidden"]
        cumsum_neighbor_size_lis = np.cumsum(neighbor_size_lis, axis=0).tolist()
        cumsum_neighbor_size_lis = [0] + cumsum_neighbor_size_lis
        # pred_num_lis = input_dic["pred_num_lis"]
        tar_id_lis = input_dic["tar_id_lis"]
        object_id_lis = input_dic["object_id_lis"]
        tar_index = [np.where(object_id_lis[i] == tar_id_lis[i])[0] for i in range(len(tar_id_lis))]
        all_agent_id = input_dic["graph_lis"].nodes("agent")
        agent_n_type_lis = output_het_graph.ndata["a_n_type"]["agent"]
        
        ## Obtain the node feature of target agents 
        targat_agent_indice_lis = []
        target_agent_indice_bool_type_lis = [[] for _ in range(3)]
        targat_agent_fea = [[] for _ in range(3)]
        target_agent_id = [[] for _ in range(3)]
        for i in range(1, len(cumsum_neighbor_size_lis)):
            now_agent_type_lis = agent_n_type_lis[cumsum_neighbor_size_lis[i-1]:cumsum_neighbor_size_lis[i]]
            now_agent_id = all_agent_id[cumsum_neighbor_size_lis[i-1]+tar_index[i-1]]
            now_agent_node_fea = agent_node_fea[cumsum_neighbor_size_lis[i-1]+tar_index[i-1]]
            now_target_agent_indice_bool_type_lis = [now_agent_type_lis[tar_index[i-1]]==_ for _ in range(3)]
            
            targat_agent_indice_lis += [cumsum_neighbor_size_lis[i-1]+tar_index[i-1]]
            for _ in range(3):
                target_agent_indice_bool_type_lis[_].append(now_target_agent_indice_bool_type_lis[_])
                targat_agent_fea[_].append(now_agent_node_fea[now_target_agent_indice_bool_type_lis[_]])
                target_agent_id[_].append(now_agent_id[now_target_agent_indice_bool_type_lis[_]])

        target_agent_indice_bool_type_lis = [torch.cat(target_agent_indice_bool_type_lis[_], dim=0) for _ in range(3)]
        targat_agent_fea = [torch.cat(targat_agent_fea[_], dim=0) for _ in range(3)]
        target_agent_id = [torch.cat(target_agent_id[_], dim=0) for _ in range(3)]

        prediction = []
        for _ in range(3):
            if targat_agent_fea[_].shape[0] == 0:
                prediction.append((torch.zeros((0, 1)), torch.zeros((0, 1))))
            else:
                prediction.append(self.decoder[_](targat_agent_fea[_], target_agent_id[_], all_agent_raw_traj, input_dic))
        agent_cls_res = [prediction[_][0] for _ in range(3)]
        agent_reg_res = [prediction[_][1] for _ in range(3)]
        return agent_reg_res, agent_cls_res, target_agent_indice_bool_type_lis



class RefineCNN(nn.Module):
    def __init__(self, in_c, dilation=1, args=None):
        super(RefineCNN, self).__init__()
        self.in_c = in_c
        self.conv1 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)     #dilation指卷积核元素之间的间距
        self.gn1 = nn.GroupNorm(num_groups=in_c, num_channels=in_c)              #GroupNorm不会改变输入张量的shape，只是按照group的方式完成normalization
        self.conv2 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)
        self.gn2 = nn.GroupNorm(num_groups=in_c, num_channels=in_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(in_c)
    def forward(self, x):
        identity = x
        out = x        
        out = self.act(self.gn1(self.conv1(out)))
        out = self.act(self.gn2(self.conv2(out)))
        out = self.se(out) + identity
        out = self.act(out)
        return out



class RefineContextLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        self.use_road_obs = args.use_road_obs
        self.num_prediction = args.num_prediction
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = int(args.hidden_dim)
        self.n_head = self.d_model // self.head_dim
        d_k = self.head_dim
        d_v = self.head_dim
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)
        self.wq = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(args.hidden_dim, self.n_head * d_k, bias=False)
        )
        
        
        self.wkvs = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(args.hidden_dim, self.n_head * d_k * 2, bias=False),
            )
            for _ in range(3)
        ])
        if args.use_planning:
            self.wkvs.append(nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(args.hidden_dim, self.n_head * d_k * 2, bias=False),
            ))
        if args.use_road_obs:
            self.wkvs.append(nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(args.hidden_dim, self.n_head * d_k * 2, bias=False),
            ))
        self.attn_fc = nn.Sequential(
            nn.Linear(self.n_head * d_v, self.d_model, bias=True),
            nn.Dropout(args.dropout),
        )
        
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_model*4, args.dropout, args)

    def forward(self, raw_q, raw_kv_lis, raw_kv_indices, input_dic):
        all_q_lis = self.wq(raw_q).view(raw_q.shape[0], self.num_prediction*self.n_head, self.head_dim)
        all_kv_lis = [self.wkvs[_](torch.cat(raw_kv_lis[_], dim=0)) for _ in range(3)]
        if self.use_planning:
            all_kv_lis.append(self.wkvs[3](torch.cat(raw_kv_lis[3], dim=0)))
        if self.use_road_obs:
            all_kv_lis.append(self.wkvs[-1](torch.cat(raw_kv_lis[-1], dim=0)))
        all_kv_lis = [[_[raw_kv_indices[_index][indices_i-1]:raw_kv_indices[_index][indices_i]] for indices_i in range(1, len(raw_kv_indices[_index]))] for _index, _ in enumerate(all_kv_lis)]
        if self.use_planning != True and self.use_road_obs != True:
            all_kv_lis = [torch.cat([all_kv_lis[0][_], all_kv_lis[1][_], all_kv_lis[2][_]], dim=0).view(-1, self.n_head, self.head_dim*2).unsqueeze(1).repeat(1, self.num_prediction, 1, 1).view(-1, self.num_prediction*self.n_head, self.head_dim*2).transpose(0, 1) for _ in range(raw_q.shape[0])]     
        elif self.use_planning and self.use_road_obs:
            all_kv_lis = [torch.cat([all_kv_lis[0][_], all_kv_lis[1][_], all_kv_lis[2][_], all_kv_lis[3][_], all_kv_lis[4][_]], dim=0).view(-1, self.n_head, self.head_dim * 2).unsqueeze(1).repeat(1, self.num_prediction, 1, 1).view(-1, self.num_prediction * self.n_head, self.head_dim * 2).transpose(0, 1) for _ in range(raw_q.shape[0])]
        else:
            all_kv_lis = [
                torch.cat([all_kv_lis[0][_], all_kv_lis[1][_], all_kv_lis[2][_], all_kv_lis[3][_]], dim=0).view(-1, self.n_head, self.head_dim * 2).unsqueeze(1).repeat(1, self.num_prediction, 1, 1).view(-1, self.num_prediction * self.n_head, self.head_dim * 2).transpose(0, 1) for _ in range(raw_q.shape[0])]
        all_out_q_lis = self.attn_fc(torch.cat([self.attention(q=all_q_lis[_].unsqueeze(1), k=all_kv_lis[_][..., :self.head_dim], v=all_kv_lis[_][..., self.head_dim:])[0].view(-1, self.num_prediction, self.n_head*self.head_dim) if all_kv_lis[_].shape[0]!=0 else torch.zeros_like(raw_q[0:1, :, :]) for _ in range(raw_q.shape[0])], dim=0)) + raw_q      #12.13：看到这里啦！
        return self.ffn(all_out_q_lis)


class RefineContextModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        self.use_road_obs = args.use_road_obs
        self.num_prediction = args.num_prediction
        self.hidden_dim = args.hidden_dim
        self.modal_emb = nn.parameter.Parameter(torch.zeros(args.num_prediction, args.hidden_dim))    #将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter
        torch.nn.init.normal(self.modal_emb)
        self.init_q = MLP(args.hidden_dim*2, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, prenorm=True)
        self.etype2hidden_name = {"other":"a_e_hidden", "l2a":"l_e_hidden", "g2a":"g_e_hidden"}
        if self.use_planning:
            self.etype2hidden_name.update({"planning": "a_e_hidden"})
        if self.use_road_obs:
            self.etype2hidden_name.update({"view": "a_e_hidden"})
        self.refine_context_layer = nn.ModuleList([RefineContextLayer(args) for _ in range(2)])

    def forward(self, agent_ids, input_dic):
        with input_dic["graph_lis"].local_scope():
            self.device = "cuda:"+str(input_dic["gpu"])
            if self.use_planning:
                raw_agent_fea = input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][agent_ids, :, 0] 
            else:
                raw_agent_fea = input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][agent_ids]
            num_agent = raw_agent_fea.shape[0]

            raw_q = torch.cat([raw_agent_fea.unsqueeze(1).repeat(1, self.num_prediction, 1), self.modal_emb.unsqueeze(0).repeat(num_agent, 1, 1)], dim=-1)
            raw_q = self.init_q(raw_q) + self.modal_emb
            
            raw_kv_lis = [[torch.zeros((0, self.hidden_dim), device=self.device)]*num_agent for _ in range(3)]
            for agent_index, agent_id in enumerate(agent_ids):
                for etype_index, etype in enumerate(["other", "l2a", "g2a"]):
                    now_type_eid_lis = input_dic["graph_lis"].in_edges(etype=etype, v=agent_id, form="eid")   #找到指定代理agent_id的边类型为etype的入边in_edges
                    if len(now_type_eid_lis) > 0:
                        raw_kv_lis[etype_index][agent_index] = input_dic["graph_lis"].edges[etype].data[self.etype2hidden_name[etype]][now_type_eid_lis]
            if self.use_planning:
                raw_kv_lis.append([torch.zeros((0, self.hidden_dim), device=self.device)]*num_agent)
                for agent_index, agent_id in enumerate(agent_ids):
                    now_type_eid_lis = input_dic["graph_lis"].in_edges(etype="planning", v=agent_id, form="eid")  # 找到指定代理agent_id的边类型为etype的入边in_edges
                    if len(now_type_eid_lis) > 0:
                        raw_kv_lis[3][agent_index] = input_dic["graph_lis"].edges["planning"].data[self.etype2hidden_name["planning"]][now_type_eid_lis]
            if self.use_road_obs:
                raw_kv_lis.append([torch.zeros((0, self.hidden_dim), device=self.device)]*num_agent)
                for agent_index, agent_id in enumerate(agent_ids):
                    now_type_eid_lis = input_dic["graph_lis"].in_edges(etype="view", v=agent_id, form="eid")  # 找到指定代理agent_id的边类型为etype的入边in_edges
                    if len(now_type_eid_lis) > 0:
                        raw_kv_lis[-1][agent_index] = input_dic["graph_lis"].edges["view"].data[self.etype2hidden_name["view"]][now_type_eid_lis]
            
            raw_kv_length = [[int(__.shape[0]) for __ in _] for _ in raw_kv_lis]     #对于三种不同的边，各预测目标的入边数量
            raw_kv_indices = [np.cumsum([0]+_) for _ in raw_kv_length]
            for _ in range(len(self.refine_context_layer)):
                raw_q = self.refine_context_layer[_](raw_q, raw_kv_lis, raw_kv_indices, input_dic) + self.modal_emb
            return raw_q


class RefineLayer(nn.Module):
    def __init__(self, d_in, d_hid, args):
        super().__init__()
        self.in_linear = nn.Linear(d_in, d_hid)
        self.context_mlp = MLP(args.hidden_dim,  args.hidden_dim//2,  d_hid, norm=nn.LayerNorm, prenorm=True)
        self.fuse_linear = nn.Linear(d_hid * 2, d_hid)
        self.cnns = nn.Sequential(
        RefineCNN(d_hid, dilation=1, args=args),
        RefineCNN(d_hid, dilation=2, args=args),
        RefineCNN(d_hid, dilation=5, args=args),
        RefineCNN(d_hid, dilation=1, args=args),
        )
        self.out_linear = MLP(d_hid, d_hid, 2, norm=None)
    def forward(self, x, context):
        x = x.view(x.shape[0], x.shape[1], 41, -1)
        output = self.in_linear(x)
        context = self.context_mlp(context).unsqueeze(-2).repeat(1, 1, 41, 1)
        output = self.fuse_linear(torch.cat([context, output], dim=-1))
        bs, num_mode, t_len, hid = output.shape
        output = output.view(bs*num_mode, t_len, hid).transpose(1, 2)
        output = self.cnns(output).transpose(1, 2).view(bs, num_mode, t_len, hid)[:, :, 16:, :]
        output = self.out_linear(output)
        return output

class AttentionModeEncoder(nn.Module):
    def __init__(self, args):
        super(AttentionModeEncoder, self).__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.hidden_dim = args.hidden_dim
        self.query_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value_linear = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.out_linear = nn.Linear(args.hidden_dim, args.hidden_dim)

    def forward(self, x, mask=None):
        # x: [seq_len, batch_size, embed_dim]
        Q = self.query_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        K = self.key_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        V = self.value_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # mask 的形状应该是 [batch_size, 1, 1, seq_len]，广播到 attention_scores
            mask = mask.unsqueeze(1).unsqueeze(2)  # 扩展维度以便与注意力分数相加
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        # 使用 softmax 对注意力分数进行归一化
        attention_scores = torch.clamp(attention_scores, min=-50, max=50)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        output = self.out_linear(attention_output)  # [batch_size, seq_len, embed_dim
        return output

class RefineDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()        
        self.hidden_dim = args.hidden_dim
        self.num_prediction = int(args.num_prediction)
        self.num_recurrent_layer = args.num_recurrent_layer
        
        self.refine_num = int(args.refine_num)
        self.refine_layer_lis = nn.ModuleList(
            [RefineLayer(2, self.hidden_dim//4, args) for _ in range(self.refine_num)]
        )
        self.is_output_vel = (args.output_vel == "True")
        self.is_cumsum_vel = (args.cumsum_vel == "True")

        self.refine_context_attn = RefineContextModule(args)
        self.m2m_propose_attn_layer = AttentionModeEncoder(args)
        
        self.reg_mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim*2, args.hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim*4, 25*2),
        )
        
        self.cls_mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim//2, 1),
        )
        
        
    def forward(self, target_agent_fea, agent_ids, agent_raw_traj, input_dic):
        refine_context = self.refine_context_attn(agent_ids, input_dic)/5.0
        for i in range(self.num_recurrent_layer):
            refine_context = self.m2m_propose_attn_layer(refine_context)
        reg_res = self.reg_mlp(refine_context).view(target_agent_fea.shape[0], self.num_prediction, 25, 2)
        cls_res = self.cls_mlp(refine_context).view(target_agent_fea.shape[0], self.num_prediction)
        if self.is_output_vel and self.is_cumsum_vel:
            reg_res = torch.cumsum(reg_res, dim=-2)
        reg_res_lis = [reg_res]
        if self.refine_num > 0:
            now_agent_raw_traj = agent_raw_traj[agent_ids].unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
            for _ in range(self.refine_num):
                now_input = torch.cat([now_agent_raw_traj, reg_res.detach()], dim=-2).view(reg_res.shape[0], self.num_prediction, -1) ## Full Traj
                reg_res = reg_res + self.refine_layer_lis[_](now_input, refine_context).view(reg_res.shape[0], self.num_prediction, 25, 2)
                reg_res_lis.append(reg_res)
        return cls_res, torch.stack(reg_res_lis, dim=1)



from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0#max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

import warnings
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)      #求标准正态分布函数x点左侧积分
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)              

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def weights_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1:               #find()检测是否包含字符串‘...‘，如果不包含则返回-1
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)            #用0来填充m.bias
            nn.init.xavier_normal_(m.weight)
        elif classname.find('Embedding') != -1:
            trunc_normal_(m.weight, mean=0, std=0.02)      #截断正态分布中的值填充m.weight，mean和std分别为正态分布的均值和方差
