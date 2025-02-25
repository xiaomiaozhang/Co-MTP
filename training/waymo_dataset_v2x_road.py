import os
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import random
import numpy as np
import pandas as pd
from functools import partial
import time
import re
import multiprocessing



class V2XDataset(Dataset):
    def __init__(self, agent_num_lis, data_file, num_current_frame, t_h, d_s, road_prediction):
        self.data_file = data_file
        # self.data_loader = File_DataLoader(data_file)
        self.num_current_frame = num_current_frame
        self.t_h = t_h
        self.d_s = d_s
        self.road_prediction = road_prediction
        self.k = 0
        self.agent_num_lis = agent_num_lis
    def __getitem__(self, idx):
        # index = idx[0]
        # if idx[1] > self.k:
        #     self.data_loader.loaded_data = None
        #     self.k += 1
        # file = self.data_loader.get_data(index // self.num_current_frame if index % self.num_current_frame != 0 else ((index // self.num_current_frame) if index != 0 else 0))
        # current_frame = index - (index // self.num_current_frame) * self.num_current_frame + self.t_h if index % self.num_current_frame != 0 else self.t_h
        # sample = {}
        # sample['fname'] = [file['fname'], current_frame]
        # sample['agent_feature'] = file['agent_feature'][current_frame][:, ::self.d_s, :]  # todo:下采样取消注释
        # sample['label'] = file['label'][current_frame][:, ::self.d_s, :]
        # sample['auxiliary_label'] = file['auxiliary_label'][current_frame][:, ::self.d_s, :]
        # sample['label_mask'] = file['label_mask'][current_frame][:, ::self.d_s]
        # sample['pred_num'] = file['pred_num'][current_frame]
        #     #        print(len(file['other_label'][current_frame]), file['other_label'][current_frame])
        #     #         sample['other_label'] = file['other_label'][current_frame][:, ::self.d_s, :] if len(file['other_label'][current_frame]) != 0 else []
        #     #         sample['other_label_mask'] = file['other_label_mask'][current_frame][:, ::self.d_s] if len(file['other_label'][current_frame]) != 0 else []
        # sample['object_id_lis'] = file['object_id_lis'][current_frame]
        # sample['agent_type'] = file['agent_type'][current_frame]
        # if not self.road_prediction:
        #     sample['AV_fut'] = np.flip(file["AV_fut"][current_frame][:, ::self.d_s, :], axis=1).copy()
        # sample['map_fea'] = [file['map_fea'][0][current_frame], file['map_fea'][1][current_frame]]
        return idx
    def __len__(self):
        return sum(map(len, self.agent_num_lis))

class File_DataLoader:
    def __init__(self, data_file):
        self.data_file = data_file
        self.current_file_index = 0
        self.loaded_data = None
        # self.lock = multiprocessing.Lock()


    def load_next(self):
        with open(self.data_file[self.current_file_index], "rb") as f:
            self.loaded_data = pickle.load(f)
        self.current_file_index = (self.current_file_index + 1) % len(self.data_file)

    def get_data(self, index):
        if self.loaded_data is None:
            # with self.lock:
            #     if self.loaded_data is None:
            self.load_next()
        return self.loaded_data[index]

## To make sure each batch has approximately the same number of node
class BalancedBatchSampler(BatchSampler):
    def __init__(self, args, agent_num_lis, data_file, seed_num, gpu, gpu_cnt, is_train):
        self.batch_size = args.batch_size
        self.agent_num_lis = agent_num_lis
        sorted_index_lis = [agent_num.argsort()[::-1].tolist() for agent_num in self.agent_num_lis]    #根据 input_size_lis 中元素从大到小排列的索引列表
        self.index_lis = [[] for _ in range(len(self.agent_num_lis))]
        self.sample_per_gpu_lis = []
        self.is_train = is_train
        for file_index, sorted_index in enumerate(sorted_index_lis):
            for i in range(self.batch_size):
                self.index_lis[file_index].append(sorted_index[int(len(sorted_index)//self.batch_size * i):int(len(sorted_index)//self.batch_size * (i+1))])
            if len(sorted_index)//self.batch_size * self.batch_size < len(sorted_index):
                self.index_lis[file_index][-1] = self.index_lis[file_index][-1] + sorted_index[len(sorted_index)//self.batch_size * self.batch_size:]    #将剩余索引添加到index_lis的最后一个子列表中
        self.seed_num = seed_num
        self.gpu = gpu
        self.data_file = data_file
        # self.data_loader = File_DataLoader(data_file)
        self.num_current_frame = args.num_current_frame
        self.t_h = args.t_h
        self.d_s = args.d_s
        self.road_prediction = args.road_prediction
        self.use_other_fut = args.use_other_fut
        self.use_road_obs = args.use_road_obs
        self.road_obs_data_path = args.road_obs_data_path
        self.m = 0

    def __iter__(self):
        for k in range(len(self.data_file)):
            with open(self.data_file[k], "rb") as f:
                data = pickle.load(f)
                self.m += 1
            if self.is_train:
                for i in range(len(self.index_lis[k])):
                    random.Random(self.seed_num+i).shuffle(self.index_lis[k][i])
            self.seed_num += 1
            # for i in range(int(self.gpu * self.sample_per_gpu_lis[k]), int((self.gpu + 1) * self.sample_per_gpu_lis[k])):
            for i in range(len(self.index_lis[k][0])):
                sample_lis = []
                for j in range(self.batch_size):
                    index = self.index_lis[k][j][i]
                    file = data[index // self.num_current_frame if index % self.num_current_frame != 0 else ((index // self.num_current_frame) if index != 0 else 0)]
                    current_frame = index - (index // self.num_current_frame) * self.num_current_frame + self.t_h if index % self.num_current_frame != 0 else self.t_h
                    sample = {}
                    sample['fname'] = [file['fname'], current_frame]
                    if self.is_train:
                        road_data = os.path.join(self.road_obs_data_path, "train", "data")
                    else:
                        road_data = os.path.join("/data/lixc/hdgt/visual_raw_data/cooperative-vehicle-infrastructure/process_newv2x_rock1/", "val", "data")
                    road_file = str(file['fname'].split('.')[0]) + ".pkl"
                    if road_file in os.listdir(road_data):
                        with open(os.path.join(road_data, road_file), "rb") as f:
                            road_file = pickle.load(f)
                    else:
                        continue
                    if road_file["tar_id"][current_frame] in file["object_id_lis"][current_frame]:
                        sample['tar_id'] = road_file['tar_id'][current_frame]
                    else:
                        continue
                    if self.use_road_obs:
                        delay_frame = current_frame - 0     #todo:延迟
                        if delay_frame < 30:
                            continue
                        sample['road_feature'] = road_file['road_feature'][delay_frame][:, ::self.d_s, :]
                        sample['road_feature'][:, :, 1] = sample['road_feature'][:, :, 1] + 0.2  #todo：噪音
                        sample['road_feature'][:, :, 6] = sample['road_feature'][:, :, 6] + 0.2
                        sample['road_feature_mask'] = road_file['road_feature_mask'][delay_frame][:, ::self.d_s]
                        sample['road_ids'] = road_file['road_ids'][delay_frame]
                        sample['road_type'] = road_file['road_type'][delay_frame]
                    sample['agent_feature'] = file['agent_feature'][current_frame][:, ::self.d_s, :]  # todo:下采样取消注释
                    # sample['agent_feature'][:, :, 1] = sample['agent_feature'][:, :, 1]  #todo:噪音
                    # sample['agent_feature'][:, :, 6] = sample['agent_feature'][:, :, 6]
                    sample['label'] = file['label'][current_frame][:, ::self.d_s, :]
                    sample['auxiliary_label'] = file['auxiliary_label'][current_frame][:, ::self.d_s, :]
                    sample['label_mask'] = file['label_mask'][current_frame][:, ::self.d_s]
                    sample['pred_num'] = file['pred_num'][current_frame]
                    #        print(len(file['other_label'][current_frame]), file['other_label'][current_frame])
                    #         sample['other_label'] = file['other_label'][current_frame][:, ::self.d_s, :] if len(file['other_label'][current_frame]) != 0 else []
                    #         sample['other_label_mask'] = file['other_label_mask'][current_frame][:, ::self.d_s] if len(file['other_label'][current_frame]) != 0 else []
                    if self.use_other_fut:
                        delay_frame = current_frame - 0  # todo:延迟
                        if delay_frame < 30:
                            continue
                        if delay_frame in file['other_label']:
                            sample['other_label'] = file['other_label'][delay_frame][:, ::self.d_s, :]
                            sample['other_label'][:, :, 0] = sample['other_label'][:, :, 0] + 0.2  #todo:噪音
                            sample['other_auxiliary_label'] = file['other_auxiliary_label'][delay_frame][:, ::self.d_s, :]
                            sample['other_auxiliary_label'][:, :, 2] = sample['other_auxiliary_label'][:, :, 2] + 0.2  #TODO：噪音
                            sample['other_label_mask'] = file['other_label_mask'][delay_frame][:, ::self.d_s]
                        else:
                            sample['other_label'] = np.array([])
                            sample['other_auxiliary_label'] = np.array([])
                            sample['other_label_mask'] = np.array([])
                    sample['object_id_lis'] = file['object_id_lis'][current_frame]
                    sample['agent_type'] = file['agent_type'][current_frame]
                    if not self.road_prediction:
                        sample['AV_fut'] = np.flip(file["AV_fut"][current_frame][:, ::self.d_s, :], axis=1).copy()
                    sample['map_fea'] = [file['map_fea'][0][current_frame], file['map_fea'][1][current_frame]]
                    sample_lis.append(sample)
                yield sample_lis
            del data
    def __len__(self):
        return sum(map(len, self.agent_num_lis))


@torch.no_grad()
def obtain_dataset(gpu, gpu_count, seed_num, args, data_file, agent_num, is_train):
#    dataset_path = os.path.join(os.path.dirname(os.getcwd()), "dataset", "V2X-Seq-TFD-Example", "cooperative-vehicle-infrastructure", "process_for_prediction_veh_new")    #os.getcwd：返回当前工作目录；os.path.dirname：去掉文件名，返回目录
#    dataset_path = "/media/ps/ba50700a-668e-4255-9ec6-a877cfa97e41/zxy/V2X-Seq-TFD/cooperative-vehicle-infrastructure/process_for_prediction_veh/"
    dataset_path = args.data_path
    if args.dev_mode == "True":
        seed_num = 0
    print(gpu, seed_num, flush=True)


    sampler = BalancedBatchSampler(args, agent_num, data_file, seed_num=seed_num, gpu=gpu, gpu_cnt=gpu_count, is_train=is_train)


    dataset = V2XDataset(agent_num, data_file=data_file, num_current_frame=args.num_current_frame, t_h=args.t_h, d_s=args.d_s, road_prediction=args.road_prediction)


    dataloader = DataLoader(dataset, pin_memory=False, collate_fn=partial(HDGT_collate_fn, args=args, is_train=is_train), batch_sampler=sampler, num_workers=args.num_worker)


    return dataloader


import numpy as np
import torch
import dgl
import random
import math

def euclid_np(label, pred):
    return np.sqrt((label[...,0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)

uv_dict = {}
## Sparse adj mat of fully connected graph of neighborhood size
def return_uv(neighborhood_size):
    global uv_dict
    if neighborhood_size in uv_dict:
        return uv_dict[neighborhood_size]
    else:
        v = torch.LongTensor([[_]*(neighborhood_size-1) for _ in range(neighborhood_size)]).view(-1)
        u = torch.LongTensor([list(range(0, _)) +list(range(_+1,neighborhood_size)) for _ in range(neighborhood_size)]).view(-1)
        uv_dict[neighborhood_size] = (u, v)
        return (u, v)

def generate_heterogeneous_graph(agent_fea, road_agent_feature,  map_fea, agent_map_size_lis, other_label_index, args):
    max_in_edge_per_type = 32 ## For saving GPU memory  #todo  #default:32
    uv_dic = {}
    plan_agent_index_lis = []
    if not args.road_prediction:
        plan_agent_index_lis = [agent_fea.shape[0] - 1]  # 如果没有v2x的话，有规划信息的只有自车
    if args.v2x_prediction:
        plan_agent_index_lis = other_label_index + plan_agent_index_lis
    uv_dic[("agent", "self", "agent")] = [list(range(agent_fea.shape[0])), list(range(agent_fea.shape[0]))]  ## Self-loop
    num_of_agent = agent_fea.shape[0]
    # Agent fut Adj  #todo:没有规划模块的时候注释掉

    if args.use_road_obs:
        uv_dic[("road", "view", "agent")] = [[], []]
        for agent_index_i in range(num_of_agent):
            if len(road_agent_feature) > 0:
                for road_agent_index in range(len(road_agent_feature)):
                    uv_dic[("road", "view", "agent")][0] += [road_agent_index]
                    uv_dic[("road", "view", "agent")][1] += [agent_index_i]


    if args.use_planning:
        uv_dic[("agent", "planning", "agent")] = [[], []]
        for plan_agent_index in plan_agent_index_lis:
            final_dist_between_agent = euclid_np(agent_fea[plan_agent_index, -1, :][np.newaxis, 1:3], agent_fea[:, -1, 1:3])
            nearby_agent_index = np.where(final_dist_between_agent < np.maximum(agent_map_size_lis[-1][np.newaxis], agent_map_size_lis))[0]
            nearby_agent_index = np.delete(nearby_agent_index, obj=np.where(nearby_agent_index == plan_agent_index))
            if len(nearby_agent_index) > max_in_edge_per_type:
                final_dist_between_agent_sorted_nearby_index = np.argsort(final_dist_between_agent[nearby_agent_index])
                nearby_agent_index = nearby_agent_index[final_dist_between_agent_sorted_nearby_index][
                                     :max_in_edge_per_type]
            nearby_agent_index = nearby_agent_index.tolist()
            if len(nearby_agent_index) > 0:
                uv_dic[("agent", "planning", "agent")][0] += [plan_agent_index] * (len(nearby_agent_index))
                uv_dic[("agent", "planning", "agent")][1] += nearby_agent_index

    ## Agent Adj
    uv_dic[("agent", "other", "agent")] = [[], []]
    for agent_index_i in range(num_of_agent):#遍历一个sample中的所有车辆
        final_dist_between_agent = euclid_np(agent_fea[agent_index_i, -1, :][np.newaxis, 1:3], agent_fea[:, -1, 1:3])
        nearby_agent_index = np.where(final_dist_between_agent < np.maximum(agent_map_size_lis[agent_index_i][np.newaxis], agent_map_size_lis))[0]         #阈值取两辆车map_size较大的那一个，因为只要车辆在对方车辆的影响范围内，这两辆车就有对应的edge。妙！！
        nearby_agent_index = np.delete(nearby_agent_index, obj=np.where(nearby_agent_index == agent_index_i))
        if len(nearby_agent_index) > max_in_edge_per_type:
            final_dist_between_agent_sorted_nearby_index = np.argsort(final_dist_between_agent[nearby_agent_index])
            nearby_agent_index = nearby_agent_index[final_dist_between_agent_sorted_nearby_index][:max_in_edge_per_type]
        nearby_agent_index = nearby_agent_index.tolist()
        if len(nearby_agent_index) > 0:
            uv_dic[("agent", "other", "agent")][0] += [agent_index_i]*(len(nearby_agent_index))
            uv_dic[("agent", "other", "agent")][1] += nearby_agent_index

                   
    polygon_index_cnt = 0
    graphindex2polygonindex = {}
    uv_dic[("polygon", "g2a", "agent")] = [[], []]
    ## Agent_Polygon Adj
    if len(map_fea[1]) > 0:
        dist_between_agent_polygon = np.stack([(euclid_np(agent_fea[:, -1, :][:, np.newaxis, 1:], _[1][np.newaxis, :, :]).min(1)) for _ in map_fea[1]], axis=-1)
        all_agent_nearby_polygon_index_lis = dist_between_agent_polygon < agent_map_size_lis[:, np.newaxis]
        for agent_index_i in range(num_of_agent):
            nearby_polygon_index_lis = np.where(all_agent_nearby_polygon_index_lis[agent_index_i, :])[0]
            if len(nearby_polygon_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_polygon = dist_between_agent_polygon[agent_index_i, :]
                nearby_polygon_index_lis_sorted = np.argsort(current_dist_between_agent_polygon[nearby_polygon_index_lis])
                nearby_polygon_index_lis = nearby_polygon_index_lis[nearby_polygon_index_lis_sorted][:max_in_edge_per_type]
            nearby_polygon_index_lis = nearby_polygon_index_lis.tolist()
            for now_cnt, nearby_polygon_index in enumerate(nearby_polygon_index_lis):
                uv_dic[("polygon", "g2a", "agent")][0].append(polygon_index_cnt)
                uv_dic[("polygon", "g2a", "agent")][1].append(agent_index_i)
                graphindex2polygonindex[polygon_index_cnt] = nearby_polygon_index
                polygon_index_cnt += 1

    laneindex2graphindex = {}
    graphindex_cnt = 0
    uv_dic[("lane", "l2a", "agent")] = [[], []]
    uv_dic[("agent", "a2l", "lane")] = [[], []]
    ## Agent-Map Adj
    if len(map_fea[0]) > 0:
        all_polyline_coor = np.array([_["xy"] for _ in map_fea[0]])
        final_dist_between_agent_lane = euclid_np(agent_fea[:, -1, 1:3][:, np.newaxis, np.newaxis, :], all_polyline_coor[np.newaxis, :, :, :]).min(2)
        all_agent_nearby_lane_index_lis =  final_dist_between_agent_lane <  agent_map_size_lis[:, np.newaxis]
        for agent_index_i in range(num_of_agent):
            nearby_road_index_lis = np.where(all_agent_nearby_lane_index_lis[agent_index_i, :])[0]#.tolist()
            if len(nearby_road_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_lane = final_dist_between_agent_lane[agent_index_i]
                nearby_road_index_lis_sorted = np.argsort(current_dist_between_agent_lane[nearby_road_index_lis])
                nearby_road_index_lis = nearby_road_index_lis[nearby_road_index_lis_sorted][:max_in_edge_per_type]
            nearby_road_index_lis = nearby_road_index_lis.tolist()
            for now_cnt, nearby_road_index in enumerate(nearby_road_index_lis):
                if nearby_road_index not in laneindex2graphindex:
                    laneindex2graphindex[nearby_road_index] = graphindex_cnt
                    graphindex_cnt += 1
                uv_dic[("agent", "a2l", "lane")][0].append(agent_index_i)
                uv_dic[("lane", "l2a", "agent")][1].append(agent_index_i)
                uv_dic[("lane", "l2a", "agent")][0].append(laneindex2graphindex[nearby_road_index])
                uv_dic[("agent", "a2l", "lane")][1].append(laneindex2graphindex[nearby_road_index])

    lane2lane_boundary_dic = {}
    ## Map-Map Adj
    for etype in ["left", "right", "prev", "follow"]:            #12.2：居然还有这么多没看，加把劲
        uv_dic[("lane", etype, "lane")] = [[], []]
        lane2lane_boundary_dic[("lane", etype, "lane")] = []
    if len(map_fea[0]) > 0:
        all_in_graph_lane = list(laneindex2graphindex.keys())
        for in_graph_lane in all_in_graph_lane:
            info_dic = map_fea[0][in_graph_lane]
            for etype in ["left", "right", "prev", "follow"]:
                neighbors = [_ for _ in info_dic[etype] if _[0] in laneindex2graphindex]
                lane2lane_boundary_dic[("lane", etype, "lane")] += [_[1] for _ in neighbors]
                neighbors = [_[0] for _ in neighbors]
                uv_dic[("lane", etype, "lane")][0] += [laneindex2graphindex[in_graph_lane]] * len(neighbors)          #为什么要用laneindex2graphindex？因为并不是所有lane都在一个场景的考虑范围内，为了方便循环遍历，将考虑范围内的lane从1开始按整数挨个计数，而laneindex2graphindex就是与这个场景相对应的lane编号查找字典
                uv_dic[("lane", etype, "lane")][1] += [laneindex2graphindex[_] for _ in neighbors]
    
    output_dic = {}
    for _ in uv_dic:
        uv_dic[_] = (torch.LongTensor(uv_dic[_][0]), torch.LongTensor(uv_dic[_][1]))

    output_dic["uv_dic"] = uv_dic
    output_dic["graphindex2polylineindex"] = {v: k for k, v in laneindex2graphindex.items()}
    output_dic["graphindex2polygonindex"] = graphindex2polygonindex
    output_dic["boundary_type_dic"] = {k:torch.LongTensor(v) for k, v in lane2lane_boundary_dic.items()}
    return output_dic

def rotate(data, cos_theta, sin_theta):
    # print(f"data的维度: {np.shape(data)}, cos_theta的维度: {np.shape(cos_theta)}, sin_theta的维度: {np.shape(sin_theta)}")
    # print((data[..., 0]*cos_theta - data[..., 1]*sin_theta).shape)
    # if len(data[..., 0].shape) == 1:
    #     cos_theta = np.squeeze(cos_theta)
    #     sin_theta = np.squeeze(sin_theta)
    data[..., 0], data[..., 1] = data[..., 0]*cos_theta - data[..., 1]*sin_theta, data[..., 1]*cos_theta + data[..., 0]*sin_theta
    return data

def normal_agent_feature(feature, ref_coor, ref_psi,  cos_theta, sin_theta):
    """
    对各车辆的历史信息进行相对坐标系的转换，包括x,y坐标、速度v_x,v_y、航向角
    输入：n辆车的历史信息feature(n,11,10)，n辆车各相对坐标系的参考坐标ref_coor，参考航向角ref_psi，旋转矩阵的构成元素cos_theta & sin_theta
    输出：(n,11,11)数组
    """
    feature = feature[..., 1:]
    feature[..., :3] -= ref_coor[:, np.newaxis, :]
    feature[..., 0], feature[..., 1] = feature[..., 0]*cos_theta - feature[..., 1]*sin_theta, feature[..., 1]*cos_theta + feature[..., 0]*sin_theta
    feature[..., 3], feature[..., 4] = feature[..., 3]*cos_theta - feature[..., 4]*sin_theta, feature[..., 4]*cos_theta + feature[..., 3]*sin_theta
    feature[..., 5] -= ref_psi
    cos_psi = np.cos(feature[..., 5])
    sin_psi = np.sin(feature[..., 5])
    feature = np.concatenate([feature[..., :5], cos_psi[...,np.newaxis], sin_psi[...,np.newaxis], feature[..., 6:]], axis=-1)
    return feature

def normal_polygon_feature(all_polygon_coor, all_polygon_type, ref_coor, cos_theta, sin_theta):
    now_polygon_coor = all_polygon_coor - ref_coor
    rotate(now_polygon_coor, cos_theta, sin_theta)
    return now_polygon_coor,  all_polygon_type

def normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_signal, polyline_index, ref_coor, cos_theta, sin_theta):
    '''
    对各个lane的相关特征信息进行相对坐标系的转换
    输出：n条lane的相对坐标(n,21,3)，类型type，限速speed_limit，stop_point在它所在的lane相对坐标系中的坐标(s,3)，stop_point所在lane的index(0~n-1)，signal_point在它所在的lane相对坐标系中的坐标，signal_point所在lane的index(0~n-1)
    '''
    output_polyline_coor = now_polyline_coor[polyline_index] - ref_coor[:, np.newaxis, :]
    rotate(output_polyline_coor, cos_theta, sin_theta)
    # output_stop_fea = {i:np.array(now_polyline_stop[_][0]) for i, _ in enumerate(polyline_index) if len(now_polyline_stop[_]) != 0}
    output_signal_fea = {i: np.array(now_polyline_signal[_]) for i, _ in enumerate(polyline_index) if len(now_polyline_signal[_]) != 0}
    # output_stop_index, output_stop_fea = list(output_stop_fea.keys()), list(output_stop_fea.values())
    ## 计算stop_point在它所在的lane相对坐标系中的坐标
    # if len(output_stop_fea) != 0:
    #     output_stop_fea = np.stack(output_stop_fea, axis=0)
    #     output_stop_fea -= ref_coor[output_stop_index]
    #     if type(cos_theta) == np.float64:
    #         rotate(output_stop_fea, cos_theta, sin_theta)
    #     else:
    #         rotate(output_stop_fea, cos_theta[output_stop_index].flatten(), sin_theta[output_stop_index].flatten())
    ## 计算signal_point在它所在的lane相对坐标系中的坐标
    output_signal_index, output_signal_fea = list(output_signal_fea.keys()), list(output_signal_fea.values())
    if len(output_signal_fea) != 0:
        output_signal_fea = np.stack(output_signal_fea, axis=0)
#        print("output_signal_fea:", output_signal_fea[..., :2].shape, "ref_coor:", ref_coor[output_signal_index].shape)
        if output_signal_fea[..., :2].shape[1] == 1: print(output_signal_fea)
        output_signal_fea[..., :2] -= ref_coor[output_signal_index]
        if type(cos_theta) == np.float64:
            rotate(output_signal_fea, cos_theta, sin_theta)
        else:
            rotate(output_signal_fea, cos_theta[output_signal_index].flatten(), sin_theta[output_signal_index].flatten())
    else:
        output_signal_fea = np.array(output_signal_fea)
    return output_polyline_coor, now_polyline_type[polyline_index], output_signal_fea, output_signal_index

def return_rel_e_feature(src_ref_coor, dst_ref_coor, src_ref_psi, dst_ref_psi):
    rel_coor = src_ref_coor - dst_ref_coor
    if rel_coor.ndim == 0 or rel_coor.ndim == 1:
        rel_coor = np.atleast_1d(rel_coor)[np.newaxis, :]
    rel_coor = rotate(rel_coor, np.cos(-dst_ref_psi),  np.sin(-dst_ref_psi))      #notice:这里已经是-theta了
    rel_psi = np.atleast_1d(src_ref_psi - dst_ref_psi)[:, np.newaxis]
    rel_sin_theta = np.sin(rel_psi)     #sin(-(theta_v - theta_u))
    rel_cos_theta = np.cos(rel_psi)
    return np.concatenate([rel_coor, rel_sin_theta, rel_cos_theta], axis=-1)


map_size_lis = {0.0: 10, 1.0: 20, 2.0: 30}
@torch.no_grad()
def HDGT_collate_fn(batch, args, is_train):
    '''
    将batch的sample进行整合，完成的主要任务：构建异构图
    输出：dict:18
    cuda_tensor_lis: 9
    lane_n_stop_sign_fea_lis: Tensor(n,3)
    lane_n_stop_sign_index_lis: Tensor(n,)
    lane_n_signal_fea_lis: Tensor(n,14)
    lane_n_signal_index_lis: Tensor(n,)
    label_lis: Tensor(n,80,2)
    auxiliary_label_lis: Tensor(n,11,4)
    auxiliary_label_future_lis: Tensor(n,80,3)
    label_mask_lis: Tensor(n,80)
    a_e_type_dict: 3 {'self': 3,'a2l': 3,'other': 3}
    a_n_type_lis: 3
    graph_lis
    neighbor_size_lis: (16,)     #16个sample场景中的agent数量
    pred_num_lis: (16,)
    case_id_lis: 16
    object_id_lis: 16            #16个sample场景中每个agent的id
    normal_lis: （n,3）
    fname: (n,)
    '''
    agent_drop = args.agent_drop

    agent_feature_lis = [item["agent_feature"] for item in batch]
    agent_type_lis = [item["agent_type"] for item in batch]
    #agent_map_size_lis = [np.vectorize(setting_dic["agenttype2mapsize"].get)(_) for _ in agent_type_lis]
    pred_num_lis = np.array([item["pred_num"] for item in batch])
    label_lis = [item["label"] for item in batch]
    auxiliary_label_lis = [item["auxiliary_label"] for item in batch]
    label_mask_lis = [item["label_mask"] for item in batch]
    tar_id_lis = [item["tar_id"] for item in batch]
    if args.use_other_fut:
        other_label_lis = [item["other_label"] for item in batch]
        other_auxiliary_label_lis = [item["other_auxiliary_label"] for item in batch]
        other_label_mask_lis = [item["other_label_mask"] for item in batch]
    if not args.road_prediction:
        AV_fut_lis = [item["AV_fut"] for item in batch]
    if args.use_road_obs:
        road_feature_lis = [item['road_feature'] for item in batch]
        road_feature_mask_lis = [item['road_feature_mask'] for item in batch]
        road_ids_lis = [item['road_ids'] for item in batch]
        road_type_lis = [item['road_type'] for item in batch]
    map_fea_lis = [item["map_fea"] for item in batch]
    object_id_lis = [item["object_id_lis"] for item in batch]
    # file_name_lis = [item["file_name"] for item in batch]

    if agent_drop > 0 and is_train:
        for i in range(len(agent_feature_lis)):
            keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            while keep_index[:pred_num_lis[i]].sum() == 0:
                keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            origin_pred_num = pred_num_lis[i]
            original_agent_num = agent_feature_lis[i].shape[0]
            target_keep_index = keep_index[:origin_pred_num]
            agent_feature_lis[i] = agent_feature_lis[i][keep_index]
            agent_type_lis[i] = agent_type_lis[i][keep_index]
            pred_num_lis[i] = int(target_keep_index.sum())

            label_lis[i] = label_lis[i][target_keep_index]
            auxiliary_label_lis[i] = auxiliary_label_lis[i][target_keep_index]
            label_mask_lis[i] = label_mask_lis[i][target_keep_index]
            # if origin_pred_num != original_agent_num:
            #     other_label_lis[i] = other_label_lis[i][keep_index[origin_pred_num:]]
            #     other_label_mask_lis[i] = other_label_mask_lis[i][keep_index[origin_pred_num:]]

    neighbor_size = np.array([int(agent_feature_lis[i].shape[0]) for i in range(len(agent_feature_lis))])

    out_lane_n_stop_sign_fea_lis = []
    out_lane_n_stop_sign_index_lis = []
    out_lane_n_signal_fea_lis = []
    out_lane_n_signal_index_lis = []

    out_normal_lis = []
    out_graph_lis = []
    out_label_lis = []
    out_label_mask_lis = []
    out_auxiliary_label_lis = []
    out_auxiliary_label_future_lis = []
    out_road_feature_mask_lis = []
    fut_mask_lis = []
    # out_other_label_lis = []     #todo:路端时删除注释
    # out_other_label_mask_lis = []
    out_av_fut_lis = []
    lane_n_cnt = 0

    for i in range(len(agent_feature_lis)):
        all_agent_obs_final_v = np.sqrt(agent_feature_lis[i][:, -1, 4]**2+agent_feature_lis[i][:, -1, 5]**2)    #frame, x, y, z, vx, vy, heading, length, width, height, mask
        all_agent_map_size = np.vectorize(map_size_lis.__getitem__)(agent_type_lis[i])
        all_agent_map_size = all_agent_obs_final_v * 5.0 + all_agent_map_size    #todo:划定交互范围（速度*8s+缓冲值）

        now_agent_feature = agent_feature_lis[i]
        road_agent_feature = []
        now_agent_type = agent_type_lis[i]
        other_label_index = []
        other_label = []
        all_agent_fut = np.empty((0, 25, 11))
        if args.v2x_prediction:
            other_label = other_label_lis[i]
            other_auxiliary_label = other_auxiliary_label_lis[i]
            other_label_index = list(range((len(now_agent_feature) - len(other_label) - 1), (len(now_agent_feature) - 1)))
        if args.use_road_obs:
            now_road_agent_type = road_type_lis[i]
            road_agent_feature = road_feature_lis[i]
            road_agent_mask = road_feature_mask_lis[i]

        graph_dic = generate_heterogeneous_graph(now_agent_feature, road_agent_feature, map_fea_lis[i], all_agent_map_size, other_label_index, args)
        g = dgl.heterograph(data_dict=graph_dic["uv_dic"])
        g.edata['boundary_type'] = graph_dic["boundary_type_dic"]

        polylinelaneindex = list(graph_dic["graphindex2polylineindex"].values())
        polygonlaneindex = list(graph_dic["graphindex2polygonindex"].values())

        if not args.road_prediction:
            Av_fut = AV_fut_lis[i]
            all_agent_fut = Av_fut
        if args.v2x_prediction:
            #            other_fut_heading = np.arctan((other_fut[:, 1:, 1] - other_fut[:, :-1, 1]) / (other_fut[:, 1:, 0] - other_fut[:, :-1, 0]))
            #            other_fut_heading = np.concatenate([other_fut_heading, other_fut_heading[:, -1][:, np.newaxis]], axis=1)[..., np.newaxis]
            if len(other_label) != 0:
                other_fut = np.flip(np.concatenate([np.zeros((other_label.shape[0], other_label.shape[1], 1)), other_label, np.zeros((other_label.shape[0], other_label.shape[1], 1)), other_auxiliary_label], axis=-1), axis=1).copy()
                if len(all_agent_fut) != 0:
                    all_agent_fut = np.concatenate([other_fut, all_agent_fut], axis=0)
                else:
                    all_agent_fut = other_fut

        ### Type 0 edge a2a self-loop
        #以下凡是source_node为agent的edge，输出的fea均为5维，是不同node之间进行坐标转换所需要的信息，分别是：在destination_node相对坐标系下计算的deta_x,deta_y,deta_z，以及sin(theta_u - theta_v),cos(theta_u - theta_v)
        type0_u, type0_v = g.edges(etype="self")
        now_t0_v_feature = now_agent_feature[type0_v, :, :]
        now_t0_e_feature = now_agent_feature[type0_u].copy()
        if len(type0_v) == 1:
            now_t0_v_feature = now_t0_v_feature[np.newaxis, :, :]
            now_t0_e_feature = now_t0_e_feature[np.newaxis, :, :]
        now_t0_e_feature = return_rel_e_feature(now_t0_e_feature[:, -1, 1:4], now_t0_v_feature[:, -1, 1:4], now_t0_e_feature[:, -1, 6], now_t0_v_feature[:, -1, 6])      #将各对象所能观察到的最后一帧作为reference
        g.edata['a_e_fea'] = {("agent", "self", "agent"):torch.as_tensor(now_t0_e_feature.astype(np.float32))}
        g.edata['a_e_type'] = {("agent", "self", "agent"):torch.as_tensor((now_agent_type[type0_u].ravel()).astype(np.int32)).long()}     #.ravel()将选择的元素展平为一维数组

        ##Type 6 edge a2a planning agent  #todo:没有规划模块的时候注释掉
        if args.use_planning:
            type6_u, type6_v = g.edges(etype="planning")
#            type6_u = torch.tensor([pre_index_dic[_] for _ in type6_u.numpy()])  # type6_u中的index是相对all_agent_fut而言的
            type6_u = type6_u - (len(now_agent_feature) - len(other_label) - 1)
            if len(type6_v) > 0:
                now_t6_v_feature = now_agent_feature[type6_v, :, :]
                now_t6_e_feature = all_agent_fut[type6_u, :, :].copy()
                if len(type6_v) == 1:
                    now_t6_v_feature = now_t6_v_feature[np.newaxis, :, :]
                if len(type6_u) == 1:
                    now_t6_e_feature = now_t6_e_feature[np.newaxis, :, :]
                #                print("now_t6_e_feature:", now_t6_e_feature.shape, " ,now_t6_v_feature:", now_t6_v_feature[:, -1, 1:4].shape)
                now_t6_e_feature = return_rel_e_feature(now_t6_e_feature[:, -1, 1:4], now_t6_v_feature[:, -1, 1:4],
                                                        now_t6_e_feature[:, -1, 6], now_t6_v_feature[:, -1, 6])  # TODO
                g.edata['a_e_fea'] = {
                    ("agent", "planning", "agent"): torch.as_tensor(now_t6_e_feature.astype(np.float32))}
                g.edata['a_e_type'] = {("agent", "planning", "agent"): torch.as_tensor(
                    (now_agent_type[type6_u].ravel()).astype(np.int32)).long()}
            else:
                g.edata['a_e_fea'] = {("agent", "planning", "agent"): torch.zeros((0, 5))}
                g.edata['a_e_type'] = {("agent", "planning", "agent"): torch.zeros((0,)).long()}

        if args.use_road_obs:
            type7_u, type7_v = g.edges(etype="view")
            if len(type7_v) > 0:
                now_t7_v_feature = now_agent_feature[type7_v, :, :]
                now_t7_e_feature = road_agent_feature[type7_u].copy()
                if len(type7_v) == 1:
                    now_t7_v_feature = now_t7_v_feature[np.newaxis, :, :]
                    now_t7_e_feature = now_t7_e_feature[np.newaxis, :, :]
                now_t7_e_feature = return_rel_e_feature(now_t7_e_feature[:, -1, 1:4], now_t7_v_feature[:, -1, 1:4],
                                                        now_t7_e_feature[:, -1, 6], now_t7_v_feature[:, -1, 6])
                g.edata['a_e_fea'] = {("road", "view", "agent"): torch.as_tensor(now_t7_e_feature.astype(np.float32))}
                g.edata['a_e_type'] = {("road", "view", "agent"): torch.as_tensor((now_road_agent_type[type7_u].ravel()).astype(np.int32)).long()}
            else:
                g.edata['a_e_fea'] = {("road", "view", "agent"): torch.zeros((0, 5))}
                g.edata['a_e_type'] = {("road", "view", "agent"): torch.zeros((0,)).long()}

        ### Type 1 edge a2a other agent
        type1_u, type1_v = g.edges(etype="other")
        if len(type1_v) > 0:
            now_t1_v_feature = now_agent_feature[type1_v, :, :]
            now_t1_e_feature = now_agent_feature[type1_u].copy()
            if len(type1_v) == 1:
                now_t1_v_feature = now_t1_v_feature[np.newaxis, :, :]
                now_t1_e_feature = now_t1_e_feature[np.newaxis, :, :]
            now_t1_e_feature = return_rel_e_feature(now_t1_e_feature[:, -1, 1:4], now_t1_v_feature[:, -1, 1:4], now_t1_e_feature[:, -1, 6], now_t1_v_feature[:, -1, 6])
            g.edata['a_e_fea'] = {("agent", "other", "agent"):torch.as_tensor(now_t1_e_feature.astype(np.float32))}
            g.edata['a_e_type'] = {("agent", "other", "agent"):torch.as_tensor((now_agent_type[type1_u].ravel()).astype(np.int32)).long()}
        else:
            g.edata['a_e_fea'] = {("agent", "other", "agent"):torch.zeros((0, 5))}
            g.edata['a_e_type'] = {("agent", "other", "agent"):torch.zeros((0, )).long()}

        ### Type 2 Edge: Agent -> Lane  a2l
        if len(polylinelaneindex) > 0:
            now_polyline_info = [map_fea_lis[i][0][_] for _ in polylinelaneindex]
            now_polyline_coor = np.stack([_["xy"] for _ in now_polyline_info], axis=0)
            now_polyline_yaw = np.array([_["yaw"] for _ in now_polyline_info])
            now_polyline_type = np.array([_["type"] for _ in now_polyline_info])
#            now_polyline_signal = [_["signal"] if len(_["signal"]) == 0 else np.concatenate([_["signal"][[0, 1]], _["signal"][2::args.d_s]], axis=0) for _ in now_polyline_info]  #todo:下采样取消注释
            now_polyline_signal =[_["signal"] for _ in now_polyline_info]
            now_polyline_mean_coor = now_polyline_coor[:, 2, :]    #论文里不是说选择中点吗，为什么这里直接用了第三个点？  TODO
            type2_u = g.edges(etype="a2l")[0]#[0][cumu_edge_type_cnt_lis[2]:cumu_edge_type_cnt_lis[3]]
            type2_v = g.edges(etype="a2l")[1]#[1][cumu_edge_type_cnt_lis[2]:cumu_edge_type_cnt_lis[3]] - now_agent_feature.shape[0] - len(polygonlaneindex)
            if len(type2_v) > 0:
                now_t2_e_feature = now_agent_feature[type2_u].copy()
                if len(now_t2_e_feature.shape) == 2:
                    now_t2_e_feature = now_t2_e_feature[np.newaxis, :, :]
                now_t2_e_feature = return_rel_e_feature(now_t2_e_feature[:, -1, 1:3], now_polyline_mean_coor[type2_v], now_t2_e_feature[:, -1, 6], now_polyline_yaw[type2_v])
                now_t2_e_feature = np.concatenate([now_t2_e_feature[:, :2], np.zeros((now_t2_e_feature.shape[0], 1), dtype=np.float32), now_t2_e_feature[:, 2:4]], axis=1)
                g.edata['a_e_fea'] = {("agent", "a2l", "lane"):torch.as_tensor(now_t2_e_feature.astype(np.float32))}
                g.edata['a_e_type'] = {("agent", "a2l", "lane"):torch.as_tensor((now_agent_type[type2_u].ravel()-1).astype(np.int32)).long()}
            else:
                g.edata['a_e_fea'] = {("agent", "a2l", "lane"): torch.zeros((0, 5))}
                g.edata['a_e_type'] = {("agent", "a2l", "lane"): torch.zeros((0,)).long()}
        else:
            g.edata['a_e_fea'] = {("agent", "a2l", "lane"): torch.zeros((0, 5))}
            g.edata['a_e_type'] = {("agent", "a2l", "lane"): torch.zeros((0,)).long()}

               

        ### Type 3 Edge: Polygon -> Agent  g2a
        type3_u = g.edges(etype="g2a")[0]#[cumu_edge_type_cnt_lis[3]:cumu_edge_type_cnt_lis[4]]  - now_agent_feature.shape[0]
        type3_v = g.edges(etype="g2a")[1]#[cumu_edge_type_cnt_lis[3]:cumu_edge_type_cnt_lis[4]]
        if len(type3_v) > 0:
            now_polygon_type = np.array([map_fea_lis[i][1][_][0] for _ in polygonlaneindex])
            now_polygon_coor = np.stack([map_fea_lis[i][1][_][1] for _ in polygonlaneindex], axis=0)
            now_t3_v_feature = now_agent_feature[type3_v]
            if len(now_t3_v_feature.shape) == 2:
                now_t3_v_feature = now_t3_v_feature[np.newaxis, :, :]
            ref_coor = now_t3_v_feature[:, -1, 1:3][:, np.newaxis, :]
            ref_psi = now_t3_v_feature[:, -1, 6][:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_t3_e_coor_feature, now_t3_e_type_feature = normal_polygon_feature(now_polygon_coor, now_polygon_type, ref_coor, cos_theta, sin_theta)
            now_t3_e_coor_feature = np.concatenate([now_t3_e_coor_feature, np.zeros((now_t3_e_coor_feature.shape[0], now_t3_e_coor_feature.shape[1], 1), dtype=np.float32)], axis=2)
            g.edata['g2a_e_fea'] = {("polygon", "g2a", "agent"): torch.as_tensor(now_t3_e_coor_feature.astype(np.float32))}
            g.edata['g2a_e_type'] = {("polygon", "g2a", "agent"): torch.as_tensor(now_t3_e_type_feature.ravel().astype(np.int32)).long()}
        # else:
        #     g.edata['g2a_e_fea'] = {("polygon", "g2a", "agent"): torch.tensor([], dtype=torch.float32)}
        #     g.edata['g2a_e_type'] = {("polygon", "g2a", "agent"): torch.tensor([], dtype=torch.int64)}

        ### Type 4 Edge: Lane -> Agent

        if len(polylinelaneindex) > 0:                   #12.3：看到这里了，加速加速
            type4_u = g.edges(etype="l2a")[0]
            type4_v = g.edges(etype="l2a")[1]
            if len(type4_v) > 0:
                now_t4_v_feature = now_agent_feature[type4_v]
                if len(now_t4_v_feature.shape) == 2:
                    now_t4_v_feature = now_t4_v_feature[np.newaxis, :, :]
                now_t4_e_feature = return_rel_e_feature(now_polyline_mean_coor[type4_u], now_t4_v_feature[:, -1, 1:3], now_polyline_yaw[type4_u], now_t4_v_feature[:, -1, 6])
                now_t4_e_feature = np.concatenate([now_t4_e_feature[:, 0:2], np.zeros((now_t4_e_feature.shape[0], 1), dtype=np.float32), now_t4_e_feature[:, 2:]], axis=1)
                g.edata['l_e_fea'] = {("lane", "l2a", "agent"):torch.as_tensor(now_t4_e_feature.astype(np.float32))} #为啥没有'l_e_type'呢？？？
        else:
            g.edata['l_e_fea'] = {("lane", "l2a", "agent"): torch.zeros((0, 5))}
        ### Type 5 Edge: Lane -> Lane
        if len(polylinelaneindex) > 0:
            for etype in ["left", "right", "prev", "follow"]:
                type5_u = g.edges(etype=etype)[0]
                type5_v = g.edges(etype=etype)[1]
                if len(type5_v) > 0:
                    now_t5_e_feature = return_rel_e_feature(now_polyline_mean_coor[type5_u], now_polyline_mean_coor[type5_v], now_polyline_yaw[type5_u], now_polyline_yaw[type5_v])
                    now_t5_e_feature = np.concatenate([now_t5_e_feature[:, 0:2], np.zeros((now_t5_e_feature.shape[0], 1), dtype=np.float32), now_t5_e_feature[:, 2:]], axis=1)
                    g.edata['l_e_fea'] = {("lane", etype, "lane"):torch.as_tensor(now_t5_e_feature.astype(np.float32))}
                else:
                    g.edata['l_e_fea'] = {("lane", etype, "lane"): torch.zeros((0, 5))}
        else:
            for etype in ["left", "right", "prev", "follow"]:
                g.edata['l_e_fea'] = {("lane", etype, "lane"): torch.zeros((0, 5))}
        tar_id = tar_id_lis[i]
        object_id = object_id_lis[i]
        pred_index = np.where(object_id == tar_id)[0]
        now_pred_num = pred_num_lis[i]
        selected_pred_indices = list(range(0, now_pred_num))
        non_pred_indices = list(range(now_pred_num, now_agent_feature.shape[0]))
        
        #AV_fut planning trajectory feature  #todo:自车规划信息处理
        if args.use_planning:
            if not args.road_prediction:
                fut_agent_index = other_label_index + [now_agent_feature.shape[0]-1]
            else:
                fut_agent_index = other_label_index
            fut_agent_ref_coor = now_agent_feature[fut_agent_index, -1, 1:3].copy()
            #            av_fut = AV_fut_lis[i].copy()
            index_in_v2x = [_ - (len(now_agent_feature) - len(other_label) - 1) for _ in fut_agent_index]
            fut_agent_xy = all_agent_fut[index_in_v2x, :, 1:3] - fut_agent_ref_coor[:, np.newaxis, :]
            fut_agent_ref_psi = now_agent_feature[fut_agent_index, -1, 6].copy()
            sin_theta = np.sin(-fut_agent_ref_psi)[..., np.newaxis]
            cos_theta = np.cos(-fut_agent_ref_psi)[..., np.newaxis]
            rotate(fut_agent_xy, cos_theta, sin_theta)
            fut_agent_v = all_agent_fut[index_in_v2x, :, 4:6]
            rotate(fut_agent_v, cos_theta, sin_theta)
            fut_agent_psi = all_agent_fut[index_in_v2x, :, 6] - fut_agent_ref_psi[..., np.newaxis]
            cos = np.cos(fut_agent_psi)
            sin = np.sin(fut_agent_psi)
            fut_agent_fea = np.concatenate(
                [fut_agent_xy, np.zeros((fut_agent_xy.shape[0], fut_agent_xy.shape[1], 1)), fut_agent_v,
                 cos[..., np.newaxis], sin[..., np.newaxis]], axis=-1)
        #x, y, z, vx, vy, cos, sin


        ## Label + Full Agent Feature
        now_full_agent_n_feature = now_agent_feature[selected_pred_indices].copy()
        ref_coor = now_full_agent_n_feature[:, -1, 1:4].copy()
        now_label = label_lis[i][selected_pred_indices].copy()
        now_auxiliary_label = auxiliary_label_lis[i][selected_pred_indices].copy()
        now_label = now_label - ref_coor[:, np.newaxis, :2]            #各预测目标相对于当前帧坐标的label
        ref_psi = now_full_agent_n_feature[:, -1, 6][:, np.newaxis].copy()
        normal_val = np.concatenate([ref_coor[..., :2], ref_psi], axis=-1)
        out_normal_lis.append(normal_val)       #各预测目标的参考坐标x,y以及参考heading
        
        sin_theta = np.sin(-ref_psi)
        cos_theta = np.cos(-ref_psi)
        rotate(now_label, cos_theta, sin_theta)
        #print("Attention!!!")
        rotate(now_auxiliary_label, cos_theta, sin_theta)       #速度只需要进行旋转rotate就ok，因为不管在哪个静止的相对坐标系下，速度的绝对大小是不变的
        now_auxiliary_label[..., 2] = now_auxiliary_label[..., 2] - ref_psi        #各预测目标相对于当前帧heading的label
        
        now_full_agent_n_feature = normal_agent_feature(now_full_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
        now_auxiliary_label_future = now_auxiliary_label.copy()
        now_auxiliary_label = np.stack([now_full_agent_n_feature[..., 3],  now_full_agent_n_feature[..., 4], now_agent_feature[selected_pred_indices, :, 6]-ref_psi, now_full_agent_n_feature[..., -1]], axis=-1)

        now_label = now_label[pred_index]
        now_auxiliary_label = now_auxiliary_label[pred_index]
        now_auxiliary_label_future = now_auxiliary_label_future[pred_index]

        now_all_agent_n_feature = now_full_agent_n_feature
        if args.use_planning:
            new_fut_agent_fea = np.zeros((fut_agent_fea.shape[0], 25, 11))
            for _ in range(fut_agent_fea.shape[0]):
                new_fut_agent_fea[_, ...] = np.concatenate([fut_agent_fea[_][np.newaxis, ...], np.full((1, 25, 1), now_agent_feature[fut_agent_index[_], -1, 7]), np.full((1, 25, 1), now_agent_feature[fut_agent_index[_], -1, 8]),
                                                       np.full((1, 25, 1), now_agent_feature[fut_agent_index[_], -1, 9]), np.full((1, 25, 1), 1)], axis=-1)
        if args.use_road_obs:
            if len(road_agent_feature) > 0:
                last_index_of_ones = [np.where(road_agent_mask[agent_id].cumsum() == road_agent_mask[agent_id].sum())[0][0] for agent_id in range(road_agent_mask.shape[0])]
                last_index_of_ones = np.stack(last_index_of_ones)
                road_ref_coor = np.array([road_agent_feature[id, last_index_of_ones[id], 1:4] for id in range(road_agent_feature.shape[0])])
                # road_ref_coor = now_agent_feature[-1, -1, 1:4][np.newaxis, ...].copy()
                road_ref_psi = np.array([road_agent_feature[id, last_index_of_ones[id], 6] for id in range(road_agent_feature.shape[0])])[:, np.newaxis]
                # road_ref_psi = np.array([now_agent_feature[-1, -1, 6]])[:, np.newaxis].copy()
                road_sin_theta = np.sin(-road_ref_psi)
                road_cos_theta = np.cos(-road_ref_psi)
                road_agent_feature = normal_agent_feature(road_agent_feature, road_ref_coor, road_ref_psi, road_cos_theta, road_sin_theta)
            else:
                road_agent_feature = np.empty((0, 16, 10))

        if now_pred_num < now_agent_feature.shape[0]:
            now_other_agent_n_feature = now_agent_feature[non_pred_indices].copy()
            ref_coor = now_other_agent_n_feature[:, -1, 1:4]     #各个非预测目标车辆的自身参考坐标x,y
            ref_psi = now_other_agent_n_feature[:, -1, 6][:, np.newaxis].copy()      #各个非预测目标车辆的自身参考航向角theta
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_other_agent_n_feature = normal_agent_feature(now_other_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
            now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, now_other_agent_n_feature], axis=0)
            if args.use_planning:
                now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, np.zeros((now_all_agent_n_feature.shape[0], 25, now_all_agent_n_feature.shape[2]))], axis=1)  # todo:有规划时解除注释
                now_all_agent_n_feature[fut_agent_index, 16:, :] = new_fut_agent_fea
        g.ndata["a_n_fea"] = {"agent": torch.as_tensor((now_all_agent_n_feature).astype(np.float32))}
        g.ndata["a_n_type"] = {"agent": torch.as_tensor((now_agent_type).astype(np.int32)).long()}
        if args.use_road_obs:
            g.ndata["a_n_fea"] = {"road": torch.as_tensor((road_agent_feature).astype(np.float32))}
            g.ndata["a_n_type"] = {"road": torch.as_tensor((now_road_agent_type).astype(np.int32)).long()}
        
        if args.use_map:
        ## Lane Node Feature
            if len(polylinelaneindex) > 0:
                ref_coor = now_polyline_mean_coor
                ref_psi = now_polyline_yaw[:, np.newaxis].copy()
                sin_theta = np.sin(-ref_psi)
                cos_theta = np.cos(-ref_psi)
                now_lane_n_coor_feature, now_lane_n_type_feature, now_lane_n_signal_feature, now_lane_n_signal_index = normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_signal, list(range(len(now_polyline_coor))), ref_coor, cos_theta, sin_theta)
                now_lane_n_coor_feature = np.concatenate([now_lane_n_coor_feature, np.zeros((now_lane_n_coor_feature.shape[0], now_lane_n_coor_feature.shape[1], 1), dtype=np.float32)], axis=2)
                now_lane_n_signal_feature = np.concatenate([now_lane_n_signal_feature[:, 0:2], np.zeros((now_lane_n_signal_feature.shape[0], 1), dtype=np.float32), now_lane_n_signal_feature[:, 2:]], axis=1) if len(now_lane_n_signal_index) != 0 else np.array([])
                g.ndata["l_n_coor_fea"] = {"lane": torch.as_tensor(now_lane_n_coor_feature.astype(np.float32))}
                g.ndata["l_n_type_fea"] = {"lane": torch.as_tensor(now_lane_n_type_feature.astype(np.int32)).long()}

            ## Polyline Feature
            if len(polylinelaneindex) > 0:
                # if len(now_lane_n_stop_index) != 0:
                #     out_lane_n_stop_sign_fea_lis.append(now_lane_n_stop_feature)
                #     out_lane_n_stop_sign_index_lis.append(np.array(now_lane_n_stop_index) + lane_n_cnt)
                if len(now_lane_n_signal_index) != 0:
                    out_lane_n_signal_fea_lis.append(now_lane_n_signal_feature)
                    out_lane_n_signal_index_lis.append(np.array(now_lane_n_signal_index)+lane_n_cnt)
                lane_n_cnt += now_lane_n_coor_feature.shape[0]

        out_graph_lis.append(g)
        out_label_lis.append(now_label)
        out_label_mask_lis.append(label_mask_lis[i][pred_index])
        out_auxiliary_label_lis.append(now_auxiliary_label)
        out_auxiliary_label_future_lis.append(now_auxiliary_label_future)
        if args.use_road_obs:
            out_road_feature_mask_lis.append(road_agent_mask)
        # if args.use_planning:
        #     fut_mask_lis.append(fut_mask)
#        out_av_fut_lis.append(now_av_fut)  #todo:无规划模块的时候注释掉

    output_dic = {}                              #12.4：看到这里啦
    #0-x, 1-y, 2-vx, 3-vy, 4-cos_psi, 5-sin_psi, 6-length, 7-width, 8-type, 9-mask
    output_dic["cuda_tensor_lis"] = ["graph_lis"]
    if args.use_planning:
        output_dic["cuda_tensor_lis"] += ["label_lis", "label_mask_lis", "auxiliary_label_lis",
                                          "auxiliary_label_future_lis", "fut_agent_index"]
        output_dic["fut_agent_index"] = torch.as_tensor(np.array(fut_agent_index).astype(np.float32))
    else:
        output_dic["cuda_tensor_lis"] += ["label_lis", "label_mask_lis", "auxiliary_label_lis",
                                          "auxiliary_label_future_lis"]
    if args.use_road_obs:
        output_dic["cuda_tensor_lis"] += ["road_mask"]
    if args.use_map:
        if len(out_lane_n_stop_sign_fea_lis) > 0:
            output_dic["cuda_tensor_lis"] += ["lane_n_stop_sign_fea_lis", "lane_n_stop_sign_index_lis"]
            out_lane_n_stop_sign_index_lis = np.concatenate(out_lane_n_stop_sign_index_lis, axis=0)
            output_dic["lane_n_stop_sign_fea_lis"] = torch.as_tensor(
                np.concatenate(out_lane_n_stop_sign_fea_lis, axis=0).astype(np.float32))
            output_dic["lane_n_stop_sign_index_lis"] = torch.as_tensor(
                out_lane_n_stop_sign_index_lis.astype(np.int32)).long()

        if len(out_lane_n_signal_fea_lis) > 0:
            output_dic["cuda_tensor_lis"] += ["lane_n_signal_fea_lis", "lane_n_signal_index_lis"]
            out_lane_n_signal_index_lis = np.concatenate(out_lane_n_signal_index_lis, axis=0)
            output_dic["lane_n_signal_fea_lis"] = torch.as_tensor(
                np.concatenate(out_lane_n_signal_fea_lis, axis=0).astype(np.float32))
            output_dic["lane_n_signal_index_lis"] = torch.as_tensor(out_lane_n_signal_index_lis.astype(np.int32)).long()
    output_dic["label_lis"] = torch.as_tensor(np.concatenate(out_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_lis"] = torch.as_tensor(
        np.concatenate(out_auxiliary_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_future_lis"] = torch.as_tensor(
        np.concatenate(out_auxiliary_label_future_lis, axis=0).astype(np.float32))
    #    output_dic["av_fut_lis"] = torch.as_tensor(np.concatenate(out_av_fut_lis, axis=0).astype(np.float32))   #todo:无规划模块块的时候注释掉

    output_dic["label_mask_lis"] = torch.as_tensor(np.concatenate(out_label_mask_lis, axis=0).astype(np.float32))
    if args.use_road_obs:
        output_dic["road_mask"] = torch.as_tensor(np.concatenate(out_road_feature_mask_lis, axis=0).astype(np.float32))

    output_g = dgl.batch(
        out_graph_lis)  # dgl.batch()接受一个图形（Graph）对象的列表作为输入参数，然后将这些图形合并成一个大的图形。这个合并后的大图形中，原始图形中的节点和边会被整合到一个更大的图形结构中，同时保留了原始图形之间的隔离性，以便在模型中进行区分和处理
    a_e_type_dict = {}
    if args.use_map:
        if args.use_planning:
            for out_etype in ["self", "a2l", "other", "planning"]:
                a_e_type_dict[out_etype] = []
                for agent_tpye_index in range(3):
                    a_e_type_dict[out_etype].append(
                        torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_tpye_index)[0])
        else:
            for out_etype in ["self", "a2l", "other"]:
                a_e_type_dict[out_etype] = []
                for agent_tpye_index in range(3):
                    a_e_type_dict[out_etype].append(
                        torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_tpye_index)[0])
    else:
        if args.use_planning:
            for out_etype in ["self", "other", "planning"]:
                a_e_type_dict[out_etype] = []
                for agent_tpye_index in range(3):
                    a_e_type_dict[out_etype].append(
                        torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_tpye_index)[0])
        else:
            for out_etype in ["self", "other"]:
                a_e_type_dict[out_etype] = []
                for agent_tpye_index in range(3):
                    a_e_type_dict[out_etype].append(
                        torch.where(output_g.edges[out_etype].data["a_e_type"] == agent_tpye_index)[0])
    if args.use_road_obs:
        a_e_type_dict["view"] = []
        for agent_tpye_index in range(3):
            a_e_type_dict["view"].append(torch.where(output_g.edges["view"].data["a_e_type"] == agent_tpye_index)[0])
    a_n_type_lis = [torch.where(output_g.nodes["agent"].data["a_n_type"] == _)[0] for _ in range(3)]
    if args.use_road_obs:
        r_n_type_lis = [torch.where(output_g.nodes["road"].data["a_n_type"] == _)[0] for _ in range(3)]
        output_dic["r_n_type_lis"] = r_n_type_lis
    output_dic["a_e_type_dict"] = a_e_type_dict
    output_dic["a_n_type_lis"] = a_n_type_lis
    output_dic["graph_lis"] = output_g
    output_dic["neighbor_size_lis"] = neighbor_size  # 每个场景中的agent数量
    output_dic["pred_num_lis"] = pred_num_lis
    output_dic["tar_id_lis"] = tar_id_lis
    output_dic["object_id_lis"] = object_id_lis  # 所有agent的id
    output_dic["normal_lis"] = np.concatenate(out_normal_lis, axis=0)  # 所有预测目标的参考坐标x,y以及参考heading
    if "fname" in batch[0]:
        all_filename = [item["fname"] for item in batch]
        output_dic["fname"] = []
        for _ in range(len(all_filename)):
            output_dic["fname"] += [all_filename[_]] * pred_num_lis[_]  # 每个预测目标对应的raw数据集文件名
        # output_dic["file_name"] = file_name_lis
    del batch
    return output_dic