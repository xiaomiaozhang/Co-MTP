#保证车端数据有完整的未来5s未来轨迹，这样才能有准确的训练标签。车端历史轨迹不用完整，但必须保证同一agent车端和路端的历史轨迹不能同时为无
import os
import json
import gc
import pandas as pd
import numpy as np
import pickle
import multiprocessing
import warnings
import argparse
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import math
from tqdm import tqdm
from functools import partial
from itertools import chain
import itertools
import copy
import h5py
from operator import itemgetter
from maps_process import maps_process


warnings.filterwarnings("ignore")

t_h = 30
t_f = 50
frame = 0.1
interaction_radius = 50
perception_radius = 50
processed_count = multiprocessing.Value('i', 0)

def euclid(label, pred):
    return np.sqrt((label[..., 0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)    #[..., 0]指取最里层所有的0号元素

def find_target(all_track_fea, AV_fea, completed_current_index_dic,interaction_radius, now_index2id):
    '''寻找在AV一定交互范围内所有车辆的index'''
    target_track_dic = {}
    target_id_dic = {}
    current_frame_index = np.array([index for index, value in enumerate(AV_fea[:, 0]) if t_h - 1 < value < 100 - t_f])
    for AV_frame_index in current_frame_index:
        AV_current_frame = AV_fea[AV_frame_index, 0]
        AV_current_pos = AV_fea[AV_frame_index, 1:3]   #AV此刻的x, y坐标
        target_track_index = []
        target_id = []
        for track_index in range(len(completed_current_index_dic)):
            completed_current_index = completed_current_index_dic[track_index]
            track_fea = all_track_fea[track_index]
            if len(completed_current_index) == 0 or (AV_current_frame not in track_fea[completed_current_index, 0]):
                continue
            else:
                track_frame_index = np.where(track_fea[completed_current_index, 0] == AV_current_frame)[0][0]  #此刻对应每个轨迹completed_current_index里元素的index
                track_frame_index = completed_current_index[track_frame_index]  #得到此刻对应每个轨迹中第index帧
                track_current_fea = track_fea[track_frame_index, :]
                track_current_pos = track_current_fea[1:3]   #这个track此刻的x, y坐标
                #判断此刻这个轨迹的坐标是不是位于AV的交互范围内
                dist = np.sqrt(np.sum(np.square(AV_current_pos - track_current_pos)))
                if dist <= interaction_radius:
                    target_track_index.append(track_index)
                    target_id.append(now_index2id[track_index][track_frame_index])
        target_track_index = np.stack(target_track_index, axis=0) if len(target_track_index) != 0 else []
        target_id = np.stack(target_id, axis=0) if len(target_id) != 0 else []
        target_track_dic[AV_frame_index] = target_track_index  #AV第index帧对应的目标车辆轨迹index是什么
        target_id_dic[AV_frame_index] = target_id
    return current_frame_index, target_track_dic, target_id_dic

def get_all_break_point(total_lis, sub_lis):
    i = 0
    j = 0
    last_conti_index = -1
    break_point_i_lis = []
    break_point_j_lis = []
    while j < len(sub_lis):
        while total_lis[i] != int(sub_lis[j]):
            i += 1
        if last_conti_index == -1:
            last_conti_index = i
        elif i == last_conti_index + 1:
            last_conti_index += 1
        else:
            break_point_i_lis.append((last_conti_index, i))
            break_point_j_lis.append((j-1, j))
            last_conti_index = i
        j += 1
    return break_point_i_lis, break_point_j_lis

def screen_completed_track(current_frame_index, track_fea, now_mask,  now_index2id, beginning_and_end = True, real_completed = False):
    # track_fea: T, x, y, z, vx, vy, theta, length, width, height
    completed_current_index = []
    completed_current_frame = []
    now_obs_fea_lis = []   #每一个当前时刻所对应的t_h内的轨迹
    tmp_label_lis = []
    tmp_auxiliary_label_lis =[]
    now_label_mask_lis = []
    tar_id_lis = []
    all_obs_fea_padded = np.array([])
    if len(current_frame_index) == 0:
        return completed_current_index, completed_current_frame, now_obs_fea_lis, tmp_label_lis, tmp_auxiliary_label_lis, now_label_mask_lis, tar_id_lis, all_obs_fea_padded
    all_obs_frame = np.array([index for index in np.arange(0, 100-t_f, dtype=float) if index in track_fea[:, 0]])
    if len(all_obs_frame) == 100 - t_f:
        all_obs_fea_padded = np.concatenate([track_fea[:(100 - t_f), :7], np.array([track_fea[:, -3].mean()] * (100 - t_f))[:, np.newaxis],
             np.array([track_fea[:, -2].mean()] * (100 - t_f))[:, np.newaxis], np.array([track_fea[:, -1].mean()] * (100 - t_f))[:, np.newaxis]], axis=-1)
    elif len(all_obs_frame) != 0:
        start_step = int(all_obs_frame[0])
        end_step = int(all_obs_frame[-1])
        padded_fea = track_fea[:len(all_obs_frame), 1:7]
        if len(all_obs_frame) != end_step - start_step + 1:
            break_point_i_lis, break_point_j_lis = get_all_break_point(list(range(0, (100 - t_f))), track_fea[:len(all_obs_frame), 0])
            line_lis = [track_fea[:break_point_j_lis[0][0] + 1, 1:7]]
            for bi in range(len(break_point_i_lis)):
                line = interp.interp1d(x=[0.0, 1.0],
                                    y=track_fea[break_point_j_lis[bi][0]:break_point_j_lis[bi][1] + 1, 1:7],
                                    assume_sorted=True, axis=0)(np.linspace(0.0, 1.0, break_point_i_lis[bi][1] - break_point_i_lis[bi][0] + 1))
                v_xy = line[1:, 3:5]
                cumsum_step = np.cumsum(v_xy / 10.0, axis=0) + line[0, 0:2][np.newaxis, :]
                line[1:, 0:2] = cumsum_step
                line_lis.append(line[1:-1, :])
                if bi == len(break_point_i_lis) - 1:
                    line_lis.append(track_fea[break_point_j_lis[bi][1]:len(all_obs_frame), 1:7])
                else:
                    line_lis.append(track_fea[break_point_j_lis[bi][1]:break_point_j_lis[bi + 1][0] + 1, 1:7])
            padded_fea = np.concatenate(line_lis, axis=0)
        if start_step > 0:
            v_xy = padded_fea[0, 3:5]
            cumsum_step = np.cumsum((-v_xy / 10.0)[np.newaxis, :].repeat(start_step, axis=0), axis=0)[::-1, :] + padded_fea[0, 0:2][np.newaxis, :]
            padded_fea = np.concatenate([padded_fea[0][np.newaxis, :].repeat(start_step, axis=0), padded_fea], axis=0)
            padded_fea[:start_step, :2] = cumsum_step
        if end_step < 100 - t_f - 1:
            end_step_fea = track_fea[len(all_obs_frame) - 1, :7]
            need_frames = 100 - t_f - end_step_fea[0] - 1
            if len(all_obs_frame) != len(track_fea):
                fut_start_fea = track_fea[len(all_obs_frame), :7]
                num_frames = int(fut_start_fea[0] - end_step_fea[0] - 1)
                cumsum_step = []
                for g in range(1, num_frames + 1):
                    if g > need_frames:
                        continue
                    else:
                        interpolated_value = end_step_fea[1:7] + (fut_start_fea[1:7] - end_step_fea[1:7]) * g / (num_frames + 1)
                        cumsum_step.append(interpolated_value)
                cumsum_step = np.stack(cumsum_step, axis=0)
                padded_fea = np.concatenate([padded_fea, cumsum_step], axis=0)
            else:
                v_xy = padded_fea[-1, 3:5]
                cumsum_step = np.cumsum((v_xy / 10.0)[np.newaxis, :].repeat(100 - t_f - end_step - 1, axis=0),
                                        axis=0) + padded_fea[-1, 0:2][np.newaxis, :]  # 假设从end_step开始，车辆的速度保持不变
                padded_fea = np.concatenate(
                    [padded_fea, padded_fea[-1][np.newaxis, :].repeat(100 - t_f - end_step - 1, axis=0)], axis=0)
                padded_fea[end_step + 1:, :2] = cumsum_step
        all_obs_fea_padded = np.concatenate([np.array([i for i in range(0, 100 - t_f)])[:, np.newaxis], padded_fea, np.array([track_fea[:, -3].mean()] * (100 - t_f))[:, np.newaxis],
            np.array([track_fea[:, -2].mean()] * (100 - t_f))[:, np.newaxis], np.array([track_fea[:, -1].mean()] * (100 - t_f))[:, np.newaxis]], axis=-1)  # 可观察到的长度取平均后进行复制
    for current_index in current_frame_index[:, 0]:
        current_index = int(current_index)
        current_frame = int(track_fea[current_index, 0])
        raw_hist_frame = list(range(current_frame - t_h, current_frame))
        raw_fut_frame = list(range(current_frame + 1, current_frame + t_f + 1))
        hist_frame = [frame_1 for frame_1 in raw_hist_frame if frame_1 in track_fea[:,0]]
        fut_frame = [frame_2 for frame_2 in raw_fut_frame if frame_2 in track_fea[:,0]]
        # 判断整条轨迹是否完整
        if real_completed:
            beginning_and_end = False
            if len(hist_frame) == t_h and len(fut_frame) == t_f:
                if max(now_index2id[current_index - t_h:current_index + t_f + 1]) > 200000: 
                    continue
                completed_current_index.append(current_index)
                completed_current_frame.append(current_frame)
                now_obs_fea_padded = np.concatenate([track_fea[current_index - t_h:current_index + 1, :], now_mask[:, np.newaxis]], axis=-1)
                now_obs_fea_lis.append(now_obs_fea_padded)
                # 处理未来时刻信息及掩码
                now_future_fea = track_fea[current_index + 1:]
                tmp_label_lis.append(now_future_fea[:, 1:3][np.newaxis, :, :])
                tmp_auxiliary_label_lis.append(now_future_fea[:, 4:7][np.newaxis, :, :])
                now_label_mask_lis.append(np.array([1] * t_f))
                tar_id_lis.append(now_index2id[current_index])
        # 判断轨迹开头、结尾是否完整
        if beginning_and_end:
            if len(hist_frame) == 0 or len(fut_frame) == 0:
                continue
            if hist_frame[0] == current_frame - t_h and fut_frame[-1] == current_frame + t_f:
                if max(now_index2id[current_index - len(hist_frame):current_index + len(fut_frame) + 1]) > 200000: 
                    continue
                completed_current_index.append(current_index)
                completed_current_frame.append(current_frame)
                start_index = np.where(track_fea[:, 0] == hist_frame[0])[0][0]
                end_index = np.where(track_fea[:, 0] == fut_frame[-1])[0][0]
                #补齐中间历史时刻缺失的轨迹
                now_obs_fea = track_fea[start_index:current_index + 1]
                if len(now_obs_fea) == t_h + 1:
                    now_obs_fea_padded = np.concatenate([now_obs_fea, now_mask[:, np.newaxis]], axis=-1)
                else:
                    padded_fea = all_obs_fea_padded[current_frame - t_h:current_frame + 1, :7]
                    x = 0
                    for new_frame in np.arange(hist_frame[0], hist_frame[0] + t_h + 1, dtype=float):
                        if new_frame not in now_obs_fea[:, 0]:
                            now_mask[x] = 0
                        x = x + 1
                    now_obs_fea_padded = np.concatenate([padded_fea, np.array([now_obs_fea[:, -3].mean()] * (t_h + 1))[:, np.newaxis],
                        np.array([now_obs_fea[:, -2].mean()] * (t_h + 1))[:, np.newaxis],np.array([now_obs_fea[:, -1].mean()] * (t_h + 1))[:, np.newaxis], now_mask[:, np.newaxis]], axis=-1)
                now_obs_fea_lis.append(now_obs_fea_padded)
                #处理未来时刻信息及掩码
                now_future_fea = track_fea[current_index + 1:end_index + 1]
                now_label_mask = np.array([0] * t_f)
                tmp_label = np.zeros((1, t_f, 2))
                tmp_auxiliary_label = np.zeros((1, t_f, 3))
                for label_time_index_i in range(now_future_fea.shape[0]):
                    label_time_index_in_lis = int(now_future_fea[label_time_index_i, 0]) - (current_frame + 1)
                    now_label_mask[label_time_index_in_lis] = 1
                    tmp_label[0, label_time_index_in_lis, :] = now_future_fea[label_time_index_i, [1, 2]]
                    tmp_auxiliary_label[0, label_time_index_in_lis, :] = now_future_fea[label_time_index_i, [4, 5, 6]]
                tmp_label_lis.append(tmp_label)
                tmp_auxiliary_label_lis.append(tmp_auxiliary_label)
                now_label_mask_lis.append(now_label_mask)
                tar_id_lis.append(now_index2id[current_index])
    return completed_current_index, completed_current_frame, now_obs_fea_lis, tmp_label_lis, tmp_auxiliary_label_lis, now_label_mask_lis, tar_id_lis, all_obs_fea_padded

def process_AV_df(AV_df, t_min):
    #把AV_df处理成和下方now_fea相同的数据格式
    AV_track = np.transpose(np.array([AV_df.timestamp, AV_df.x, AV_df.y, AV_df.z, AV_df.v_x, AV_df.v_y, AV_df.theta]))
    AV_fea = []
    for index in range(len(AV_track)):
        AV_track[index, 0] = int(np.round((AV_track[index, 0] - t_min) / frame))
        AV_fea.append(np.array([int(AV_track[index, 0]), float(AV_track[index, 1]), float(AV_track[index, 2]),
                                 float(AV_track[index, 3]), float(AV_track[index, 4]), float(AV_track[index, 5]), float(AV_track[index, 6]), ]))
    AV_fea = np.stack(AV_fea, axis=0)
    return AV_fea

def cut(num, c):
    str_num = str(num)

    return int(str_num[:str_num.index('.') + 1 + c])

def mask_obs_traj(traj, frame):
    mask_lis = []
    type_lis = []
    traj_new_lis = []
    all_ids = np.unique(traj[:, -2])
    obs_ids = all_ids.copy()
    for id in all_ids:
        mask = np.zeros(31)
        traj_new = np.zeros((31, 10))
        id_traj = traj[np.where(traj[:, -2] == id)[0]]
        type = id_traj[0, -1]
        mask[id_traj[:, 0].astype(int) - (frame - 30)] = 1
        traj_new[id_traj[:, 0].astype(int) - (frame - 30)] = id_traj[:, :-2]
        traj_new[:, 0] = np.arange(frame - 30, frame + 1)
        mask_ds = mask[::2]
        if mask_ds.max() != 1:
            obs_ids = np.delete(obs_ids, np.where(obs_ids == id)[0])
            continue
        mask_lis.append(mask)
        type_lis.append(type)
        traj_new_lis.append(traj_new)
    mask = np.stack(mask_lis, axis=0)
    type = np.array(type_lis, dtype=int)
    traj_new = np.stack(traj_new_lis, axis=0)
    return traj_new, mask, obs_ids, type

def mask_fut_traj(traj, frame):
    mask_lis = []
    type_lis = []
    traj_new_lis = []
    all_ids = np.unique(traj[:, -2])
    obs_ids = all_ids.copy()
    for id in all_ids:
        mask = np.zeros(50)
        traj_new = np.zeros((50, 10))
        id_traj = traj[np.where(traj[:, -2] == id)[0]]
        type = id_traj[0, -1]
        mask[id_traj[:, 0].astype(int) - (frame + 1)] = 1
        traj_new[id_traj[:, 0].astype(int) - (frame + 1)] = id_traj[:, :-2]
        traj_new[:, 0] = np.arange(frame + 1, frame + 51)
        mask_ds = mask[::2]
        if mask_ds.max() != 1:
            obs_ids = np.delete(obs_ids, np.where(obs_ids == id)[0])
            continue
        mask_lis.append(mask)
        type_lis.append(type)
        traj_new_lis.append(traj_new)
    mask = np.stack(mask_lis, axis=0)
    type = np.array(type_lis, dtype=int)
    traj_new = np.stack(traj_new_lis, axis=0)
    return traj_new, mask, obs_ids, type




def process(iter, x):
    lane_fea = copy.deepcopy(new_lane_fea)
    polygon_fea = copy.deepcopy(all_polygon_fea)
    car_df = pd.read_csv(os.path.join(car_data_path, iter))
    road_df = pd.read_csv(os.path.join(road_data_path, iter))
    # co_df = pd.read_csv(os.path.join(co_data_path, iter))
    car_raw_df = pd.read_csv(os.path.join(car_raw_path, iter))
    light_df = pd.read_csv(os.path.join(traffic_light_path, iter))
    AV_df = car_raw_df.loc[car_raw_df.tag == 'AV']
    car_df = car_df[[
        'header.lidar_timestamp',
        'id',
        'type',
        'sub_type',
        'position.x',
        'position.y',
        'position.z',
        'length',
        'width',
        'height',
        'theta',
        'velocity.x',
        'velocity.y',
        'tag'
    ]]
    car_df.columns = [
        'timestamp', 'id', 'type', 'sub_type',
        'x', 'y', 'z', 'length', 'width', 'height', 'theta',
        'v_x', 'v_y', 'tag'
    ]
    road_df = road_df[[
        'header.lidar_timestamp',
        'id',
        'type',
        'sub_type',
        'position.x',
        'position.y',
        'position.z',
        'length',
        'width',
        'height',
        'theta',
        'velocity.x',
        'velocity.y',
        'tag'
    ]]
    road_df.columns = [
        'timestamp', 'id', 'type', 'sub_type',
        'x', 'y', 'z', 'length', 'width', 'height', 'theta',
        'v_x', 'v_y', 'tag'
    ]
    
    road_df.drop(road_df.loc[road_df.type == 'EGO_VEHICLE', 'type'].index, inplace=True)
    road_df.type = road_df.type.map(lambda x: name2id[x])
    car_df.type = car_df.type.map(lambda x: name2id[x])
    t_min = min(AV_df.timestamp.astype(float))

    #提取AV中的有效信息
    AV_fea = process_AV_df(AV_df, t_min)

    tar_id_dic = {}
    car_obs_fea_dic = {}
    road_obs_fea_dic = {}
    label_dic = {}
    other_fut_fea_dic = {}
    car_obs_mask_dic = {}
    road_obs_mask_dic = {}
    label_mask_dic = {}
    other_fut_mask_dic = {}
    car_obs_ids_dic = {}
    road_obs_ids_dic = {}
    label_ids_dic = {}
    other_fut_ids_dic = {}
    car_type_dic = {}
    road_type_dic = {}
    label_type_dic = {}
    other_fut_type_dic = {}
    AV_fut_dic = {}

    #找出这个场景中x, y坐标的最小值
    x_min = min(min(car_df.x), min(road_df.x))
    x_max = max(max(car_df.x), max(road_df.x))
    y_min = min(min(car_df.y), min(road_df.y))
    y_max = max(max(car_df.y), max(road_df.y))
    car_track = np.transpose(np.array([round((car_df.timestamp.astype(float) - t_min) / frame), car_df.x.astype(float), car_df.y.astype(float), car_df.z.astype(float),car_df.v_x.astype(float),
                                       car_df.v_y.astype(float), car_df.theta.astype(float), car_df.length.astype(float), car_df.width.astype(float), car_df.height.astype(float),
                                       car_df.id.astype(int), car_df.type.astype(int)]))
    agent_tag = np.array([car_df.tag.astype(int)])
    del car_df
    road_track = np.transpose(np.array([round((road_df.timestamp.astype(float) - t_min) / frame), road_df.x.astype(float), road_df.y.astype(float), road_df.z.astype(float),
                                        road_df.v_x.astype(float), road_df.v_y.astype(float), road_df.theta.astype(float),
                                        road_df.length.astype(float), road_df.width.astype(float), road_df.height.astype(float), road_df.id.astype(int), road_df.type.astype(int)]))
    del road_df
    
    target_id = np.unique(car_track[np.where(agent_tag[0, :] == 1)[0], -2])   
    other_id = np.unique(car_track[np.where(agent_tag[0, :] == 6)[0], -2])
    AV_id = np.unique(car_track[np.where(agent_tag[0, :] == 0)[0], -2])
    for frame_index in range(30, 50):
        current_frame = AV_fea[frame_index, 0]
        AV_pos = AV_fea[frame_index, [1, 2]]
        AV_fut = AV_fea[frame_index + 1:frame_index + t_f + 1, :][np.newaxis, :, :]

        car_obs_fea_dic[frame_index] = car_track[np.where((car_track[:, 0] >= (frame_index - t_h)) & (car_track[:, 0] <= frame_index))[0]]
        car_obs_fea_dic[frame_index], car_obs_mask_dic[frame_index], car_obs_ids_dic[frame_index], car_type_dic[frame_index] = mask_obs_traj(car_obs_fea_dic[frame_index], frame_index)
        road_obs_fea_dic[frame_index] = road_track[np.where((road_track[:, 0] >= (frame_index - t_h)) & (road_track[:, 0] <= frame_index))[0]]
        road_obs_fea_dic[frame_index], road_obs_mask_dic[frame_index], road_obs_ids_dic[frame_index], road_type_dic[frame_index] = mask_obs_traj(road_obs_fea_dic[frame_index], frame_index)
        label_dic[frame_index] = car_track[np.where((car_track[:, 0] > frame_index) & (car_track[:, 0] <= (frame_index + t_f)))[0]]
        label_dic[frame_index], label_mask_dic[frame_index], label_ids_dic[frame_index], label_type_dic[frame_index] = mask_fut_traj(label_dic[frame_index], frame_index)
        AV_fut_dic[frame_index] = AV_fea[np.where((AV_fea[:, 0] > frame_index) & (AV_fea[:, 0] <= (frame_index + t_f)))[0]]

        #当把所有标agent的都看作target时，解除注释  todo
        # agent_id = np.setdiff1d(car_obs_ids_dic[frame_index], other_id)
        # target_id = np.setdiff1d(agent_id, AV_id)

        #只把指定target看作目标时解除注释 todo
        if len(np.where(car_obs_ids_dic[frame_index] == target_id)[0]) == 0:
            print("No target!")
            return
        #当把所有标agent的都看作target时，解除注释  todo
        # for id in target_id:
        #     if len(np.where(car_obs_ids_dic[frame_index] == id)[0]) == 0:
        #         target_id = np.delete(target_id, np.where(target_id == target_id[0])[0])
        # if len(target_id) == 0:
        #     print("No target!!!")
        #     return
        tar_id_dic[frame_index] = target_id

        other_fut_ids_dic[frame_index] = other_id
        fut_to_all = [np.where(label_ids_dic[frame_index] == other_id)[0][0] for other_id in other_fut_ids_dic[frame_index] if other_id in label_ids_dic[frame_index]]
        other_fut_fea_dic[frame_index] = label_dic[frame_index][fut_to_all]
        other_fut_mask_dic[frame_index] = label_mask_dic[frame_index][fut_to_all]
        other_fut_type_dic[frame_index] = label_type_dic[frame_index][fut_to_all]
        label_to_all = [np.where(label_ids_dic[frame_index] == tar_id)[0][0] for tar_id in tar_id_dic[frame_index]]
        label_dic[frame_index] = label_dic[frame_index][label_to_all]
        label_mask_dic[frame_index] = label_mask_dic[frame_index][label_to_all]
        label_type_dic[frame_index] = label_type_dic[frame_index][label_to_all]

    ##处理交通信号灯信息
    light_df.state = light_df.color_1.map(lambda x: color2num[x])
    light_data = np.concatenate([np.array(light_df.timestamp)[:, np.newaxis], np.array(light_df.x)[:, np.newaxis], np.array(light_df.y)[:, np.newaxis], np.array(light_df.lane_id)[:, np.newaxis], np.array(light_df.state)[:, np.newaxis]], axis=1)
    for index in range(len(light_data)):
        light_data[index, 0] = int(np.round((light_data[index, 0] - t_min) / frame))
    all_dynamic_map_fea_dic = {}
    dynamic_map_key = []
    all_current_frame = AV_fea[30:50, 0]
    for light_index, light_dataleaf in enumerate(light_data):   #bug: 83155210_1_-2
        if light_dataleaf[3] not in list(map_data['lane_id_to_numid'].keys()):
            continue
        if [light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]]] not in dynamic_map_key:
            dynamic_map_key.append([light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]]])
            all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])] = {}
            for current_frame in all_current_frame:
                all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])][int(current_frame)] = [3]*(t_h + 1)  #没有信息的置为3，算是一种掩码
        for current_frame in all_current_frame:
            if current_frame - t_h <= light_dataleaf[0] <= current_frame:
                all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])][int(current_frame)][light_dataleaf[0]-(int(current_frame)-t_h)] = light_dataleaf[4]
    #检查每一帧是否都有交通灯信息
    for light_id, light_fea in all_dynamic_map_fea_dic.items():
        for current_frame, light_value in light_fea.items():
            if [3] in light_value:
                for index, value in enumerate(light_value):
                    if value == [3]:
                        all_dynamic_map_fea_dic[light_id][current_frame][index] = all_dynamic_map_fea_dic[light_id][current_frame][index - 1]
#    all_dynamic_map_fea_dic = {k: v[::track_ds] for k, v in all_dynamic_map_fea_dic.items()}  # 对动态地图信息下采样
    all_dynamic_map_fea = {int(k[2]): {current_frame: np.array([k[0], k[1]] + flag) for current_frame, flag in v.items()} for k, v in all_dynamic_map_fea_dic.items()}

    for old_lane_id in all_dynamic_map_fea:
        new_lane_id_lis = old_lane_id_to_new_lane_index_lis[old_lane_id]
        for _ in new_lane_id_lis:
            if _ < lane_num:
                lane_fea[_]["signal"].append(all_dynamic_map_fea[old_lane_id])

    all_agent_map_size_dic = {k: np.ones(v.shape[0])*600.0 for k, v in car_obs_fea_dic.items()}   #todo:999.0可根据需求更改
    # ## Split Too Much Agent
    final_lane_fea_dic = {}
    final_polygon_fea_dic = {}
    lane_new_index_to_final_index_dic = {}
    polygon_new_index_to_final_index_dic = {}
    for frame_index in range(30, 50):
        lane_new_index_to_final_index_dic[frame_index] = {}
        if len(lane_fea) > 0:
            new_dist_between_agent_lane = (euclid(car_obs_fea_dic[frame_index][:, -1, [1, 2]][:, np.newaxis, np.newaxis, :], np.stack([_["xy"] for _ in lane_fea])[np.newaxis, :, :, :]).min(2) < all_agent_map_size_dic[frame_index][:, np.newaxis])
            nearby_lane_new_index_lis = np.where(new_dist_between_agent_lane)[1].tolist() 
            nearby_lane_new_index_lis = np.unique(np.array(nearby_lane_new_index_lis))
            lane_new_index_to_final_index_dic[frame_index] = {index_i: i for i, index_i in enumerate(nearby_lane_new_index_lis)}  # lane只要和场景中的任一一辆有效车的距离在限制范围内，那这个lane就保留下来；新的index是与agent_id有关系的
        final_lane_fea_dic[frame_index] = [{} for _ in range(len(lane_new_index_to_final_index_dic[frame_index]))]
        for lane_new_index in lane_new_index_to_final_index_dic[frame_index]:
            i = lane_new_index_to_final_index_dic[frame_index][lane_new_index]
            for transfer_key in ["xy", "type", "signal", "yaw"]:
                if transfer_key == "signal":
                    final_lane_fea_dic[frame_index][i][transfer_key] = [] if len(lane_fea[lane_new_index][transfer_key]) == 0 else lane_fea[lane_new_index][transfer_key][0][frame_index]
                else:
                    final_lane_fea_dic[frame_index][i][transfer_key] = lane_fea[lane_new_index][transfer_key]
            for transfer_key in ["left", "right", "prev", "follow"]:
                if transfer_key in lane_fea[lane_new_index] and len(lane_fea[lane_new_index][transfer_key]) != 0:
                    final_lane_fea_dic[frame_index][i][transfer_key] = []
                    for _ in lane_fea[lane_new_index][transfer_key]:
                        if len(_) != 0:
                            if _[0] in lane_new_index_to_final_index_dic[frame_index]:
                                final_lane_fea_dic[frame_index][i][transfer_key].append([lane_new_index_to_final_index_dic[frame_index][_[0]], _[1]])
                else:
                    final_lane_fea_dic[frame_index][i][transfer_key] = []
        polygon_new_index_to_final_index_dic[frame_index] = {}

        if len(polygon_fea) > 0:
            new_dist_between_agent_polygon = (euclid(car_obs_fea_dic[frame_index][:, -1, [1, 2]][:, np.newaxis, np.newaxis, :], np.stack([_[1] for _ in polygon_fea], axis=0)[np.newaxis, :, :, :]).min(2) < all_agent_map_size_dic[frame_index][:, np.newaxis])
            nearby_polygon_lis = np.where(new_dist_between_agent_polygon)[1].tolist()
            nearby_polygon_lis = np.unique(np.array(nearby_polygon_lis))
            polygon_new_index_to_final_index_dic[frame_index] = {index_i: i for i, index_i in enumerate(nearby_polygon_lis)}
        final_polygon_fea_dic[frame_index] = [[] for _ in range(len(polygon_new_index_to_final_index_dic[frame_index]))]
        for polygon_new_index in polygon_new_index_to_final_index_dic[frame_index]:
            final_polygon_fea_dic[frame_index][polygon_new_index_to_final_index_dic[frame_index][polygon_new_index]] = polygon_fea[polygon_new_index]

    #all_data里加入AV的信息
    for current_frame, fea in car_obs_fea_dic.items():
        AV = np.concatenate([AV_fea[current_frame - t_h:current_frame + 1, :], np.full((t_h+1, 1), 4), np.full((t_h+1, 1), 2), np.full((t_h+1, 1), 1.5)], axis=1)
        car_obs_fea_dic[current_frame] = np.concatenate([fea, AV[np.newaxis, :, :]], axis=0)
    for current_frame, id in car_obs_ids_dic.items():
        car_obs_ids_dic[current_frame] = np.concatenate([id, [int(np.array(AV_df.id)[0])]])
    for current_frame, type in car_type_dic.items():
        car_type_dic[current_frame] = np.concatenate([type, [2]])

    all_data = {}
    all_data["fname"] = iter
    all_data["agent_feature"] = car_obs_fea_dic  # 车端所感知到的历史轨迹（预测目标+AV）
    all_data["agent_feature_mask"] = car_obs_mask_dic
    all_data["agent_ids"] = car_obs_ids_dic
    all_data["agent_type"] = car_type_dic
    all_data["road_feature"] = road_obs_fea_dic
    all_data["road_feature_mask"] = road_obs_mask_dic
    all_data["road_ids"] = road_obs_ids_dic
    all_data["road_type"] = road_type_dic
    all_data["tar_id"] = tar_id_dic
    all_data["label"] = label_dic  # 所有预测目标的真实未来信息
    all_data["label_mask"] = label_mask_dic  # 所有预测目标关于未来50帧轨迹是否有效的掩码
    all_data["other_fut_fea"] = other_fut_fea_dic
    all_data["other_fut_mask"] = other_fut_mask_dic
    all_data["other_fut_ids"] = other_fut_ids_dic
    all_data["other_fut_type"] = other_fut_type_dic

    all_data["AV_fut"] = AV_fut_dic
    all_data["map_fea"] = [final_lane_fea_dic, final_polygon_fea_dic]

    return all_data, {frame_index: car_obs_fea_dic[frame_index].shape[0] + len(final_lane_fea_dic[frame_index]) + len(final_polygon_fea_dic[frame_index]) for frame_index in range(30, 50)}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='process fused v2x-seq to prediction')
    parser.add_argument("--data_root", type=str, default="/data/lixc/Co-MTP/visual_raw_data/")     
    parser.add_argument("--split", help="split.", type=str, default='val')  # train; val; test_obs

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    name2id = {
        "VEHICLE": 2,
        "BICYCLE": 1,
        "PEDESTRIAN": 0,
        "UNKNOWN_UNMOVABLE": 3,
        "UNKNOWN": 4
    }

    color2num = {
        "RED": 0,
        "YELLOW": 1,
        "GREEN": 2
    }

    car_data_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/tfd_car', args.split, 'data')
    road_data_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/tfd_road', args.split, 'data')
    traffic_light_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/traffic-light/', args.split)
    data_dest = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/process_newv2x', args.split, 'data')  #存储数据处理的地址  # 'data'  #路端存储数据的地址：process_for_prediction_road

    data_ls_1 = os.listdir(car_data_path)
    data_ls_2 = os.listdir(road_data_path)

    car_raw_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/vehicle-trajectories', args.split)

    if not os.path.exists(data_dest):
        os.makedirs(data_dest)


    with open('/data/lixc/Co-MTP/V2X-Seq-TFD/map_files/yizhuang_PEK_vector_map.json', 'r') as file:     
        map_data = json.load(file)
    new_lane_fea, old_lane_id_to_new_lane_index_lis, all_polygon_fea = maps_process(map_data)
    lane_num = len(new_lane_fea)
    print("Static Map Done!!! ")

    num_of_element_lis = []
    tasks = [(data, i) for i, data in enumerate(data_ls_1)]
    epo = 0
    save_count = 0
    data_num = len(tasks)

    with multiprocessing.Pool(processes=8) as pool:
        while 500 * epo < data_num:
            start_index = 500 * epo
            end_index = min((epo + 1) * 500, data_num)
            result = pool.starmap(process, tasks[start_index:end_index])  # 使用 starmap 来传递多个参数
            result = list(filter(lambda x: x is not None, result))
            for num in range(len(result)):
                with open(os.path.join(data_dest, str(save_count + num) + ".pkl"), "wb") as g:
                    pickle.dump(result[num][0], g)
            num_of_element_lis.append([e[1] for e in result])
            print("保存成功！")
            save_count = save_count + len(result)
            epo = epo + 1
            del result
            gc.collect()
    pool.close()
    pool.join()
 
    print("Preprocess Done")
 
    num_of_element_lis = list(itertools.chain(*num_of_element_lis))
    with open(os.path.join(data_dest, "number_of_dataset.pkl"), "wb") as g:
        pickle.dump(np.array(np.stack(num_of_element_lis)), g)

    print("Dataset Processing Done!!!")
    
    
