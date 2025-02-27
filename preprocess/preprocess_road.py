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
#from tqdm import tqdm
from functools import partial
from itertools import chain
import itertools
import copy
from operator import itemgetter
from maps_process import maps_process

warnings.filterwarnings("ignore")

t_h = 30
t_f = 50
frame = 0.1
interaction_radius = 50
perception_radius = 50
processed_count = multiprocessing.Value('i', 0)
# manager = multiprocessing.Manager()
# num_of_element_lis = manager.list()
# result_lis = manager.list()

def euclid(label, pred):
    return np.sqrt((label[..., 0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)    #[..., 0]指取最里层所有的0号元素
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
 #               now_mask[break_point_i_lis[bi][0] + 1:break_point_i_lis[bi][1]] = 0
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
 #           now_mask[:start_step] = 0
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
 #           now_mask[end_step + 1:] = 0
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
                tar_id_lis.append(now_index2id)
                completed_current_index.append(current_index)
                completed_current_frame.append(current_frame)
                now_obs_fea_padded = np.concatenate([track_fea[current_index - t_h:current_index + 1, :], now_mask[:, np.newaxis]], axis=-1)
                now_obs_fea_lis.append(now_obs_fea_padded)
                # 处理未来时刻信息及掩码
                now_future_fea = track_fea[current_index + 1:]
                tmp_label_lis.append(now_future_fea[:, 1:3][np.newaxis, :, :])
                tmp_auxiliary_label_lis.append(now_future_fea[:, 4:7][np.newaxis, :, :])
                now_label_mask_lis.append(np.array([1] * t_f))
        # 判断轨迹开头、结尾是否完整
        if beginning_and_end:
            if len(hist_frame) == 0 or len(fut_frame) == 0:
                continue
            if hist_frame[0] == current_frame - t_h and fut_frame[-1] == current_frame + t_f:
                tar_id_lis.append(now_index2id)
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
    return completed_current_index, completed_current_frame, now_obs_fea_lis, tmp_label_lis, tmp_auxiliary_label_lis, now_label_mask_lis, tar_id_lis, all_obs_fea_padded

def cut(num, c):
    str_num = str(num)

    return int(str_num[:str_num.index('.') + 1 + c])

def process(iter, x):
    lane_fea = copy.deepcopy(new_lane_fea)
    polygon_fea = copy.deepcopy(all_polygon_fea)
    road_df = pd.read_csv(os.path.join(data_path, iter))
    light_df = pd.read_csv(os.path.join(traffic_light_path, iter))
    road_df = road_df[[
        'timestamp',
        'id',
        'type',
        'sub_type',
        'x',
        'y',
        'z',
        'length',
        'width',
        'height',
        'theta',
        'v_x',
        'v_y'
    ]]
    
    road_df.type = road_df.type.map(lambda x: name2id[x])
    road_df.id, road_df.type = road_df.id.astype('int32'), road_df.type.astype('int32')
    t_min, t_max = min(road_df.timestamp), max(road_df.timestamp)

    new_df = []

    ids = road_df.id.unique()
    for id in ids[:]:
        id_df = road_df.loc[road_df.id == id]
        id_df_array = np.array(id_df)
        new_df.append(id_df_array)

    all_agent_type = []

    now_index2id = {}
    current_frame_dic = {}

    completed_current_index_dic = {}
    completed_current_frame_dic = {}
    now_obs_fea_dic = {}
    tmp_label_dic = {}
    tmp_auxiliary_label_dic = {}
    now_label_mask_dic = {}
    tar_id_dic = {}
    all_track_fea = {}
    all_obs_fea_dic = {}
    x_min, y_min = 0, 0
    x_max, y_max = 0, 0
    for now_track_index, now_track in enumerate(new_df):
        # 找出这个场景中x, y坐标的最小值
        a, b = now_track[:, 4].min(), now_track[:, 5].min()
        c, d = now_track[:, 4].max(), now_track[:, 5].max()
        if x_min == 0 and x_max == 0:
            x_min, x_max = a, c
        else:
            if a < x_min:
                x_min = a
            if c > x_max:
                x_max = c
        if y_min == 0 and y_max == 0:
            y_min, y_max = b, d
        else:
            if b < y_min:
                y_min = b
            if d > y_max:
                y_max = d
        now_mask = np.array([1] * (t_h + 1))
        now_index2id[now_track_index] = now_track[:, 1][0]

        now_fea = []
        for index in range(len(now_track)):
            now_track[index, 0] = int(np.round((now_track[index, 0] - t_min) / frame))
            now_fea.append(np.array([int(now_track[index, 0]), float(now_track[index, 4]), float(now_track[index, 5]), float(now_track[index, 6]), float(now_track[index, 11]),
                                     float(now_track[index, 12]), float(now_track[index, 10]), float(now_track[index, 7]), float(now_track[index, 8]), float(now_track[index, 9])]))
        now_fea = np.stack(now_fea, axis=0)
        all_track_fea[now_track_index] = now_fea
            # T, x, y, z, vx, vy, theta, length, width, height
        current_frame_dic[now_track_index] = np.array([[index, value] for index, value in enumerate(now_fea[:, 0]) if t_h - 1 < value < 100 - t_f])
            # 筛掉首尾数据缺失的车辆轨迹，输出完整轨迹的当前时刻index
        completed_current_index, completed_current_frame, now_obs_fea_lis, tmp_label_lis, tmp_auxiliary_label_lis, now_label_mask_lis, tar_id_lis, all_obs_fea_padded = screen_completed_track(current_frame_dic[now_track_index], now_fea, now_mask, now_index2id[now_track_index])
        completed_current_index_dic[now_track_index] = completed_current_index
        completed_current_frame_dic[now_track_index] = completed_current_frame
        now_obs_fea_dic[now_track_index] = now_obs_fea_lis
        tmp_label_dic[now_track_index] = tmp_label_lis
        tmp_auxiliary_label_dic[now_track_index] = tmp_auxiliary_label_lis
        now_label_mask_dic[now_track_index] = now_label_mask_lis
        tar_id_dic[now_track_index] = tar_id_lis
        all_obs_fea_dic[now_track_index] = all_obs_fea_padded
        all_agent_type.append(int(now_track[0, 2]))

    current_lis = list(chain.from_iterable(completed_current_frame_dic.values()))
    if len(np.unique(current_lis)) != 20:
        print("Error: ", iter, "has no target !!!")
        return
    pred_num = {}
    all_predict_agent_feature_dic = {}
    all_label_dic = {}
    all_auxiliary_label_dic = {}
    all_label_mask_dic = {}
    new_predict_index_dic = {}
    all_predict_agent_type_dic = {}
    all_object_id_dic = {}
    all_agent_type_dic = {}
    all_agent_feature_dic = {}

    #  AV_current_frame = []
    for frame_index in range(30, 50):
        pred_num[frame_index] = current_lis.count(frame_index)

        all_predict_agent_feature = []
        all_label = []
        all_auxiliary_label = []
        all_label_mask = []
        all_predict_agent_type = []
        new_predict_index = []
        all_object_id = []
        for track_index in range(len(all_track_fea)):
            if len(completed_current_index_dic[track_index]) != 0:    #if len(completed_current_index_dic[track_index]) != 0 and track_index not in target_index:
                completed_current_frame = completed_current_frame_dic[track_index]
                current_frame = np.int64(frame_index)
                if current_frame in completed_current_frame:
                    current_agent_index = np.where(completed_current_frame == current_frame)[0][0]
                    all_predict_agent_feature.append(now_obs_fea_dic[track_index][current_agent_index])
                    all_label.append(tmp_label_dic[track_index][current_agent_index])
                    all_auxiliary_label.append(tmp_auxiliary_label_dic[track_index][current_agent_index])
                    all_label_mask.append(now_label_mask_dic[track_index][current_agent_index])
                    new_predict_index.append(track_index)
                    all_predict_agent_type.append(all_agent_type[track_index])
                    all_object_id.append(tar_id_dic[track_index][current_agent_index])
        all_predict_agent_feature_dic[frame_index] = all_predict_agent_feature
        all_label_dic[frame_index] = all_label
        all_auxiliary_label_dic[frame_index] = all_auxiliary_label
        all_label_mask_dic[frame_index] = all_label_mask
        new_predict_index_dic[frame_index] = new_predict_index
        all_predict_agent_type_dic[frame_index] = all_predict_agent_type
        all_object_id_dic[frame_index] = all_object_id
        all_agent_type_dic[frame_index] = np.array(all_predict_agent_type_dic[frame_index])  # np.array(all_predict_agent_type_dic[frame_index] + all_other_agent_type_dic[frame_index])
        all_agent_feature_dic[frame_index] = np.stack(all_predict_agent_feature_dic[frame_index], axis=0)

    ##处理交通信号灯信息
    light_df.state = light_df.color_1.map(lambda x: color2num[x])
    # light_data = np.concatenate([np.array(light_df.timestamp), np.array(light_df.x), np.array(light_df.y), np.array(light_df.lane_id),
    #                             np.array(light_df.color_1), np.array(light_df.remain_1), np.array(light_df.color_2), np.array(light_df.remain_2), np.array(light_df.color_3), np.array(light_df.remain_3)], axis=1)
    light_data = np.concatenate([np.array(light_df.timestamp)[:, np.newaxis], np.array(light_df.x)[:, np.newaxis], np.array(light_df.y)[:, np.newaxis], np.array(light_df.lane_id)[:, np.newaxis], np.array(light_df.state)[:, np.newaxis]], axis=1)
    for index in range(len(light_data)):
        light_data[index, 0] = int(np.round((light_data[index, 0] - t_min) / frame))
    all_dynamic_map_fea_dic = {}
    dynamic_map_key = []
    for light_index, light_dataleaf in enumerate(light_data):  # bug: 83155210_1_-2
        if light_dataleaf[3] not in list(map_data['lane_id_to_numid'].keys()):
            continue
        if [light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]]] not in dynamic_map_key:
            dynamic_map_key.append([light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]]])
            all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])] = {}
            for current_frame in range(30, 50):
                all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])][int(current_frame)] = [3] * (t_h + 1)  # 没有信息的置为3，算是一种掩码
        for current_frame in range(30, 50):
            if current_frame - t_h <= light_dataleaf[0] <= current_frame:
                all_dynamic_map_fea_dic[(light_dataleaf[1], light_dataleaf[2], map_data['lane_id_to_numid'][light_dataleaf[3]])][int(current_frame)][light_dataleaf[0] - (int(current_frame) - t_h)] = light_dataleaf[4]
    # 检查每一帧是否都有交通灯信息
    for light_id, light_fea in all_dynamic_map_fea_dic.items():
        for current_frame, light_value in light_fea.items():
            if [3] in light_value:
                for index, value in enumerate(light_value):
                    if value == [3]:
                        all_dynamic_map_fea_dic[light_id][current_frame][index] = \
                        all_dynamic_map_fea_dic[light_id][current_frame][index - 1]
    #    all_dynamic_map_fea_dic = {k: v[::track_ds] for k, v in all_dynamic_map_fea_dic.items()}  # 对动态地图信息下采样
    all_dynamic_map_fea = {
        int(k[2]): {current_frame: np.array([k[0], k[1]] + flag) for current_frame, flag in v.items()} for k, v in
        all_dynamic_map_fea_dic.items()}

    for old_lane_id in all_dynamic_map_fea:
        new_lane_id_lis = old_lane_id_to_new_lane_index_lis[old_lane_id]
        for _ in new_lane_id_lis:
            if _ < lane_num:
                lane_fea[_]["signal"].append(all_dynamic_map_fea[old_lane_id])

    all_agent_map_size_dic = {k: np.ones(v.shape[0]) * 600.0 for k, v in all_agent_feature_dic.items()}  # todo:999.0可根据需求更改
    final_lane_fea_dic = {}
    final_polygon_fea_dic = {}
    lane_new_index_to_final_index_dic = {}
    polygon_new_index_to_final_index_dic = {}
    for frame_index in range(30, 50):
        lane_new_index_to_final_index_dic[frame_index] = {}
        if len(lane_fea) > 0:
            new_dist_between_agent_lane = (euclid(all_agent_feature_dic[frame_index][:, -1, [1, 2]][:, np.newaxis, np.newaxis, :], np.stack([_["xy"] for _ in lane_fea])[np.newaxis, :, :, :]).min(2) < all_agent_map_size_dic[frame_index][:, np.newaxis])
            nearby_lane_new_index_lis = np.where(new_dist_between_agent_lane)[1].tolist()  
            nearby_lane_new_index_lis = np.unique(np.array(nearby_lane_new_index_lis))
            lane_new_index_to_final_index_dic[frame_index] = {index_i: i for i, index_i in enumerate(nearby_lane_new_index_lis)}  # lane只要和场景中的任一一辆有效车的距离在限制范围内，那这个lane就保留下来；新的index是与agent_id有关系的
        final_lane_fea_dic[frame_index] = [{} for _ in range(len(lane_new_index_to_final_index_dic[frame_index]))]
        for lane_new_index in lane_new_index_to_final_index_dic[frame_index]:
            i = lane_new_index_to_final_index_dic[frame_index][lane_new_index]
            for transfer_key in ["xy", "type", "signal", "yaw"]:
                if transfer_key == "signal":
                    #                    if len(lane_fea[lane_new_index][transfer_key]) != 0: print(lane_fea[lane_new_index][transfer_key])
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
            new_dist_between_agent_polygon = (euclid(all_agent_feature_dic[frame_index][:, -1, [1, 2]][:, np.newaxis, np.newaxis, :], np.stack([_[1] for _ in polygon_fea], axis=0)[np.newaxis, :, :, :]).min(2) < all_agent_map_size_dic[frame_index][:, np.newaxis])
            nearby_polygon_lis = np.where(new_dist_between_agent_polygon)[1].tolist()
            nearby_polygon_lis = np.unique(np.array(nearby_polygon_lis))
            polygon_new_index_to_final_index_dic[frame_index] = {index_i: i for i, index_i in enumerate(nearby_polygon_lis)}
        final_polygon_fea_dic[frame_index] = [[] for _ in range(len(polygon_new_index_to_final_index_dic[frame_index]))]
        for polygon_new_index in polygon_new_index_to_final_index_dic[frame_index]:
            final_polygon_fea_dic[frame_index][polygon_new_index_to_final_index_dic[frame_index][polygon_new_index]] = polygon_fea[polygon_new_index]

    all_data = {}
    all_data["fname"] = iter
    all_data["agent_feature"] = all_agent_feature_dic  # 所有车辆的历史轨迹
    all_data["label"] = {k: np.concatenate(v, axis=0) for k, v in all_label_dic.items()}  # 所有预测目标的真实未来轨迹x,y
    all_data["auxiliary_label"] = {k: np.concatenate(v, axis=0) for k, v in
                                   all_auxiliary_label_dic.items()}  # 所有预测目标的真实未来速度v，加速度a，方向heading等
    all_data["label_mask"] = {k: np.stack(v, axis=0) for k, v in all_label_mask_dic.items()}  # 所有预测目标关于未来50帧轨迹是否有效的掩码

    all_data["pred_num"] = pred_num  # 预测目标的数量
    all_data["object_id_lis"] = {k: np.array(v) for k, v in all_object_id_dic.items()}
    all_data["agent_type"] = all_agent_type_dic
    all_data["map_fea"] = [final_lane_fea_dic, final_polygon_fea_dic]

    if processed_count.value % 100 == 0:
        print(args.split, processed_count.value, "done", flush=True)
    with processed_count.get_lock():
        processed_count.value += 1

    return all_data, {frame_index: all_agent_feature_dic[frame_index].shape[0] + len(final_lane_fea_dic[frame_index]) + len(final_polygon_fea_dic[frame_index]) for frame_index in range(30, 50)}, all_data["fname"]



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='process road v2x-seq to prediction')
    parser.add_argument("--data_root", type=str, default="/home/zhangxy/Co-MTP/dataset/V2X-Seq-TFD/")      
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

    data_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/infrastructure-trajectories', args.split)
    traffic_light_path = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/traffic-light/', args.split)
    data_dest = os.path.join(args.data_root, 'cooperative-vehicle-infrastructure/process_road_for_v2x', args.split, 'data')  #存储数据处理的地址  # 'data'  #路端存储数据的地址：process_for_prediction_road

    data_ls = os.listdir(data_path)

    if not os.path.exists(data_dest):
        os.makedirs(data_dest)

    with open('/media/ps/ba50700a-668e-4255-9ec6-a877cfa97e41/zxy/V2X-Seq-TFD/map_files/yizhuang_PEK_vector_map.json', 'r') as file:    
        map_data = json.load(file)
    new_lane_fea, old_lane_id_to_new_lane_index_lis, all_polygon_fea = maps_process(map_data)
    lane_num = len(new_lane_fea)
    print("Static Map Done!!! ")

    num_of_element_lis = []
    tasks = [(data, i) for i, data in enumerate(data_ls)]
    epo = 0
    save_count = 0
    data_num = len(tasks)
    
    ##train&val时用这个
 #    with multiprocessing.Pool(processes=8) as pool:
 #        while 500 * epo < data_num:
 #            start_index = 500 * epo
 #            end_index = min((epo + 1) * 500, data_num)
 #            result = pool.starmap(process, tasks[start_index:end_index])  # 使用 starmap 来传递多个参数
 #            result = list(filter(lambda x: x is not None, result))
 #            for num in range(len(result)):
 #                with open(os.path.join(data_dest, str(save_count + num) + ".pkl"), "wb") as g:
 #                    pickle.dump(result[num][0], g)
 #            num_of_element_lis.append([e[1] for e in result])
 #            print("保存成功！")
 #            save_count = save_count + len(result)
 #            epo = epo + 1
 #            del result
 #            gc.collect()
 #    pool.close()
 #    pool.join()
 #
 #    print("Preprocess Done")
 #
 #    num_of_element_lis = list(itertools.chain(*num_of_element_lis))
 #    with open(os.path.join(data_dest, "number_of_dataset.pkl"), "wb") as g:
 #        #pickle.dump(np.array(np.stack(num_of_element_lis)), g)
 #        pickle.dump(np.array(np.stack(num_of_element_lis)), g)
 # #       g.write(json_num.encode())
 #
 #    print("Dataset Processing Done!!!")

    #test时用这个
    with multiprocessing.Pool(processes=8) as pool:
        while 500 * epo < data_num:
            start_index = 500 * epo
            end_index = min((epo + 1) * 500, data_num)
            result = pool.starmap(process, tasks[start_index:end_index])  # 使用 starmap 来传递多个参数
            result = list(filter(lambda x: x is not None, result))
            for num in range(len(result)):
                with open(os.path.join(data_dest, result[num][2].split('.')[0] + ".pkl"), "wb") as g:
                    pickle.dump(result[num][0], g)
            num_of_element_lis.append(list(map(itemgetter(1), result)))
            print("保存成功！")
            save_count = save_count + len(result)
            epo = epo + 1
            del result
            gc.collect()
    pool.close()
    pool.join()

    print("Dataset Processing Done!!!")
