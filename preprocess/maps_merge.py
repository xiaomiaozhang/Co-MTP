import os
import json
import pandas as pd
import numpy as np
#from tqdm import tqdm
import argparse

#from argoverse.utils.centerline_utils import centerline_to_polygon

def merge_map_json(maps):

    lane_dict = {}
    lane_id_to_numid = {}
    crosswalk_dict = {}
    stop_line_dict = {}
    junction_dict = {}

    id_num = 0

#    for map in tqdm(maps, desc='merge_map_json'):
    for map in maps:
        
        with open(os.path.join(map_dir, map), encoding='utf-8') as f:
            map_data = json.load(f)
        lane = map_data['LANE']
        stop_line = map_data['STOPLINE']
        crosswalk = map_data['CROSSWALK']
        junction = map_data['JUNCTION']
        for lane_id, lane_fea in lane.items():
            lane_data = {}
            centerline, left_boundary, right_boundary, type, left_id, right_id, predecessors, successors = lane_fea['centerline'], lane_fea['left_boundary'], lane_fea['right_boundary'], lane_fea['lane_type'], lane_fea['l_neighbor_id'], lane_fea['r_neighbor_id'], lane_fea['predecessors'], lane_fea['successors']
            # 处理中心线数据
            pt = [point.strip("()").split(", ") for point in centerline]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            lane_data['centerline'] = xy
            #处理道路左边线数据
            pt = [point.strip("()").split(", ") for point in left_boundary]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            lane_data['left_boundary'] = xy
            #处理道路右边线数据
            pt = [point.strip("()").split(", ") for point in right_boundary]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            lane_data['right_boundary'] = xy
            #处理道路类型   'CITY_DRIVING'：0；'BIKING'：1；'LEFT_TURN_WAITING_ZONE'：2
            if type == 'CITY_DRIVING':
                type = 0
            elif type == 'BIKING':
                type = 1
            else:
                type = 2
            #处理道路id的转换
            for id in [lane_id] + [left_id] + [right_id] + [_ for _ in predecessors] + [_ for _ in successors]:
                if id != 'None' and id not in lane_id_to_numid:
                    lane_id_to_numid[id] = id_num
                    id_num = id_num + 1
            lane_data['type'], lane_data['left_id'], lane_data['right_id'], lane_data['predecessors'], lane_data['successors'] = type, [] if left_id =='None' else lane_id_to_numid[left_id], [] if right_id =='None' else lane_id_to_numid[right_id], [] if predecessors =='None' else [lane_id_to_numid[_] for _ in predecessors], [] if successors =='None' else [lane_id_to_numid[_] for _ in successors]
                
            lane_dict[lane_id_to_numid[lane_id]] = lane_data

        for crosswalk_id, crosswalk_fea in crosswalk.items():
            crosswalk_data = {}
            polygon = crosswalk_fea['polygon']
            pt = [point.strip("()").split(", ") for point in polygon]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            crosswalk_data['polygon'] = xy

            crosswalk_dict[crosswalk_id] = crosswalk_data

        for stop_line_id, stop_line_fea in stop_line.items():
            stop_line_data = {}
            centerline = stop_line_fea['centerline']
            pt = [point.strip("()").split(", ") for point in centerline]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            stop_line_data['centerline'] = xy

            stop_line_dict[stop_line_id] = stop_line_data

        for junction_id, junction_fea in junction.items():
            junction_data = {}
            polygon = junction_fea['polygon']
            pt = [point.strip("()").split(", ") for point in polygon]
            x = np.array([float(point[0]) for point in pt])
            y = np.array([float(point[1]) for point in pt])
            xy = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1).tolist()
            junction_data['polygon'] = xy

            junction_dict[junction_id] = junction_data

    map_list = {'lane': lane_dict, 'lane_id_to_numid': lane_id_to_numid, 'crosswalk': crosswalk_dict, 'stop_line': stop_line_dict, 'junction': junction_dict}
    with open(os.path.join(output_dir, 'yizhuang_PEK_vector_map.json'),"w") as f:
        json.dump(map_list, f)
    print('Completed!!!')


# def merge_halluc(maps):
#
#     idx = 0
#     map_dict = {}
#     polygon_ls = []
#
#     lane_dict = {}
#     for map in maps:
#         with open(os.path.join(map_dir, map), encoding='utf-8') as f :
#             for lane_id, lane in json.load(f)['LANE'].items():
#                 lane_dict[lane_id] = lane
#
#     for lane_id, lane in tqdm(lane_dict.items(), desc='merge_halluc'):
#         map_dict[str(idx)] = lane_id
#
#         lane_ls = []
#
#         for pt in lane['centerline']:
#             xy = []
#             xy.append(float(pt[1:14]))
#             xy.append(float(pt[-15:-1]))
#             lane_ls.append(xy)
#
#         polygon = centerline_to_polygon(np.array(lane_ls))
#
#         x_min, y_min = 500000, 5000000
#         x_max, y_max = 0, 0
#
#         for iter in polygon:
#
#             x_min = min(x_min, iter[0])
#             x_max = max(x_max, iter[0])
#             y_min = min(y_min, iter[1])
#             y_max = max(y_max, iter[1])
#
#         polygon_xy = [x_min, y_min, x_max, y_max]
#         polygon_ls.append(polygon_xy)
#
#         idx += 1
#
#     polygon_ls = np.array(polygon_ls)
#
#     with open(os.path.join(output_dir, 'yizhuang_PEK_tableidx_to_laneid_map.json'),"w") as f:
#         json.dump(map_dict,f)
#
#     np.save(os.path.join(output_dir, 'yizhuang_PEK_halluc_bbox_table.npy'), polygon_ls)

def merge_multiple_maps():

    maps = os.listdir(map_dir)

    merge_map_json(maps)

#    merge_halluc(maps)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='merge maps')
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/project/HDGT-main/dataset/V2X-Seq-TFD-Example/")

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    print(args)

    map_dir = os.path.join(args.data_root, 'maps')
    output_dir = os.path.join(args.data_root, 'map_files')
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    merge_multiple_maps()