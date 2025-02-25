import numpy as np
import math
import scipy.interpolate as interp

def interpolate_polyline(polyline, num_points):
    if np.allclose(polyline[0], polyline[1]):
        return polyline[0][np.newaxis, :].repeat(num_points, axis=0)
    tck, u = interp.splprep(polyline.T, s=0, k=1)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def maps_process(map_data):
    lane = map_data['lane']
    crosswalk = map_data['crosswalk']
    stop_line = map_data['stop_line']

    all_polygon_fea = []
    all_lane_fea = {}
    lane_numid = 0
    lane_types = []

    for crosswalk_id, crosswalk_fea in crosswalk.items():
        now_polygon_fea = [0, np.array([[_[0], _[1]] for _ in crosswalk_fea['polygon']])]
        all_polygon_fea.append(now_polygon_fea)

    # for stop_line_id, stop_line_fea in stop_line.items():
    #     now_polygon_fea = [1, np.array([[_[0], _[1]] for _ in stop_line_fea['centerline']])]
    #     now_polygon_fea[1] = np.concatenate([now_polygon_fea[1], now_polygon_fea[1][0]], axis=0)
    #     all_polygon_fea.append(now_polygon_fea)

    for lane_id, lane_fea in lane.items():
        all_lane_fea[int(lane_id)] = {}
        all_lane_fea[int(lane_id)]["type"] = lane_fea['type']  #3types
        all_lane_fea[int(lane_id)]["xy"] = np.stack(lane_fea['centerline'], axis=0)
        all_lane_fea[int(lane_id)]["entry"] = lane_fea['predecessors']
        all_lane_fea[int(lane_id)]["exit"] = lane_fea['successors']
        ##ID -> Neighbor ID, self_start, self_end, neighbor_start, neighbor_end
        all_lane_fea[int(lane_id)]["left"] = [[lane_fea['left_id']] + [0] + [len(lane_fea['centerline'])-1] + [0] + [len(lane_fea['left_boundary'])-1]] if lane_fea['left_id'] != [] else []
        all_lane_fea[int(lane_id)]["right"] = [[lane_fea['right_id']] + [0] + [len(lane_fea['centerline']) - 1] + [0] + [len(lane_fea['right_boundary']) - 1]] if lane_fea['right_id'] != [] else []
        all_polygon_fea.append([1, np.stack(lane_fea['left_boundary'], axis=0)])
        all_polygon_fea.append([1, np.stack(lane_fea['right_boundary'], axis=0)])

    print("Conversion Completed!!!")
#    return all_lane_fea, all_polygon_fea
    length_per_polyline = 40.0  # 20 meters
    point_per_polyline = 21
    space = int(length_per_polyline // (point_per_polyline - 1))

    new_lane_fea = []
    old_lane_id_to_new_lane_index_lis = {}

    for old_lane_id, old_lane_info in all_lane_fea.items():
        if old_lane_info["xy"].shape[0] <= length_per_polyline:
            old_lane_id_to_new_lane_index_lis[old_lane_id] = [len(new_lane_fea)]
            new_lane_xy = old_lane_info["xy"]
            if new_lane_xy.shape[0] > 1:
                new_lane_xy = interpolate_polyline(new_lane_xy, point_per_polyline)
            else:
                new_lane_xy = np.broadcast_to(new_lane_xy, (point_per_polyline, 3))
            new_lane_fea.append({"xy": new_lane_xy, "type": old_lane_info["type"], "left": [], "right": [], "prev": [], "follow": [], "signal": []})
        else:
            num_of_new_lane = math.ceil(old_lane_info["xy"].shape[0] / length_per_polyline)
            now_lanelet_new_index_lis = list(range(len(new_lane_fea), len(new_lane_fea) + num_of_new_lane))
            old_lane_id_to_new_lane_index_lis[old_lane_id] = now_lanelet_new_index_lis
            new_lane_xy = []
            for _ in range(num_of_new_lane - 1):
                tmp = old_lane_info["xy"][int(_ * length_per_polyline):int(_ * length_per_polyline + length_per_polyline + 1)]
                new_lane_xy.append(tmp[::space, :])
            tmp = old_lane_info["xy"][int((num_of_new_lane - 1) * length_per_polyline):]
            if tmp.shape[0] == 1:
                tmp = np.concatenate(
                    [old_lane_info["xy"][int((num_of_new_lane - 1) * length_per_polyline - 1)][np.newaxis, :], tmp], axis=0)
            new_lane_xy.append(interpolate_polyline(tmp, point_per_polyline))
            # tmp = tmp[::2, :]
            for _ in range(len(new_lane_xy)):
                new_lane_fea.append({"xy": new_lane_xy[_], "type": old_lane_info["type"], "left": [], "right": [], "prev": [], "follow": [], "signal": []})

    ## Update relations
    for old_lane_id, new_lane_lis in old_lane_id_to_new_lane_index_lis.items():
        if len(new_lane_lis) > 0:
            for j in range(1, len(new_lane_lis)):
                prev_index = new_lane_lis[j - 1]
                next_index = new_lane_lis[j]
                new_lane_fea[prev_index]["follow"].append([next_index, 0])
                new_lane_fea[next_index]["prev"].append([prev_index, 1])
        ## Follow
        tmp_index = new_lane_lis[-1]
        for old_adj_index in all_lane_fea[old_lane_id]["exit"]:
            new_lane_fea[tmp_index]["follow"].append([old_lane_id_to_new_lane_index_lis[old_adj_index][0], 0] if old_adj_index in old_lane_id_to_new_lane_index_lis.keys() else [])

        ## Prev
        tmp_index = new_lane_lis[0]
        for old_adj_index in all_lane_fea[old_lane_id]["entry"]:
            new_lane_fea[tmp_index]["prev"].append([old_lane_id_to_new_lane_index_lis[old_adj_index][-1], 1] if old_adj_index in old_lane_id_to_new_lane_index_lis.keys() else [])

        ## Left Right
        for edge_type in ["left", "right"]:
            old_adj_info_lis = all_lane_fea[old_lane_id][edge_type]
            ## ID, self_start, end, neighbor_start, end, type
            for old_adj_info in old_adj_info_lis:
                if old_adj_info[0] not in old_lane_id_to_new_lane_index_lis.keys():
                    continue
                can_turn_new_lane_lis = new_lane_lis[int(old_adj_info[1] // length_per_polyline):int(old_adj_info[2] // length_per_polyline + 1)]
                can_turn_new_adj_lane_lis = old_lane_id_to_new_lane_index_lis[old_adj_info[0]][int(old_adj_info[3] // length_per_polyline):int(old_adj_info[4] // length_per_polyline + 1)]
                l1 = len(can_turn_new_lane_lis)
                l2 = len(can_turn_new_adj_lane_lis)
                boundary_type = 1  #default: old_adj_info[5]  todo:把所有的道路边类型都设为1
                if l1 == l2:
                    for tmp_index_i in range(l1):
                        tmp_index = can_turn_new_lane_lis[tmp_index_i]
                        new_lane_fea[tmp_index][edge_type].append([can_turn_new_adj_lane_lis[tmp_index_i], boundary_type + 2])  # 为什么+2？是因为“following”和"prev"占了0和1这两个标志吗？
                elif l1 < l2:
                    ratio = int(math.ceil(float(l2) / float(l1)))
                    for tmp_index_i in range(l1):
                        tmp_index = can_turn_new_lane_lis[tmp_index_i]
                        ratio_index = 0
                        gap = ratio - 1
                        if l2 % l1 == 0:
                            gap += 1
                        while ratio_index < ratio and ratio_index + tmp_index_i * gap < l2:
                            new_lane_fea[tmp_index][edge_type].append([can_turn_new_adj_lane_lis[int(ratio_index + tmp_index_i * gap)], boundary_type + 2])  # 11.19：没看明白，明天继续（小喵，Fighting!） 11.20：看懂啦，小喵，你好棒！
                            ratio_index += 1
                elif l1 > l2:
                    ratio = int(math.ceil(float(l1) / float(l2)))
                    for adj_index_i in range(l2):
                        tmp_adj_index = can_turn_new_adj_lane_lis[adj_index_i]
                        ratio_index = 0
                        gap = ratio - 1
                        if l1 % l2 == 0:
                            gap += 1
                        while ratio_index < ratio and ratio_index + adj_index_i * gap < l1:
                            tmp_index = can_turn_new_lane_lis[ratio_index + adj_index_i * gap]
                            new_lane_fea[tmp_index][edge_type].append([tmp_adj_index, boundary_type + 2])  # 这样操作的话，new_lane_fea里有的lane就有可能没有与之相对应的"left"or"right"lane编号（头上多了三个问号？？？）
                            ratio_index += 1
    for _ in range(len(new_lane_fea)):
        new_lane_fea[_]["yaw"] = np.arctan2(new_lane_fea[_]["xy"][-1, 1] - new_lane_fea[_]["xy"][0, 1], new_lane_fea[_]["xy"][-1, 0] - new_lane_fea[_]["xy"][0, 0])

    ##Split and Regularize Polygon fea
    ##20m per piece, 20 point
    new_polygon_fea = []
    for polygon_index in range(len(all_polygon_fea)):
        if all_polygon_fea[polygon_index][0] != 0:
            if len(all_polygon_fea[polygon_index][1]) > length_per_polyline:
                num_of_piece = int(len(all_polygon_fea[polygon_index][1]) // length_per_polyline + 1)
                length_per_piece = len(all_polygon_fea[polygon_index][1]) // num_of_piece + 1
                for _ in range(num_of_piece):
                    polygon_coor_of_current_piece = all_polygon_fea[polygon_index][1][int(_ * length_per_piece):int((_ + 1) * length_per_piece)]
                    if polygon_coor_of_current_piece.shape[0] > 1:
                        new_polygon_fea.append([all_polygon_fea[polygon_index][0], polygon_coor_of_current_piece])
            else:
                if all_polygon_fea[polygon_index][1].shape[0] > 1:
                    new_polygon_fea.append(all_polygon_fea[polygon_index])
        else:
            new_polygon_fea.append([all_polygon_fea[polygon_index][0], np.concatenate([all_polygon_fea[polygon_index][1], all_polygon_fea[polygon_index][1][0, :][np.newaxis, :]], axis=0)])

    all_polygon_fea = [[_[0], interpolate_polyline(_[1], point_per_polyline)] for _ in new_polygon_fea]

    return new_lane_fea, old_lane_id_to_new_lane_index_lis, all_polygon_fea



