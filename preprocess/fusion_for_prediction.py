from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os
import numpy as np
import copy
import pandas as pd
import multiprocessing
import logging
logger = logging.getLogger(__name__)

from fusion import Fusion
from utils import cal_matched_ids
from config import name2id, id2name, subname2id, id2subname, tag2id

class PredictionFusion(Fusion):

    def __init__(self,iou_threshold = 0.3, hungarian=True, time_step = 100, ego_offset = 6.0, source_flag = 'baidu',solve_wrong_association = True,dist_flag = 'iou_2d'):
        super().__init__(iou_threshold, 0, hungarian, dist_flag,solve_wrong_association)

        self.source_flag = source_flag
        time_step = 100
        if self.source_flag == 'tfd':
            time_step = 0.1
        self.time_step = time_step
        
        self.veh2inf_frame_id = {}

        self.ego_offset = ego_offset
        self.id_log = {'set1': np.zeros(shape=(0, 3),dtype=np.int64) - 1, 'set2': np.zeros(shape=(0, 3),dtype=np.int64) - 1}

    def remove_cannot_fusion_ids(self,tracks1,tracks2,v_ind, r_ind,cannot_fusion_v2i):  
        new_v_ind = np.empty(shape=(0,1),dtype=np.int64)
        new_r_ind = np.empty(shape=(0,1),dtype=np.int64)

        v_ind_len = len(v_ind)
        r_ind_len = len(r_ind)
        if not v_ind_len == r_ind_len:
            return new_v_ind,new_r_ind
        
        for i in range(v_ind_len):
            tracklet1 = tracks1[v_ind[i]]
            tracklet2 = tracks2[r_ind[i]]
            if not (tracklet1[2],tracklet2[2]) in cannot_fusion_v2i:
                new_v_ind = np.vstack([new_v_ind,[v_ind[i]]])
                new_r_ind = np.vstack([new_r_ind,[r_ind[i]]])
            # else:
            #     print('cannot fusion: v_id: %s,  r_id: %s'%(tracklet1[2],tracklet2[2]))
        

        return new_v_ind,new_r_ind


    def fuse_tracks(self, tracks1,tracks2,cannot_fusion_v2i):
        #cal matched_ids,tracks1_unmatched_boxes_index,tracks2_unmatched_boxes_index
        v_ind, r_ind = cal_matched_ids(tracks1,tracks2,self.iou_threshold,self.hungarian)
        v_ind, r_ind = self.remove_cannot_fusion_ids(tracks1,tracks2,v_ind, r_ind,cannot_fusion_v2i)

        cur_frame_track_id = set()

        # initialize joined output
        # go through all car side tracklets
        for track1_index in range(len(tracks1)):
            tracklet1 = tracks1[track1_index]
            tracklet = copy.deepcopy(tracklet1)

            # MATCHED TRACKLETS:
            if track1_index in v_ind:
                tracklet2 = tracks2[r_ind[v_ind == track1_index]][0]

                #road side points
                self.new_road_outputs.append(tracklet2)

                tracklet = np.concatenate([tracklet,[1, tracklet1[2], tracklet2[2]]]) # from: 1:car side, 2:road side; car side id; road side id

                # update tracking id ***
                # check, whether tracking ids are already inside the id log structure
                log_flag1 = tracklet1[2] in self.id_log['set1'][:, 0]
                log_flag2 = tracklet2[2] in self.id_log['set2'][:, 0]

                # copy id log values if already contained in id_log
                if log_flag1:
                    log_val1 = copy.deepcopy(self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet1[2]])[0]
                if log_flag2:
                    log_val2 = copy.deepcopy(self.id_log['set2'][self.id_log['set2'][:, 0] == tracklet2[2]])[0]

                # if both tracking ids are new, crete a new common tracking id
                if not log_flag1 and not log_flag2:
                    self.id_log['set1'] = np.vstack([self.id_log['set1'], [tracklet1[2], self.id_counter, tracklet1[0]]])
                    self.id_log['set2'] = np.vstack([self.id_log['set2'], [tracklet2[2], self.id_counter, tracklet1[0]]])
                    self.id_counter += 1

                # if ct tracking id is unknown, copy the tracking id from cp
                elif log_flag1 and not log_flag2:
                    self.id_log['set2'] = np.vstack(
                        [self.id_log['set2'], [tracklet2[2], log_val1[1], log_val1[2]]])

                # if cp tracking id is unknown, copy the tracking id from ct
                elif not log_flag1 and log_flag2:
                    self.id_log['set1'] = np.vstack(
                        [self.id_log['set1'], [tracklet1[2], log_val2[1], log_val2[2]]])

                # if both tracking ids are already known (used before), ...
                elif log_flag1 and log_flag2:
                    if log_val1[1] != log_val2[1]:  # ... and if not equal ...
                        if log_val1[2] <= log_val2[2]:  # ... take the older tracking id (overwrite the younger one)
                            self.id_log['set2'][np.where(self.id_log['set2'][:, 0] == tracklet2[2])[0][0]][1] = log_val1[1]
                            self.id_log['set2'][np.where(self.id_log['set2'][:, 0] == tracklet2[2])[0][0]][2] = log_val1[2]

                            self.logger.info('line 215 error need debug!!!! ')  
                        else:
                            self.id_log['set1'][np.where(self.id_log['set1'][:, 0] == tracklet1[2])[0][0]][1] = log_val2[1]
                            self.id_log['set1'][np.where(self.id_log['set1'][:, 0] == tracklet1[2])[0][0]][2] = log_val2[2]

                            self.logger.info('line 219 error need debug!!!! ')

                # save the (new) common tracking id
                tracklet[2] = self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet[2]][0][1]

                if tracklet[2] not in cur_frame_track_id:
                    cur_frame_track_id.add(tracklet[2])
                    # add current tracklet to the tracklets list
                    self.new_tracks_fusion.append(tracklet)
            else:
                # if tracking id is not known yet, give new id (else: don't change its id)
                if tracklet[2] not in self.id_log['set1'][:, 0]:
                    self.id_log['set1'] = np.vstack([self.id_log['set1'], [tracklet[2], self.id_counter, tracklet[0]]])
                    self.id_counter += 1

                # save the (new) tracking id
                tracklet[2] = self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet[2]][0][1]

                tracklet = np.concatenate([tracklet,np.array([1, tracklet1[2], -1])]) # from: 1:car side, 2:road side; car side id; road side id

                if tracklet[2] not in cur_frame_track_id:
                    cur_frame_track_id.add(tracklet[2])
                    # add current tracklet to the tracklets list
                    self.new_tracks_fusion.append(tracklet)

        # go through all road side tracklets
        for track2_index in range(len(tracks2)):        
            # UNMATCHED TRACKLETS of second modality
            if track2_index not in r_ind:    # (simply ignore all matches, since we have them already)
                #road side output according car side frame rate
                tracklet = copy.deepcopy(tracks2[track2_index])

                track2_id = tracklet[2]

                fusion_flag = True
                if track2_id in self.id_log['set2'][:, 0]:
                    fusion_track_id = self.id_log['set2'][self.id_log['set2'][:, 0] == track2_id][0][1]
                    if fusion_track_id in cur_frame_track_id:
                        fusion_flag = False
                else:
                    self.id_log['set2'] = np.vstack([self.id_log['set2'], [tracklet[2], self.id_counter, tracklet[0]]])
                    fusion_track_id = self.id_counter
                    self.id_counter += 1
                    # fusion_flag = False  #error!!!!!!!!!!!!!!!!!!!!!!!
                
                if fusion_flag:                                  
                    tracklet[2] = fusion_track_id
                    tracklet = np.concatenate([tracklet,[2, -1, track2_id]]) # from: 1:car side, 2:road side; car side id; road side id

                    # add current tracklet to the tracklets list
                    cur_frame_track_id.add(tracklet[2])
                    self.new_tracks_fusion.append(tracklet)  
                    self.new_road_outputs.append(tracks2[track2_index])   #self.new_road_outputs：可融合的tracks2+不可融合的tracks2
                else:
                    self.new_road_outputs.append(tracks2[track2_index])
            
        return self.new_tracks_fusion,self.new_road_outputs   #self.new_tracks_fusion：车端不可融合轨迹+可融合部分直接采用车端轨迹+路端不可融合轨迹

    def id_cvt_name(self,ids,sub_flag=False):
        id_name = []
        for id in ids:
            if sub_flag:
                id_name.append(id2subname[id])
            else:
                id_name.append(id2name[id])

        return id_name 

    def find_ego_vehicle(self, host_car_pose,tracks):
        #find host car box and type = 5 ("EGO_VEHICLE")
        tracks_times = np.sort(np.unique(tracks[:,0]))

        for cur_time in tracks_times:
            cur_tracks = tracks[tracks[:,0] == cur_time]

            min_dist_index = -1
            min_dist = 1e18
            cur_index = 0
            for cur_track in cur_tracks:
                dist = np.sqrt(np.sum(np.square(host_car_pose - np.array([cur_track[13],cur_track[14]],dtype=float))))
                if dist < min_dist:
                    min_dist = dist
                    min_dist_index = cur_index               
                cur_index += 1

            # print("min_dist: ",min_dist)
            
            if min_dist <= self.ego_offset:
                #road side host car pose by car side
                for i in range(len(cur_tracks)):
                    tracks[np.where(tracks[:, 0] == cur_time)[0][i]][24] = host_car_pose[0]    #default:host_car_pose[0]
                    tracks[np.where(tracks[:, 0] == cur_time)[0][i]][25] = host_car_pose[1]    #default:host_car_pose[1]

                #this is ego
                tracks[np.where(tracks[:, 0] == cur_time)[0][min_dist_index]][1] = 5  

    def cvt_format_prediction2tracking(self,tracks1):
        rows,cols = tracks1.shape
        #0:frame,1:type,2:id,3:
        new_tracks1 = np.zeros(([rows,cols]),dtype=float)
        new_tracks1[:,0:3] = tracks1[:,0:3]
        new_tracks1[:,3] = tracks1[:,22]
        new_tracks1[:,4] = tracks1[:,23]
        new_tracks1[:,5] = tracks1[:,26]
        new_tracks1[:,6:10] = tracks1[:,3:7]

        new_tracks1[:,10:17] = tracks1[:,8:15]
        new_tracks1[:,17:21] = tracks1[:,16:20]
        new_tracks1[:,21] = tracks1[:,21]
        new_tracks1[:,22] = tracks1[:,7]
        new_tracks1[:,23] = -1  
        new_tracks1[:,24:26] = tracks1[:,24:26]
        new_tracks1[:,27] = tracks1[:,27]
        return new_tracks1

    def gen_data(self,tracks1_data,tracks2_data):
        self.veh2inf_frame_id = {}
        tracks1_data_new,tracks2_data_new = [],[]

        if len(tracks1_data) > 0:
            tracks1_times = np.sort(np.unique(tracks1_data[:,0]))
            tracks1_data_new = np.zeros(([0,tracks1_data.shape[1]]),dtype=float)
        
        if len(tracks2_data) > 0:
            tracks2_times = np.sort(np.unique(tracks2_data[:,0]))
            tracks2_data_new = np.zeros(([0,tracks2_data.shape[1]]),dtype=float)
        
        if len(tracks1_data) > 0 and len(tracks2_data) > 0:
            if self.source_flag == 'baidu':
                sec_start = min(int(tracks1_times[0]*1e-2)*1e2, int(tracks2_times[0]*1e-2)*1e2)
                sec_end = max(int(tracks1_times[-1]*1e-2)*1e2, int(tracks2_times[-1]*1e-2)*1e2)
            elif self.source_flag == 'tfd':
                sec_start = min(tracks1_times[0],tracks2_times[0])
                sec_end = max(tracks1_times[-1],tracks2_times[-1])
        elif len(tracks1_data) > 0:
            if self.source_flag == 'baidu':
                sec_start = int(tracks1_times[0]*1e-2)*1e2
                sec_end = int(tracks1_times[-1]*1e-2)*1e2
            elif self.source_flag == 'tfd':
                sec_start = tracks1_times[0]
                sec_end = tracks1_times[-1]
        elif len(tracks2_data) > 0:
            if self.source_flag == 'baidu':
                sec_start = int(tracks2_times[0]*1e-2)*1e2
                sec_end = int(tracks2_times[-1]*1e-2)*1e2
            elif self.source_flag == 'tfd':
                sec_start = tracks2_times[0]
                sec_end = tracks2_times[-1]

        frame_id = 0

        sec = sec_start
        while sec <= sec_end:
            # make full use of road info
            tracks1 = copy.deepcopy(tracks1_data[np.where(np.logical_and(tracks1_data[:,0] >= sec, tracks1_data[:,0] < sec + self.time_step))[0]])
            tracks2 = copy.deepcopy(tracks2_data[np.where(np.logical_and(tracks2_data[:,0] >= sec, tracks2_data[:,0] < sec + self.time_step))[0]])

            if len(tracks1) > 0:
                tracks1[:,21] = tracks1[:,0]
                tracks1[:,0] = frame_id

                new_tracks1 = self.cvt_format_prediction2tracking(tracks1)

                tracks1_data_new = np.vstack([tracks1_data_new,new_tracks1])
            if len(tracks2) > 0:
                tracks2[:,21] = tracks2[:,0]
                tracks2[:,0] = frame_id

                new_tracks2 = self.cvt_format_prediction2tracking(tracks2)
                
                tracks2_data_new = np.vstack([tracks2_data_new,new_tracks2])
            
            if len(tracks1) > 0 and len(tracks2) > 0:
                self.veh2inf_frame_id[frame_id] = frame_id

            sec += self.time_step
            frame_id += 1

        return tracks1_data_new, tracks2_data_new


    def fuse_for_prediction_per_seq(self, tracks1_data_src,tracks1_data_tocken,tracks2_data_src,tracks2_data_tocken, car_results_save_path, road_results_save_path, av_pos):
        '''
            tracks1: frame,type,tracking_id,bbox-left,bbox-top,bbox-right,bbox-bottom,score,dimensions-height,dimensions-width,dimensions-length,
                     camera_bottom_center_x, camera_bottom_center_y, camera_bottom_center_z, rotation_y, alpha, lidar_center_x, lidar_center_y, lidar_center_z, rotation_z, truncated, occlude, token
            tracks2: frame,type,tracking_id,bbox-left,bbox-top,bbox-right,bbox-bottom,score,dimensions-height,dimensions-width,dimensions-length, 
                     camera_bottom_center_x, camera_bottom_center_y, camera_bottom_center_z, rotation_y, alpha, lidar_center_x, lidar_center_y, lidar_center_z, rotation_z, truncated, occlude, token
            1.The fusion strategy is designed based on the same accuracy of tracks1 and tracks2. 2.tracks1_data is car side, tracks2_data is road side.
        '''
        if len(tracks1_data_src) <= 0 and len(tracks2_data_src) <= 0:
            return [],[]
        tracks1_data = np.array(tracks1_data_src,dtype=float)
        tracks2_data = np.array(tracks2_data_src,dtype=float)
        AV = np.array(av_pos,dtype=float)

        #sample by 10hz; modify format
        tracks1_data_new,tracks2_data_new = self.gen_data(tracks1_data,tracks2_data)
        #matching
        cannot_fusion_v2i = set()
        if len(tracks1_data_new) > 0 and len(tracks2_data_new) and len(self.veh2inf_frame_id) > 0:
            cannot_fusion_v2i = self.get_tracks_fusion_info_per_seq(tracks1_data_new,tracks2_data_new)

        #fusion
        tracks1_data = tracks1_data_new
        tracks2_data = tracks2_data_new

        self.new_tracks_fusion = []
        self.new_road_outputs = []

        tracks2_ids = np.unique(tracks2_data[:,2])

        tracks1_frame_id = np.sort(np.unique(tracks1_data[:, 0]))
    
        for track1_frame_id in tracks1_frame_id:
            track1_frame_id = int(track1_frame_id)
            tracks1 = tracks1_data[tracks1_data[:, 0] == track1_frame_id]
            av_cur = AV[track1_frame_id]
           
            if track1_frame_id in self.veh2inf_frame_id:
                #coop pairs
                track2_frame_id = self.veh2inf_frame_id[track1_frame_id]
                tracks2 = tracks2_data[tracks2_data[:,0] == track2_frame_id]
               
                if len(tracks1) > 0 and len(tracks2):
                    #begin fusion

                    #frame pairs
                    #find ego vehicle
                    host_car_pose_x = av_cur[0]   #default: tracks1[0][24]
                    host_car_pose_y = av_cur[1]   #default: tracks1[0][25]

                    self.find_ego_vehicle(np.array([host_car_pose_x,host_car_pose_y],dtype=float),tracks2)
                    tracks2_data[tracks2_data[:, 0] == track2_frame_id] = tracks2

        tracks1_frame = np.sort(np.unique(tracks1_data[:, 21]))
        tracks2_frame = np.sort(np.unique(tracks2_data[:, 21]))
        start_frame = min(tracks1_frame.min(), tracks2_frame.min())
        tracks1_data[:, 0] = (tracks1_data[:, 21] - start_frame) * 10  #采样频率为10hz
        tracks2_data[:, 0] = (tracks2_data[:, 21] - start_frame) * 10
        #save fusion tracking results
        #save csv
        df_1 = pd.DataFrame({"header.lidar_timestamp":np.array(tracks1_data[:,21]), \
            "id":np.array(tracks1_data[:,2],dtype=int), \
                "type":self.id_cvt_name(tracks1_data[:,1]),\
                    "height":tracks1_data[:,10],"width":tracks1_data[:,11], \
                        "length":tracks1_data[:,12],"position.x":tracks1_data[:,13], \
                            "position.y":tracks1_data[:,14],"position.z":tracks1_data[:,15], \
                                "theta":tracks1_data[:,16], \
                                    "velocity.x":tracks1_data[:,3],"velocity.y":tracks1_data[:,4], \
                                        "host_car_pose.position.x":tracks1_data[:,24], \
                                            "host_car_pose.position.y":tracks1_data[:,25], \
                                                "sub_type":self.id_cvt_name(tracks1_data[:,5],True), \
                                                "frame_id":np.array(tracks1_data[:,0],dtype=int), \
                                                    "tag":np.array(tracks1_data[:,27],dtype=int)
                })
        df_1.to_csv(car_results_save_path,index=False)

        df_2 = pd.DataFrame({"header.lidar_timestamp": np.array(tracks2_data[:, 21]), \
                           "id": np.array(tracks2_data[:, 2], dtype=int), \
                           "type": self.id_cvt_name(tracks2_data[:, 1]), \
                           "height": tracks2_data[:, 10], "width": tracks2_data[:, 11], \
                           "length": tracks2_data[:, 12], "position.x": tracks2_data[:, 13], \
                           "position.y": tracks2_data[:, 14], "position.z": tracks2_data[:, 15], \
                           "theta": tracks2_data[:, 16], \
                           "velocity.x": tracks2_data[:, 3], "velocity.y": tracks2_data[:, 4], \
                           "host_car_pose.position.x": tracks2_data[:, 24], \
                           "host_car_pose.position.y": tracks2_data[:, 25], \
                           "sub_type": self.id_cvt_name(tracks2_data[:, 5], True), \
                           "frame_id": np.array(tracks2_data[:, 0], dtype=int), \
                           "tag": np.array(tracks2_data[:, 27], dtype=int)
                           })
        df_2.to_csv(road_results_save_path, index=False)
          
        return 

def fuse_for_prediction_from_tfd(seq):

    iou_threshold_2d = 0.3
    hungarian = True
    time_step = 0.1
    ego_offset = 6.0
    source_flag = 'tfd'
    solve_wrong_association = True

    car_results_save_path = os.path.join(tfd_car_save_path, seq)
    road_results_save_path = os.path.join(tfd_road_save_path, seq)
    print(car_results_save_path)
    print(road_results_save_path)
    
    car_file_path = os.path.join(tfd_data_path, 'vehicle-trajectories', split, seq)
    road_file_path = os.path.join(tfd_data_path, 'infrastructure-trajectories', split, seq)

    if not os.path.exists(car_file_path) or not os.path.exists(road_file_path):
        return

    tracks1_data = []
    av_pos = []
    tracks1_data_tocken = []
    tracks2_data = []
    tracks2_data_tocken = []
    with open(car_file_path) as track:
        for l in track.readlines()[1:]:
            if l.strip(','):
                if len(l.split(',')) == 16:
                    l = l.split(',')
                    if not l[11]:
                        av_xy = l[6:8]
                        av_pos.append(av_xy)
                        continue
                    data = [0] * 28
                    data[27] = tag2id[l[5]]
                    data[0], data[1], data[2], data[26] = l[1], name2id[l[3]], l[2], subname2id[l[4]]  #0:timestamp,1:type,2:agent_id
                    data[8], data[9], data[10] = l[11], l[10], l[9]   #8:hight,9:width,10:length
                    data[11:13] = l[6:8]   #11:x,12:y,13:z
                    data[22:24] = l[13:15]  #22:v_x,23:v_y
                    data[14] = l[12]       #14:theta
                    tracks1_data.append(data)
                    tracks1_data_tocken.append('')
    index = 0            
    with open(road_file_path) as track:
        for l in track.readlines()[1:]:
            if l.strip(','):
                if len(l.split(',')) == 16:
                    l = l.split(',')
                    if not l[11]:
                        continue
                    data = [0] * 28
                    data[27] = tag2id[l[5]]
                    data[0], data[1], data[2] = l[1], name2id[l[3]], l[2]
                    data[8], data[9], data[10] = l[11], l[10], l[9]
                    data[11:13] = l[6:8]
                    data[22:24] = l[13:15]
                    data[14] = l[12]
                    tracks2_data.append(data)
                    tracks2_data_tocken.append('')
                    index = index + 1

    matching_fusion = PredictionFusion(iou_threshold_2d, hungarian, time_step, ego_offset, source_flag, solve_wrong_association)
    matching_fusion.fuse_for_prediction_per_seq(tracks1_data,tracks1_data_tocken,tracks2_data,tracks2_data_tocken,car_results_save_path, road_results_save_path, av_pos)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='mathing fusion')

    parser.add_argument("--data_root", type=str, default="/data/lixc/hdgt/visual_raw_data/")     
    parser.add_argument("--split", help="split.", type=str, default='val') #train; val; test_obs
 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    data_root = args.data_root
    tfd_data_path = os.path.join(data_root, 'cooperative-vehicle-infrastructure')
    tfd_car_save_path = os.path.join(data_root, 'cooperative-vehicle-infrastructure/tfd_car', args.split, 'data')
    tfd_road_save_path = os.path.join(data_root, 'cooperative-vehicle-infrastructure/tfd_road', args.split, 'data')
    if not os.path.exists(tfd_car_save_path):
       os.makedirs(tfd_car_save_path)
    if not os.path.exists(tfd_road_save_path):
       os.makedirs(tfd_road_save_path)
    split = args.split

    seq_list = os.listdir(os.path.join(tfd_data_path, 'vehicle-trajectories', split))
    print('seq_list len: ', len(seq_list))
   
    pool = multiprocessing.Pool(processes = 16)
    pool.map_async(fuse_for_prediction_from_tfd, seq_list).get()
    pool.close()
    pool.join()


    

