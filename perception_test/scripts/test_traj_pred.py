#! /usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import os.path as osp
import importlib
import numpy as np
import torch
import mmengine
from perception_test.msg import track_result, traj_pred_result


def data_preprocess(tracks):
    
    global config
    obs = []
    neis = []
    n_neighbors = []
    neis_mask = []
    refs = []
    rot_mats = []
    ids = []
    all_ids = list(tracks.keys())
    all_tracks = np.array([tracks[k] for k in all_ids])
    for i in range(len(all_tracks)):
        if all_tracks[i][-1][0] > 1e8:
            continue
        ob = all_tracks[i].copy()
        for j in range(6, -1, -1):
            if ob[j][0] > 1e8:
                ob[j] = ob[j+1]
        nei = all_tracks[np.arange(len(all_tracks)) != i]
        dist = np.linalg.norm(ob.reshape(1, 8, 2) - nei, axis=-1)
        dist = np.min(dist, axis=-1)
        nei = nei[dist < config.dist_threshold]
        
        refs.append(ob[-1])
        nei = nei - ob[-1]
        ob = ob - ob[-1]
        
        angle = np.arctan2(ob[0][1], ob[0][0])
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                            [np.sin(angle), np.cos(angle)]])
        ob = np.matmul(ob, rot_mat)
        if nei.shape[0] > 0:
            nei = np.matmul(nei, rot_mat)
            
        obs.append(ob)
        neis.append(nei)
        n_neighbors.append(nei.shape[0])
        rot_mats.append(rot_mat)
        ids.append(all_ids[i])
        
    max_neighbors = max(n_neighbors)
    if max_neighbors == 0:
        max_neighbors = 1
    neis_pad = []
    for neighbor, n in zip(neis, n_neighbors):
        neis_pad.append(np.pad(neighbor, ((0, max_neighbors-n), (0, 0),  (0, 0)), "constant"))
        mask = np.zeros((max_neighbors, max_neighbors))
        mask[:n, :n] = 1
        neis_mask.append(mask)
        
    obs = torch.tensor(np.stack(obs), dtype=torch.float32).cuda()
    neis = torch.tensor(np.stack(neis_pad), dtype=torch.float32).cuda()
    neis_mask = torch.tensor(np.stack(neis_mask), dtype=torch.float32).cuda()
    refs = torch.tensor(np.stack(refs), dtype=torch.float32).cuda()
    rot_mats = torch.tensor(np.stack(rot_mats), dtype=torch.float32).cuda()
    
    return obs, neis, neis_mask, refs, rot_mats, ids


def update_tracks(tracks, update_ids, update_bboxes):
    
    is_updated = {k: False for k in tracks.keys()}
    for i in range(len(update_ids)):
        if tracks.get(update_ids[i]) is None:
            tracks[update_ids[i]] = [[1e9, 1e9]] * 7 + [update_bboxes[i][:2].tolist()]
        else:
            tracks[update_ids[i]].pop(0)
            tracks[update_ids[i]].append(update_bboxes[i][:2].tolist())
            # 对丢失的track进行线性插值
            lost_frame = -1
            for j in range(1, 7):
                if tracks[update_ids[i]][j][0] > 1e8 and tracks[update_ids[i]][j-1][0] < 1e8:
                    lost_frame = j
                    break
            if lost_frame != -1:
                start = tracks[update_ids[i]][lost_frame - 1]
                end = tracks[update_ids[i]][-1]
                for j in range(lost_frame, 7):
                    tracks[update_ids[i]][j] = [start[0] + (end[0] - start[0]) / (8 - lost_frame) * (j - lost_frame + 1),
                                                start[1] + (end[1] - start[1]) / (8 - lost_frame) * (j - lost_frame + 1)]
            is_updated[update_ids[i]] = True
            
    for k in list(tracks.keys()):
        if is_updated.get(k, True) == False:
            tracks[k].pop(0)
            tracks[k].append([1e9, 1e9])
                    

def callback(data):
    
    global motion_modes, model, tracks
    rospy.loginfo("traj_pred frame: %s", data.token)
    
    pedestrian_label = 1
    track_labels = np.array(data.labels)
    mask = track_labels == pedestrian_label
    track_ids = np.array(data.ids)[mask]
    track_bboxes = np.array(data.bboxes).reshape(-1, 8)[mask]
    track_states = np.array(data.states)[mask]
    update_ids = []
    update_bboxes = []
    for i in range(len(track_bboxes)):
        state = track_states[i].split('_')
        if state[0] == 'birth' or (state[0] == 'alive' and int(state[1]) == 1):
            update_ids.append(track_ids[i])
            update_bboxes.append(track_bboxes[i])
    update_tracks(tracks, update_ids, update_bboxes)
            
    data_input = data_preprocess(tracks)
    with torch.no_grad():
        obs, neis, neis_mask, refs, rot_mats, ids = data_input
        preds, scores = model(obs, neis, motion_modes, neis_mask, None, test=True, num_k=3)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        preds = preds.reshape(preds.shape[0], preds.shape[1], 12, 2)
        rot_mats_T = rot_mats.transpose(1, 2)
        obs_ori = torch.matmul(obs, rot_mats_T) + refs.unsqueeze(1)
        preds_ori = torch.matmul(preds, rot_mats_T.unsqueeze(1)) + refs.unsqueeze(1).unsqueeze(2)
    
    msg = traj_pred_result()
    msg.token = data.token
    msg.timestamp = data.timestamp
    msg.ego2global = data.ego2global
    msg.ids = ids
    msg.obs = obs_ori.cpu().numpy().flatten()
    msg.preds = preds_ori.cpu().numpy().flatten()
    msg.scores = scores.cpu().numpy().flatten()
    msg.pred_num = 3
    pub.publish(msg)


if __name__ == '__main__':
    
    rospy.init_node('traj_pred', anonymous=True)
    rospy.Subscriber('track_result', track_result, callback)
    pub = rospy.Publisher('traj_pred_result', traj_pred_result, queue_size=10)
    
    sys.path.append(osp.abspath('./'))
    from src.trajectory_prediction.scripts.model.model import TrajectoryModel
    
    config_path = './src/trajectory_prediction/scripts/configs/coda.py'
    motion_modes_file = './src/trajectory_prediction/scripts/checkpoints/CODA/coda_motion_modes.pkl'
    checkpoint = './src/trajectory_prediction/scripts/checkpoints/CODA/coda.pth'
    
    spec = importlib.util.spec_from_file_location('config', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    obs_len = config.OB_HORIZON
    pred_len = config.PRED_HORIZON
    
    motion_modes = torch.tensor(mmengine.load(motion_modes_file), dtype=torch.float32).cuda()
    model = TrajectoryModel(in_size=2, obs_len=obs_len, pred_len=pred_len, embed_size=config.model_hidden_dim, 
            enc_num_layers=2, int_num_layers_list=[1,1], heads=4, forward_expansion=2)
    model.load_state_dict(torch.load(checkpoint))
    model.cuda().eval()
    tracks = {}
    
    rospy.loginfo('finish init traj_pred node')
    rospy.spin()