#! /usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import os.path as osp
sys.path.append(osp.abspath('./'))
sys.path.append(osp.abspath('./src/tracking/scripts'))
import yaml
import numpy as np
import torch
import mmengine
from mmengine.config import Config
from mmdet3d.registry import MODELS, TRANSFORMS
from mmdet3d.utils import register_all_modules
from mmengine.runner import load_checkpoint
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures import Box3DMode
from perception_test.msg import inputs, visual_msgs
from src.perception_test.scripts.utils import *
from mot_3d.mot import MOTModel
from mot_3d.data_protos import BBox
from mot_3d.frame_data import FrameData
from src.trajectory_prediction.scripts.model.model import TrajectoryModel


def callback(data):
    
    global info_path, data_info, count, pipeline, detect_model, class_labels, class_names, trackers, traj_pred_model
    rospy.loginfo("get input: %s", data.data)
    time = rospy.get_time()
    
    if data.data == 'n' and count < len(data_info['data_list']):
        input = data_info['data_list'][count]
        token = input['token']
        rospy.loginfo("infer frame: %s", token)
        timestamp = input['timestamp']
        ego2global = input['ego2global']
        path_prefix = info_path[:info_path.rfind('/')+1]
        input['lidar_points']['lidar_path'] = path_prefix + input['lidar_points']['lidar_path']
        for image in input['images'].values():
            image['img_path'] = path_prefix + image['img_path']
        input['box_type_3d'] = LiDARInstance3DBoxes
        input['box_mode_3d'] = Box3DMode.LIDAR

        # detection
        with torch.no_grad():
            for transform in pipeline:
                input = transform(input)
            input['data_samples'] = [input['data_samples']]
            input['inputs']['points'] = [input['inputs']['points']]
            input['inputs']['img'] = [input['inputs']['img']]
            output = detect_model.data_preprocessor(input, training=False)
            output = detect_model(**output, mode='predict')
            bboxes_3d = output[0].get('pred_instances_3d')['bboxes_3d']
            labels_3d = output[0].get('pred_instances_3d')['labels_3d']
            scores_3d = output[0].get('pred_instances_3d')['scores_3d']
            bboxes_3d = bboxes_3d[scores_3d > 0.1].tensor.cpu().numpy()
            labels_3d = labels_3d[scores_3d > 0.1].cpu().numpy()
            scores_3d = scores_3d[scores_3d > 0.1].cpu().numpy()
        
        # tracking
        track_labels = []
        track_ids = []
        track_bboxes = []
        track_states = []
        for i, label in enumerate(class_labels):
            mask = labels_3d == label
            dets = np.concatenate([bboxes_3d[mask], scores_3d[mask][:, None]], axis=1).tolist()
            frame_data = FrameData(dets=dets, ego=ego2global, pc=None, det_types=labels_3d[mask], time_stamp=float(timestamp))
            frame_data.dets = [BBox.bbox2world(ego2global, det) for det in frame_data.dets]
            results = trackers[i].frame_mot(frame_data)
            track_labels.append([trk[3] for trk in results])
            track_ids.append([trk[1] for trk in results])
            track_bboxes.append(np.array([BBox.bbox2array(trk[0]) for trk in results]))
            track_states.append([trk[2] for trk in results])
            
        # trajectory prediction
        topK = 3
        update_labels = []
        update_ids = []
        update_xys = []
        update_bboxes = []
        for i, label in enumerate(class_labels):
            for j in range(len(track_bboxes[i])):
                state = track_states[i][j].split('_')
                if state[0] == 'birth' or (state[0] == 'alive' and int(state[1]) == 1):
                    update_labels.append(label)
                    update_ids.append(track_ids[i][j])
                    update_xys.append(track_bboxes[i][j][:2])
                    update_bboxes.append(track_bboxes[i][j])
        update_labels.append(3)
        update_ids.append(0)
        update_xys.append(ego2global[:2, 3])
        update_bboxes.append(np.zeros(8))
        update_tracks(tracks, update_labels, update_ids, update_xys, update_bboxes, traj_pred_config)
        data_input = data_preprocess(tracks, traj_pred_config)
        if data_input is not None:
            with torch.no_grad():
                data_input = [tensor.cuda() for tensor in data_input]
                obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, obs_bboxes = data_input
                preds, scores, _ = traj_pred_model(obs, neis, nei_masks, self_labels, nei_labels)
                scores = torch.nn.functional.softmax(scores, dim=-1)
                topK_scores, topK_indices = torch.topk(scores, topK, dim=-1) # [B topK], [B topK]
                topK_preds = torch.gather(preds, 1, topK_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, preds.size(-2), preds.size(-1))) # [B topK pred_len in_size]
                rot_mats_T = rot_mats.transpose(1, 2)
                obs_ori = torch.matmul(obs, rot_mats_T) + refs.unsqueeze(1)
                preds_ori = torch.matmul(topK_preds, rot_mats_T.unsqueeze(1)) + refs.unsqueeze(1).unsqueeze(2)
                obs_ori = obs_ori.cpu().numpy()
                preds_ori = preds_ori.cpu().numpy()
        
        # visual message
        points = input['inputs']['points'][0][:, :3].cuda()
        points = torch.cat((points, torch.ones((points.shape[0], 1)).cuda()), dim=1)
        points = torch.matmul(torch.tensor(ego2global).float().cuda(), points.T).T[:, :3].cpu().numpy()
        bboxes = np.stack(update_bboxes)
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 5] / 2
        msg = visual_msgs()
        msg.token = token
        msg.timestamp = timestamp
        msg.ego2global = ego2global.flatten()
        msg.points = points.flatten()
        msg.bboxes = bboxes.flatten()
        msg.labels = update_labels
        msg.ids = update_ids
        if data_input is not None:
            msg.obs_trajs = obs_ori.flatten()
            msg.pred_trajs = preds_ori.flatten()
            msg.traj_labels = self_labels.cpu().numpy()
            msg.obs_h = obs_bboxes[:, 2].cpu().numpy()
        pub.publish(msg)
        
        count += 1
        rospy.loginfo("infer time: %s", rospy.get_time() - time)
    
    
if __name__ == '__main__':
    
    rospy.init_node('perception_infer', anonymous=True)
    rospy.Subscriber("perception_input", String, callback)
    pub = rospy.Publisher('perception_result', visual_msgs, queue_size=10)
    
    # load detection model
    cfg = Config.fromfile('./src/detection/scripts/configs/cmdt_coda.py')
    checkpoint = './src/detection/ckpts/cmdt_coda.pth'
    info_path = './src/data/CODA/coda_infos_ros_test.pkl'
    register_all_modules()
    detect_model = MODELS.build(cfg.model)
    pipeline = []
    for transform in cfg.test_dataloader.dataset.pipeline:
        pipeline.append(TRANSFORMS.build(transform))
    checkpoint = load_checkpoint(detect_model, checkpoint, map_location='cpu')
    detect_model.cuda().eval()
    
    # load tracking model
    config_path = './src/tracking/scripts/configs/coda_configs/diou.yaml'
    track_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    class_labels = [0, 1, 2]
    class_names = ['car', 'pedestrian', 'cyclist']
    trackers = [MOTModel(track_config, class_names[label]) for label in class_labels]
    
    # load trajectory prediction model
    config_path = './src/trajectory_prediction/scripts/configs/coda.yaml'
    checkpoint = './src/trajectory_prediction/checkpoints/coda_best.pth'
    traj_pred_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    traj_pred_model = TrajectoryModel(num_class=traj_pred_config['num_class'], in_size=2, 
                    obs_len=traj_pred_config['obs_len'], pred_len=traj_pred_config['pred_len'], 
                    embed_size=traj_pred_config['embed_size'], num_decode_layers=traj_pred_config['num_decode_layers'], 
                    scale=traj_pred_config['scale'], pred_single=False)
    traj_pred_model.load_state_dict(torch.load(checkpoint))
    traj_pred_model.cuda().eval()

    data_info = mmengine.load(info_path)
    tracks = {}
    count = 0
    
    rospy.loginfo("finish init infer node")
    rospy.spin()