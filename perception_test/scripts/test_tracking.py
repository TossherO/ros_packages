#! /usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import os.path as osp
import yaml
import numpy as np
import torch
from perception_test.msg import detect_result, track_result


def callback(data):
    
    global trackers, class_labels
    rospy.loginfo("tracking frame: %s", data.token)
    
    ego2global = np.array(data.ego2global).reshape(4, 4)
    pre_labels = np.array(data.labels)
    pre_bboxes = np.array(data.bboxes).reshape(-1, 7)
    pre_scores = np.array(data.scores)
    labels = []
    ids = []
    bboxes = []
    states = []
    for i, label in enumerate(class_labels):
        mask = pre_labels == label
        dets = np.concatenate([pre_bboxes[mask], pre_scores[mask][:, None]], axis=1).tolist()
        frame_data = FrameData(dets=dets, ego=ego2global, pc=None, det_types=pre_labels[mask], time_stamp=float(data.timestamp))
        frame_data.dets = [BBox.bbox2world(ego2global, det) for det in frame_data.dets]
        results = trackers[i].frame_mot(frame_data)
        labels.extend([trk[3] for trk in results])
        ids.extend([trk[1] for trk in results])
        result_pred_bboxes = np.array([BBox.bbox2array(trk[0]) for trk in results])
        bboxes.extend(result_pred_bboxes.flatten().tolist())
        states.extend([trk[2] for trk in results])

    msg = track_result()
    msg.token = data.token
    msg.timestamp = data.timestamp
    msg.ego2global = data.ego2global
    msg.labels = labels
    msg.ids = ids
    msg.bboxes = bboxes
    msg.states = states
    pub.publish(msg)
    

if __name__ == '__main__':
    
    rospy.init_node('tracking', anonymous=True)
    rospy.Subscriber('detect_result', detect_result, callback)
    pub = rospy.Publisher('track_result', track_result, queue_size=10)
    
    sys.path.append(osp.abspath('./'))
    sys.path.append(osp.abspath('./src/tracking/scripts'))
    from mot_3d.mot import MOTModel
    from mot_3d.data_protos import BBox
    from mot_3d.frame_data import FrameData
    config_path = './src/tracking/scripts/configs/coda_configs/diou.yaml'
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    class_labels = [0, 1, 2]
    class_names = ['car', 'pedestrian', 'cyclist']
    trackers = [MOTModel(configs, class_names[label]) for label in class_labels]
    
    rospy.loginfo('finish init tracking node')
    rospy.spin()