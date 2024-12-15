#! /usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import os.path as osp
import numpy as np
import torch
import mmengine
from mmengine.config import Config
from mmdet3d.registry import MODELS, TRANSFORMS
from mmdet3d.utils import register_all_modules
from mmengine.runner import load_checkpoint
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures import Box3DMode
from perception_test.msg import detect_result


def callback(data):
    
    global model, pipeline, info_path, data_info, count
    rospy.loginfo("get input: %s", data.data)
    
    if data.data == 'n' and count < len(data_info['data_list']):
        input = data_info['data_list'][count]
        token = input['token']
        rospy.loginfo("infer frame: %s", token)
        timestamp = input['timestamp']
        ego2global = input['ego2global']
        path_prefix = info_path[:info_path.rfind('/')+1]
        input['lidar_points']['lidar_path'] = path_prefix + input['lidar_points']['lidar_path']
        # for sweep in input['lidar_sweeps']:
        #     sweep['lidar_points']['lidar_path'] = path_prefix + sweep['lidar_points']['lidar_path']
        for image in input['images'].values():
            image['img_path'] = path_prefix + image['img_path']
        input['box_type_3d'] = LiDARInstance3DBoxes
        input['box_mode_3d'] = Box3DMode.LIDAR
    
        with torch.no_grad():
            for transform in pipeline:
                input = transform(input)
            input['data_samples'] = [input['data_samples']]
            input['inputs']['points'] = [input['inputs']['points']]
            input['inputs']['img'] = [input['inputs']['img']]
            output = model.data_preprocessor(input, training=False)
            output = model(**output, mode='predict')
            bboxes_3d = output[0].get('pred_instances_3d')['bboxes_3d']
            labels_3d = output[0].get('pred_instances_3d')['labels_3d']
            scores_3d = output[0].get('pred_instances_3d')['scores_3d']
            bboxes_3d = bboxes_3d[scores_3d > 0.3].tensor.cpu().numpy().flatten()
            labels_3d = labels_3d[scores_3d > 0.3].cpu().numpy().flatten()
            scores_3d = scores_3d[scores_3d > 0.3].cpu().numpy().flatten()
        
        msg = detect_result()
        msg.token = token
        msg.bboxes = bboxes_3d
        msg.labels = labels_3d
        msg.scores = scores_3d
        msg.timestamp = timestamp
        msg.ego2global = ego2global.flatten()
        pub.publish(msg)
        count += 1
    
    
if __name__ == '__main__':
    
    rospy.init_node('detection', anonymous=True)
    rospy.Subscriber("perception_input", String, callback)
    pub = rospy.Publisher('detect_result', detect_result, queue_size=10)
    
    sys.path.append(osp.abspath('./'))
    cfg = Config.fromfile('./src/detection/scripts/configs/cmt_coda.py')
    checkpoint = './src/detection/ckpts/cmt_coda.pth'
    info_path = './src/data/CODA/coda_infos_ros_test.pkl'
    register_all_modules()
    model = MODELS.build(cfg.model)
    pipeline = []
    for transform in cfg.test_dataloader.dataset.pipeline:
        pipeline.append(TRANSFORMS.build(transform))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cuda().eval()
    data_info = mmengine.load(info_path)
    count = 0
    
    rospy.loginfo("finish init detection node")
    rospy.spin()