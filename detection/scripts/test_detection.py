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

def callback(data):
    rospy.loginfo("get info_path: %s", data.data)
    input = mmengine.load(data.data)
    path_prefix = data.data[:data.data.rfind('/')+1]
    input['lidar_points']['lidar_path'] = path_prefix + input['lidar_points']['lidar_path']
    for sweep in input['lidar_sweeps']:
        sweep['lidar_points']['lidar_path'] = path_prefix + sweep['lidar_points']['lidar_path']
    for image in input['images'].values():
        image['img_path'] = path_prefix + image['img_path']
    
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
        bboxes_3d = bboxes_3d[scores_3d > 0.3].tensor.cpu().numpy()
        labels_3d = labels_3d[scores_3d > 0.3].cpu().numpy()
        scores_3d = scores_3d[scores_3d > 0.3].cpu().numpy()
        
    msg = String(data = 'bbox:\n' + str(bboxes_3d) + ' label:\n' + str(labels_3d) + ' score:\n' + str(scores_3d))
    pub.publish(msg)
    
    
if __name__ == '__main__':
    rospy.init_node('detection', anonymous=True)
    rospy.Subscriber("info_path", String, callback)
    pub = rospy.Publisher('model_result', String, queue_size=10)
    
    sys.path.append(osp.abspath('./'))
    cfg = Config.fromfile('./src/detection/scripts/configs/cmt_nus.py')
    checkpoint = './src/detection/ckpts/cmt_nus.pth'
    register_all_modules()
    model = MODELS.build(cfg.model)
    pipeline = []
    for transform in cfg.test_dataloader.dataset.pipeline:
        pipeline.append(TRANSFORMS.build(transform))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cuda().eval()
    
    rospy.loginfo("finish init model")
    rospy.spin()