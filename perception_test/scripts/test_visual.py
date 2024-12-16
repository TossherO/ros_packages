#! /usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mmengine
from perception_test.msg import traj_pred_result


def callback(data):
    
    global points_paths, pcd_range, data_plot
    token = data.token
    if points_paths.get(token) is not None:
        ego2global = np.array(data.ego2global).reshape(4, 4)
        pred_num = data.pred_num
        ids = data.ids
        obs = np.array(data.obs).reshape(-1, 16, 2)
        preds = np.array(data.preds).reshape(-1, pred_num, 24, 2)
        scores = np.array(data.scores).reshape(-1, pred_num)
        
        # 获取自身的二维位置和朝向
        self_pos = ego2global[:3, 3]
        self_yaw = np.arctan2(ego2global[1, 0], ego2global[0, 0])
        plot_range = pcd_range.copy()
        plot_range[0] += self_pos[0]
        plot_range[1] += self_pos[0]
        plot_range[2] += self_pos[1]
        plot_range[3] += self_pos[1]
        plot_range[4] += self_pos[2]
        plot_range[5] += self_pos[2]
        
        # 加载并处理点云数据
        points = np.fromfile(points_paths[token], dtype=np.float32).reshape(-1, 4)[:, :3]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (points @ ego2global.T)[:, :3]
        mask = (points[:, 0] > plot_range[0]) & (points[:, 0] < plot_range[1]) & \
                (points[:, 1] > plot_range[2]) & (points[:, 1] < plot_range[3]) & \
                (points[:, 2] > plot_range[4]) & (points[:, 2] < plot_range[5])
        points = points[mask]
        
        data_plot['token'] = token
        data_plot['plot_range'] = plot_range
        data_plot['self_pos'] = self_pos
        data_plot['self_yaw'] = self_yaw
        data_plot['points'] = points
        data_plot['ids'] = ids
        data_plot['obs'] = obs
        data_plot['preds'] = preds
        data_plot['scores'] = scores


if __name__ == '__main__':
    
    rospy.init_node('test_visual', anonymous=True)
    rospy.Subscriber("traj_pred_result", traj_pred_result, callback)
    
    info_path = './src/data/CODA/coda_infos_ros_test.pkl'
    data_info = mmengine.load(info_path)
    points_paths = {}
    path_prefix = info_path[:info_path.rfind('/')+1]
    for data in data_info['data_list']:
        token = data['token']
        points_paths[token] = path_prefix + data['lidar_points']['lidar_path']
    pcd_range = [-20.0, 20.0, -20.0, 20.0, -0.5, 6.0]
    pre_token = None
    data_plot = {
        'token': None,
        'plot_range': None,
        'self_pos': None,
        'self_yaw': None,
        'points': None,
        'ids': None,
        'obs': None,
        'preds': None,
        'scores': None,
    }
    
    plt.ion()
    plt.figure(figsize=(30, 30))
    plt.axis('off')
    rospy.loginfo("finish init visual node")
    
    while not rospy.is_shutdown():
        plt.pause(0.1)
        if data_plot['token'] != pre_token:
            pre_token = data_plot['token']
            plt.cla()
            plt.xlim(data_plot['plot_range'][0], data_plot['plot_range'][1])
            plt.ylim(data_plot['plot_range'][2], data_plot['plot_range'][3])
            size = 0.05
            points = data_plot['points']
            ids = data_plot['ids']
            obs = data_plot['obs']
            preds = data_plot['preds']
            scores = data_plot['scores']
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=size)
            plt.scatter(data_plot['self_pos'][0], data_plot['self_pos'][1], c='r', s=100*size, zorder=3)
            plt.arrow(data_plot['self_pos'][0], data_plot['self_pos'][1], np.cos(data_plot['self_yaw']), np.sin(data_plot['self_yaw']), color='r', width=size, zorder=2)
            for i in range(obs.shape[0]):
                plt.scatter(obs[i, -1, 0], obs[i, -1, 1], c='r', s=50*size, zorder=3)
                # plt.text(obs[i, -1, 0], obs[i, -1, 1], ids[i], fontsize=6*size, color='black', zorder=4)
                plt.plot(obs[i, :, 0], obs[i, :, 1], c='r', linestyle='-', linewidth=10*size, zorder=2)
                for j in range(preds.shape[1]):
                    plt.plot(preds[i, j, :, 0], preds[i, j, :, 1], c='b', linestyle='--', linewidth=10*size, zorder=1)
                    plt.scatter(preds[i, j, -1, 0], preds[i, j, -1, 1], c='b', marker='*', s=30*size, zorder=1)
            plt.axis('off')
            