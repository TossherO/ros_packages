#! /usr/bin/env python

import rospy
import numpy as np
import torch
import open3d as o3d
from perception_test.msg import visual_msgs


def callback(data):
    
    global vis, ctr, label_colors, visual_data
    
    visual_data['ego2global'] = np.array(data.ego2global).reshape(4, 4)
    visual_data['points'] = np.array(data.points).reshape(-1, 3)
    visual_data['bboxes'] = np.array(data.bboxes).reshape(-1, 8)
    visual_data['labels'] = np.array(data.labels)
    if data.obs_trajs is not None:
        visual_data['obs_trajs'] = np.array(data.obs_trajs).reshape(-1, 8, 2)
        visual_data['pred_trajs'] = np.array(data.pred_trajs).reshape(-1, 3, 12, 2)
        visual_data['traj_labels'] = np.array(data.traj_labels)
        visual_data['obs_h'] = np.array(data.obs_h)
    else:
        visual_data['obs_trajs'] = None
        visual_data['pred_trajs'] = None
        visual_data['traj_labels'] = None
        visual_data['obs_h'] = None
    visual_data['update'] = True
    

if __name__ == '__main__':
    
    rospy.init_node('perception_visaul', anonymous=True)
    rospy.Subscriber("/perception_result", visual_msgs, callback)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 1
    render_option.line_width = 2
    ctr = vis.get_view_control()
    label_colors = [[0, 1, 0], [0, 0, 1]]
    visual_data = {}
    visual_data['update'] = False
    
    while not rospy.is_shutdown():
        
        if visual_data['update'] is True:
            
            vis.clear_geometries()
            ego2global = visual_data['ego2global']
            points = visual_data['points']
            bboxes = visual_data['bboxes']
            labels = visual_data['labels']
            obs_trajs = visual_data['obs_trajs']
            pred_trajs = visual_data['pred_trajs']
            traj_labels = visual_data['traj_labels']
            obs_h = visual_data['obs_h']
            global2ego = np.linalg.inv(ego2global)
            
            # pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1, 1, 1])
            
            # bboxes
            for i, bbox in enumerate(bboxes):
                if i == len(bboxes) - 1:
                    continue
                color = label_colors[labels[i]]
                bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                                R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
                np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
                bbox_geometry.color = color
                vis.add_geometry(bbox_geometry)
            vis.add_geometry(pcd)
            
            # trajectories
            if obs_trajs is not None:
                for i in range(len(obs_trajs)):
                    obs_points = np.concatenate([obs_trajs[i], np.ones((obs_trajs.shape[1], 1)) * obs_h[i]], axis=1)
                    obs_points = np.dot(global2ego, np.concatenate([obs_points, np.ones((obs_points.shape[0], 1))], axis=1).T).T[:, :3]
                    lines = [[j, j+1] for j in range(obs_trajs.shape[1]-1)]
                    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(obs_points), lines=o3d.utility.Vector2iVector(lines))
                    line_set.colors = o3d.utility.Vector3dVector([label_colors[traj_labels[i]]] * len(lines))
                    vis.add_geometry(line_set)
                    for j in range(pred_trajs.shape[1]):
                        pred_points = np.concatenate([pred_trajs[i, j], np.ones((pred_trajs.shape[2], 1)) * obs_h[i]], axis=1)
                        pred_points = np.dot(global2ego, np.concatenate([pred_points, np.ones((pred_points.shape[0], 1))], axis=1).T).T[:, :3]
                        pred_points = np.concatenate([obs_points[-1].reshape(1, 3), pred_points], axis=0)
                        lines = [[j, j+1] for j in range(pred_trajs.shape[2])]
                        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pred_points), lines=o3d.utility.Vector2iVector(lines))
                        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 1]] * len(lines))
                        vis.add_geometry(line_set)
                    
            # camera view
            # lookat = ego2global[:3, 3]
            # yaw = np.arctan2(ego2global[1, 0], ego2global[0, 0])
            # cam_dir = [-np.cos(yaw), -np.sin(yaw)] / np.linalg.norm([-np.cos(yaw), -np.sin(yaw)])
            # lookat[:2] -= cam_dir * 6
            # ctr.set_lookat(lookat)
            # ctr.set_zoom(0.2)
            # ctr.set_front([cam_dir[0], cam_dir[1], 1])
            # ctr.set_up([0, 0, 1])
            ctr.set_lookat([5, 0, 0])
            ctr.set_zoom(0.3)
            ctr.set_front([-1, 0, 1])
            ctr.set_up([0, 0, 1])
            
            visual_data['update'] = False

        vis.poll_events()
        vis.update_renderer()
        rospy.sleep(0.05)