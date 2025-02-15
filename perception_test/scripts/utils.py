import numpy as np
import torch


def data_preprocess(tracks, config):
    
    obs_len = config['obs_len']
    nei_radius = config['nei_radius']
    obs = []
    neis_ = []
    num_neis = []
    self_labels = []
    nei_labels_ = []
    refs = []
    rot_mats = []
    bboxes = []
    all_label_ids = list(tracks.keys())
    all_labels = np.array([int(label_id.split('_')[0]) for label_id in all_label_ids])
    all_tracks = np.array([tracks[k]['data'] for k in all_label_ids])
    all_bboxes = np.array([tracks[k]['bbox'] for k in all_label_ids])

    for i in range(len(all_tracks)):
        if all_tracks[i][-1][0] > 1e8 or all_tracks[i][obs_len-2][0] > 1e8 or all_labels[i] == 3:
            continue
        ob = all_tracks[i].copy()
        for j in range(obs_len - 2, -1, -1):
            if ob[j][0] > 1e8:
                ob[j] = ob[j+1]
        nei = all_tracks[np.arange(len(all_tracks)) != i]
        nei_labels = all_labels[np.arange(len(all_labels)) != i]
        now_nei_radius = [nei_radius[label] for label in nei_labels]
        dist_threshold = np.maximum(nei_radius[all_labels[i]], now_nei_radius)
        dist = np.linalg.norm(ob[:obs_len].reshape(1, obs_len, 2) - nei, axis=-1)
        dist = np.min(dist, axis=-1)
        nei = nei[dist < dist_threshold]
        nei_labels = nei_labels[dist < dist_threshold]
        
        ref = ob[-1]
        ob = ob - ref
        if nei.shape[0] != 0:
            nei = nei - ref
        angle = np.arctan2(ob[0][1], ob[0][0])
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                            [np.sin(angle), np.cos(angle)]])
        ob = np.dot(ob, rot_mat)
        if nei.shape[0] != 0:
            nei = np.dot(nei, rot_mat)
        
        obs.append(ob)
        neis_.append(nei)
        num_neis.append(nei.shape[0])
        self_labels.append(all_labels[i])
        nei_labels_.append(nei_labels)
        refs.append(ref.flatten())
        rot_mats.append(rot_mat)
        bboxes.append(all_bboxes[i])
        
    if len(obs) == 0:
        return None
            
    max_num_nei = max(num_neis)
    if max_num_nei == 0:
        max_num_nei = 1
    nei_masks = torch.zeros(len(obs), max_num_nei, dtype=torch.bool)
    neis = torch.zeros(len(obs), max_num_nei, obs_len, 2)
    nei_labels = torch.zeros(len(obs), max_num_nei, dtype=torch.int32) - 1
    for i in range(len(obs)):
        nei_masks[i, :num_neis[i]] = True
        neis[i, :num_neis[i]] = torch.tensor(neis_[i])
        nei_labels[i, :num_neis[i]] = torch.tensor(nei_labels_[i])
    
    obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32)
    self_labels = torch.tensor(self_labels, dtype=torch.int32)
    refs = torch.tensor(np.stack(refs, axis=0), dtype=torch.float32)
    rot_mats = torch.tensor(np.stack(rot_mats, axis=0), dtype=torch.float32)
    bboxes = torch.tensor(np.stack(bboxes, axis=0), dtype=torch.float32)
    return obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, bboxes


def update_tracks(tracks, labels, ids, xys, bboxes, config):

    obs_len = config['obs_len']
    is_updated = {k: False for k in tracks.keys()}
    for i in range(len(labels)):
        label_id = str(labels[i]) + '_' + str(ids[i])
        if tracks.get(label_id) is None:
            tracks[label_id] = {
                'data': [[1e9, 1e9] for _ in range(obs_len - 1)] + [xys[i].tolist()],
                'bbox': bboxes[i],
                'label': labels[i], 
                'lost_frame': 0}
        else:
            tracks[label_id]['data'].pop(0)
            tracks[label_id]['data'].append(xys[i].tolist())
            tracks[label_id]['bbox'] = bboxes[i]
            tracks[label_id]['lost_frame'] = 0
            is_updated[label_id] = True

    for k in is_updated.keys():
        if not is_updated[k]:
            if tracks[k]['lost_frame'] < obs_len:
                tracks[k]['data'].pop(0)
                tracks[k]['data'].append([1e9, 1e9])
                tracks[k]['lost_frame'] += 1
            else:
                del tracks[k]
    return tracks