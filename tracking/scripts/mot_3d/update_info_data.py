""" a general interface for aranging the things inside a single tracklet
    data structure for updating the life cycles and states of a tracklet 
"""
from .data_protos import BBox


class UpdateInfoData:
    def __init__(self, mode, bbox: BBox, frame_index, ego, dets=None, pc=None):
        self.mode = mode   # association state
        self.bbox = bbox
        self.ego = ego    
        self.frame_index = frame_index
        self.pc = pc
        self.dets = dets