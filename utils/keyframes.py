import numpy as np
import operator

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)

def extract_keyframes_indexes(frames, keyframe_num):
    if len(frames) <= keyframe_num:
        return range(len(frames))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    new_frames = []
    for i in range(len(frames)):
        curr_frame = frames[i]
        if curr_frame is not None and \
            prev_frame is not None:
            diff = np.asarray(abs(curr_frame - prev_frame))
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / len(diff)
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            new_frames.append(frame)
        prev_frame = curr_frame

    # 计算关键帧
    keyframe_id_set = set()
    # 排序取前N帧
    new_frames.sort(key=operator.attrgetter("diff"), reverse=True)
    for keyframe in new_frames[:keyframe_num]:
        keyframe_id_set.add(keyframe.id)
    return list(keyframe_id_set)
