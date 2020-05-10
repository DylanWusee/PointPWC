import numpy as np


def next_pixel2pc(flow, disparity, save_path=None, f=-1050., cx=479.5, cy=269.5):
    height, width = disparity.shape

    BASELINE = 1.0
    depth = -1. * f * BASELINE / disparity

    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx + flow[..., 0]) * -1. / disparity)[:,
        :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy + flow[..., 1]) * 1. / disparity)[:,
        :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)

    if save_path is not None:
        np.save(save_path, pc)
    return pc


def pixel2pc(disparity, save_path=None, f=-1050., cx=479.5, cy=269.5):
    height, width = disparity.shape

    BASELINE = 1.0
    depth = -1. * f * BASELINE / disparity

    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx) * -1. / disparity)[:, :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy) * 1. / disparity)[:, :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)

    if save_path is not None:
        np.save(save_path, pc)
    return pc