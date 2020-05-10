import os, sys
import os.path as osp
import numpy as np
from multiprocessing import Pool

from kitti_utils import *

calib_root = './utils/calib_cam_to_cam/'
data_root = sys.argv[1]
disp1_root = osp.join(data_root, 'training/disp_occ_0')
disp2_root = osp.join(data_root, 'training/disp_occ_1')
op_flow_root = osp.join(data_root, 'training/flow_occ')

save_path = sys.argv[2]


def process_one_frame(idx):
    sidx = '{:06d}'.format(idx)

    calib_path = osp.join(calib_root, sidx + '.txt')
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    disp1_path = osp.join(disp1_root, sidx + '_10.png')
    disp1, valid_disp1 = load_disp(disp1_path)
    depth1 = disp_2_depth(disp1, valid_disp1, focal_length_pixel)
    pc1 = pixel2xyz(depth1, P_rect_left)

    disp2_path = osp.join(disp2_root, sidx + '_10.png')
    disp2, valid_disp2 = load_disp(disp2_path)
    depth2 = disp_2_depth(disp2, valid_disp2, focal_length_pixel)

    valid_disp = np.logical_and(valid_disp1, valid_disp2)

    op_flow, valid_op_flow = load_op_flow(osp.join(op_flow_root, '{:06d}_10.png'.format(idx)))
    vertical = op_flow[..., 1]
    horizontal = op_flow[..., 0]
    height, width = op_flow.shape[:2]

    px2 = np.zeros((height, width), dtype=np.float32)
    py2 = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            if valid_op_flow[i, j] and valid_disp[i, j]:
                try:
                    dx = horizontal[i, j]
                    dy = vertical[i, j]
                except:
                    print('error, i,j:', i, j, 'hor and ver:', horizontal[i, j], vertical[i, j])
                    continue

                px2[i, j] = j + dx
                py2[i, j] = i + dy

    pc2 = pixel2xyz(depth2, P_rect_left, px=px2, py=py2)

    final_mask = np.logical_and(valid_disp, valid_op_flow)

    valid_pc1 = pc1[final_mask]
    valid_pc2 = pc2[final_mask]

    truenas_path = osp.join(save_path, '{:06d}'.format(idx))
    os.makedirs(truenas_path, exist_ok=True)
    np.save(osp.join(truenas_path, 'pc1.npy'), valid_pc1)
    np.save(osp.join(truenas_path, 'pc2.npy'), valid_pc2)


pool = Pool(4)
indices = range(200)
pool.map(process_one_frame, indices)
pool.close()
pool.join()
