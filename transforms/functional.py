"""
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import torch

def to_tensor(array):
    """Convert a 2D `numpy.ndarray`` to tensor, do transpose first.

    See ``ToTensor`` for more details.

    Args:
        array (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    assert len(array.shape) == 2
    array = array.transpose((1, 0))

    return torch.from_numpy(array)


