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


