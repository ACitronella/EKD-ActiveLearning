
import numpy as np
import torch


def load_embededing_from_path(path:str, normalized:bool=True):
    if path.endswith('.npy'):
        features = np.load(path)
    elif path.endswith('.pth'):
        features = torch.load(path)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features

