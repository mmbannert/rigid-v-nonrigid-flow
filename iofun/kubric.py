import numpy as np
import cv2 as cv
import json
import pyquaternion as pyquat
import os.path as op


def gaus2d(size, std, scale, offset):
    H, W = size
    min_std = 1.0 / min(size)

    # Normalize grid generation
    grid_x = np.linspace(-1, 1, W)
    grid_y = np.linspace(-1, 1, H)

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Adjusting std for aspect ratio
    std_y = std
    std_x = std * (H / W)

    # 2D Gaussian
    gaus2d = np.exp(-1 * (grid_x**2 / (2 * std_x**2) + grid_y**2 / (2 * std_y**2)))

    return scale * gaus2d + offset

def depth_read(file_path):
    return cv.imread(file_path, cv.IMREAD_UNCHANGED) * gaus2d((512, 512), 1.3754354715e+00, 3.8593712449e-01, 6.1287075281e-01)


def flow_read(filename):
    mov_dir = op.split(filename)[0]

    # need the min, max values of flow to scale the raw optical flow
    data_ranges_path = op.join(mov_dir, 'data_ranges.json')
    with open(data_ranges_path, 'r') as f:
         data_ranges = json.load(f)
    
    flo_min = data_ranges['forward_flow']['min']
    flo_max = data_ranges['forward_flow']['max']
    
    # load optical flow
    flow_unscaled = cv.imread(filename, cv.IMREAD_UNCHANGED)[..., 1:3]
    
    # scale flow
    flow = flow_unscaled / 65535 * (flo_max - flo_min) + flo_min
    return flow


def cam_read(file_path: str, frame_num: int):
    with open(file_path, 'r') as f:
        metadata = json.load(f)

    '''
    Load camera intrinsics
    '''
    intrinsics = np.array(metadata['camera']['K'])
    # focal length and principal points need to be positive
    intrinsics = np.abs(intrinsics)
    # ... and expressed in pixels
    intrinsics[0, :] *= metadata['metadata']['resolution'][0]
    intrinsics[1, :] *= metadata['metadata']['resolution'][1]

    ''' 
    Load camera extrinsics
    '''
    # Use scipy's Rotation module to convert from quaternion to matrix
    # representation
    quat = metadata['camera']['quaternions'][frame_num]
    rot = pyquat.Quaternion(quat).rotation_matrix

    trans = np.array(
        metadata['camera']['positions'][frame_num]).reshape((3, 1))
    extrinsics = np.hstack((rot, trans))
    extrinsics = np.vstack((extrinsics, np.array([[0, 0, 0, 1]])))

    return intrinsics, extrinsics
