import os.path as op
import cv2 as cv
import numpy as np
import json
from . import sintel, spring, kubric

''' 
Define functions to load ...
    1. RGB image pair
    2. Depth image pair
    3. Ground truth forward optical flow between pairs
    4. Camera matrices
    
For each dataset the inputs are loaded a little differently; yet what they
return in each case is the same.
'''

# Load the RGB image pair
def get_rgb(dataset: str, data_dir: str, mov_name: str, frame1_num: int):

    dataset = dataset.lower()
    if dataset=='sintel':
        rgb1_path = op.join(data_dir, 'training/final', mov_name, 
                        'frame_%04.i.png' % frame1_num)
        rgb2_path = op.join(data_dir, 'training/final', mov_name, 
                            'frame_%04.i.png' % (frame1_num+1))
        rgb1 = cv.imread(rgb1_path, -1)[...,::-1]
        rgb1 = rgb1.astype('float32') / 255.

        rgb2 = cv.imread(rgb2_path, -1)[...,::-1]
        rgb2 = rgb2.astype('float32') / 255.

    elif dataset=='spring':
        rgb1_path = op.join(data_dir, mov_name,
                            'frame_left/frame_left_%04.i.png' % 
                            frame1_num)
        rgb2_path = op.join(data_dir, mov_name,
                            'frame_left/frame_left_%04.i.png' % 
                            (frame1_num+1))
        
    elif dataset=='kubric':
        rgb1_path = op.join(data_dir, mov_name, 'rgba_%05.i.png' % frame1_num)
        rgb2_path = op.join(data_dir, mov_name, 'rgba_%05.i.png' % (
            frame1_num+1))

    rgb1 = cv.imread(rgb1_path, cv.IMREAD_COLOR)[..., ::-1]
    rgb2 = cv.imread(rgb2_path, cv.IMREAD_COLOR)[..., ::-1]

    if rgb1.dtype=='uint8':
        rgb1 = rgb1.astype('float32') / 255.

    if rgb2.dtype=='uint8':
        rgb2 = rgb2.astype('float32') / 255.

    return rgb1, rgb2


def get_depth(dataset: str, data_dir: str, mov_name: str, frame1_num: int):

    dataset = dataset.lower()
    if dataset=='sintel':
        depth1_path = op.join(data_dir, 'mpi_sintel_depth/training/depth',
                      mov_name, 'frame_%04.i.dpt' % frame1_num)
        depth2_path = op.join(data_dir, 'mpi_sintel_depth/training/depth',
                            mov_name, 'frame_%04.i.dpt' % (
                                frame1_num+1))
        depth1 = sintel.depth_read(depth1_path)
        depth2 = sintel.depth_read(depth2_path)

    elif dataset=='spring':
        disp1_path = op.join(data_dir, mov_name,
                             'disp1_left/disp1_left_%04.i.dsp5' % 
                             frame1_num)
        disp2_path = op.join(data_dir, mov_name,
                             'disp1_left/disp1_left_%04.i.dsp5' % (
                             frame1_num+1))
        
        # need intrinsics to convert disparities to depths
        intrinsics_path = op.join(data_dir, mov_name, 'cam_data',
                                  'intrinsics.txt')
        intrinsics = np.loadtxt(intrinsics_path)

        disp1 = spring.disp_read(disp1_path)
        disp2 = spring.disp_read(disp2_path)

        depth1 = intrinsics[frame1_num-1][0] * .065 / disp1
        depth2 = intrinsics[frame1_num][0] * .065 / disp2

    elif dataset=='kubric':
        depth1 = kubric.depth_read(op.join(
            data_dir, mov_name,'depth_%05.i.tiff' % frame1_num))
        depth2 = kubric.depth_read(op.join(
            data_dir, mov_name,'depth_%05.i.tiff' % (frame1_num+1)))
    
    return depth1, depth2        



def get_camera(dataset: str, data_dir: str, mov_name: str, frame1_num: int):

    dataset = dataset.lower()
    if dataset=='sintel':
        cam1_path = op.join(data_dir, 'mpi_sintel_depth/training/camdata_left',
                            mov_name, 'frame_%04.i.cam' % frame1_num)
        cam2_path = op.join(data_dir, 'mpi_sintel_depth/training/camdata_left',
                            mov_name, 'frame_%04.i.cam' % (
                                frame1_num+1))

        cam1 = sintel.cam_read(cam1_path)
        cam2 = sintel.cam_read(cam2_path)

    elif dataset=='spring':
        cam_dir = op.join(data_dir, mov_name,'cam_data')

        intrinsics, extrinsics = spring.cam_read(cam_dir)

        intrinsics1 = np.array([
            [intrinsics[frame1_num-1, 0], 0, intrinsics[frame1_num-1, 2]],
            [0, intrinsics[frame1_num-1, 1], intrinsics[frame1_num-1, 3]],
            [0, 0, 1],
            ])

        intrinsics2 = np.array([
            [intrinsics[frame1_num, 0], 0, intrinsics[frame1_num, 2]],
            [0, intrinsics[frame1_num, 1], intrinsics[frame1_num, 3]],
            [0, 0, 1],
            ])

        extrinsics1 = extrinsics[frame1_num-1, :].reshape((4, 4))[:3, :]
        extrinsics2 = extrinsics[frame1_num, :].reshape((4, 4))[:3, :]

        cam1 = [intrinsics1, extrinsics1]
        cam2 = [intrinsics2, extrinsics2]

    elif dataset=='kubric':
        metadata_path = op.join(data_dir, mov_name, 'metadata.json')

        cam1 = kubric.cam_read(metadata_path, frame1_num)
        cam2 = kubric.cam_read(metadata_path, frame1_num+1)

    return cam1, cam2
            

def get_flow(dataset: str, data_dir: str, mov_name: str, frame1_num: int):

    dataset = dataset.lower()
    if dataset=='sintel':
        flow_path = op.join(data_dir, 'training/flow', mov_name,
                   'frame_%04.i.flo' % frame1_num)
        flow = np.stack(sintel.flow_read(flow_path), axis=2)

    elif dataset=='spring':
        flow_path = op.join(data_dir, mov_name, 'flow_FW_left',
                    'flow_FW_left_%04.i.flo5' % frame1_num)
        flow = spring.flow_read(flow_path)

    elif dataset=='kubric':
        flow_path = op.join(data_dir, mov_name, 'forward_flow_%05.i.png' % 
                            frame1_num)
        flow = kubric.flow_read(flow_path)

    return flow