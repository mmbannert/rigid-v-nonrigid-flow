import numpy as np
from .utils import *

def get_hompxcoords(depth_img, img_sz):
    img_height, img_width = depth_img.shape
    Y, X = np.mgrid[:img_height, :img_width]
    Y = Y.astype('float32') * img_sz[0] / depth_img.shape[0]
    X = X.astype('float32') * img_sz[1] / depth_img.shape[1]
    return img2mat(np.stack((X, Y, np.ones_like(X), 1/depth_img), axis=2))


def calc_cameramatrix(M, N):
    '''
    Calculate camera matrix P from intrinsic matrix M and extrinsic matrix N.
    (see Szeliski, 2010, p. 60)

    M : array-like, shape(3, 3)
        Intrinsic camera matrix M
    N : array-like, shape(3, 4)
        Extrinsic camera matrix N
    '''
    K_tilde = np.hstack((
        np.vstack((M, np.zeros((1, 3)))),
        np.array([[0, 0, 0, 1]]).T))
    
    E = np.vstack((N, np.array([0, 0, 0, 1])))
    return K_tilde @ E
    

def px2world(pixel_coords, P):
    '''
    Compute (homogeneous) world coordinates from (homogeneous) pixel
    coordinates using camera matrix P.

    pixel_coords : array-like, shape (4, N)
        Homogenous pixel coordinates of N points. First row is x, second is y,
        third is 1s, fourth is 1/depth.
    
    P : array-like, shape (4, 4)
        Camera matrix mapping word coordinates from 

    '''
    world_coords = np.linalg.inv(P) @ pixel_coords
    world_coords = world_coords / np.tile(world_coords[3,:], (4, 1))
    return world_coords


def world2px(world_coords, P):
    '''
    Compute (homogeneous) pixel coordinates from (homogeneous) world
    coordinates using camera matrix P.

    pixel_coords : array-like, shape (4, N)
        Homogenous pixel coordinates of N points. First row is x, second is y,
        third is 1s, fourth is 1/depth.
    
    P : array-like, shape (4, 4)
        Camera matrix mapping word coordinates from 

    '''
    pixel_coords = P @ world_coords
    pixel_coords = pixel_coords / np.tile(pixel_coords[2,:], (4, 1))
    return pixel_coords
