import numpy as np
from .camgeom import * 

def calc_rigid_and_nonrigid_flow(depth_pair: tuple, cam_pair: tuple, flo: 
    np.ndarray, img_sz) -> tuple:

    depth1, depth2 = depth_pair
    cam1, cam2 = cam_pair

    # get homogeneous screen coordinates for all pixels using depth map from first
    # frame
    px_s1 = get_hompxcoords(depth1, img_sz)
    
    # compute camera matrix from camera intrinsic and extrinsic matrices using 
    # equation 2.64 on page 60 in Szeliski (2010).
    P1 = calc_cameramatrix(cam1[0], cam1[1])
    
    # use camera matrix P1 to transform pixel locations from screen coordinates to
    # world coordinates
    px_w1 = px2world(px_s1, P1)

    '''
    The camera matrix in the next frame provides information about how much the 
    camera has moved (rotation and translation). To compute by how much each pixel
    would have moved due to camera motion alone, one can transform the points in
    world coordinates back to screen coordinates and check how much motion across
    pixels would have occurred.
    '''

    # first compute the camera matrix for frame 2 using intrinsic and extrinsic
    # matrices
    P2 = calc_cameramatrix(cam2[0], cam2[1])

    # use the new camera matrix (i.e., factor in the camera motion that has
    # occurred between frames 2 and 1) to transform the points back into screen
    # coordinates -- these are the expected screen coordinates assuming camera
    # motion alone
    px_s1_rigid = world2px(px_w1, P2)

    # subtract original pixel coordinates from the hypothetical coordinates to
    # obtain rigid flow
    rigid_flo_mat = px_s1_rigid[:2,:] - px_s1[:2,:]

    # reshape rigid flow to add spatial axes for visualization
    rigid_flo = mat2img(rigid_flo_mat.T, flo.shape)

    # the difference between gt optical flow and rigid flow is the nonrigid flow
    nonrigid_flo = flo - rigid_flo

    '''
    What does this mean for our model of fMRI signals? The regressors for total
    optical flow, rigid optical flow, nonrigid optical flow will correspond to the
    spatial mean of the flow magnitudes. The flow magnitude at a given pixel is
    simply the L2 norm of the flow vector at that location. 

    The regressors for this frame pair and the three types of flow therefore looks
    as follows.
    '''
    of_reg = np.linalg.norm(flo, axis=2).mean()
    rof_reg = np.linalg.norm(rigid_flo, axis=2).mean()
    nof_reg = np.linalg.norm(nonrigid_flo, axis=2).mean()

    print('mean optical flow magnitude         : %.04f' % of_reg)
    print('mean rigid optical flow magnitude   : %.04f' % rof_reg)
    print('mean nonrigid optical flow magnitude: %.04f' % nof_reg)
    
    return rigid_flo, nonrigid_flo