import numpy as np
from scipy.interpolate import griddata

def img2mat(img_in):
    assert len(img_in.shape) <= 3
    if len(img_in.shape) < 3:
        n_channels = 1
        n_pixels = np.prod(img_in.shape)
    else:
        n_channels = img_in.shape[-1]
        n_pixels = np.prod(img_in.shape[:-1])
    return img_in.reshape((n_pixels, n_channels)).T


def mat2img(pxlist_in, img_shape):
    return np.reshape(pxlist_in, img_shape)


def interpolate_nans_channel(nan_ch):
    assert len(nan_ch.shape) == 2, "expected a 2d matrix"
    out_ch = nan_ch.copy()
    if np.isnan(nan_ch).any():
        
        coords = np.mgrid[0:nan_ch.shape[0], 0:nan_ch.shape[1]]
        
        pts = np.asarray(coords).transpose(1, 2, 0).reshape(-1, 2)
        vals = nan_ch.ravel()
        
        nan_ids = np.where(np.isnan(vals))[0]
        nonnan_pts = pts[np.logical_not(np.isnan(vals))]
        nonnan_vals = vals[np.logical_not(np.isnan(vals))]

        interp_val = griddata(nonnan_pts, nonnan_vals, pts[nan_ids, :],
                              method='nearest')
        out_ch[pts[nan_ids, 0], pts[nan_ids, 1]] = interp_val
                    
    return out_ch
    

def interpolate_nans(nan_img):
    is_single_channel = len(nan_img.shape) == 2
    
    if np.isnan(nan_img).any():
        if is_single_channel:
            nan_img = np.asarray([nan_img]).transpose(1, 2, 0)

        out_img = nan_img.copy()
        for ch_num in range(nan_img.shape[2]):
            out_img[..., ch_num] = interpolate_nans_channel(nan_img[...,
                                                                    ch_num])
            
        if is_single_channel:
            out_img = out_img[..., 0]

    else:
        print('no nans found in input!')
        out_img = nan_img
    return out_img