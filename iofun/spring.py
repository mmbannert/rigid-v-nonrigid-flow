import h5py, os
import numpy as np

def disp_read(file_path):
    with h5py.File(file_path, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {file_path} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


def flow_read(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]


def cam_read(file_path):
    intrinsics_path=os.path.join(file_path, "intrinsics.txt")
    extrinsics_path=os.path.join(file_path, "extrinsics.txt")
    M=np.loadtxt(intrinsics_path)
    N=np.loadtxt(extrinsics_path)
    
    return M,N