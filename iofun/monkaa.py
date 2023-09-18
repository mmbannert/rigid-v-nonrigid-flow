
import re
import numpy as np
focal_length=1050


def read(file):
    if file.endswith('.flo'): return flow_read(file)
 
    elif file.endswith('.pfm'): return disp_read(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)
        
def disp_read(file):
    
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian 
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data,scale
        
def flow_read(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return disp_read(name)[0][:,:,0:2]
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    #return flow
    return flow.astype(np.float32)

def cam_read(file_path):
    focal_length=1050
    M = np.zeros((3, 3), dtype='float32')
    M[0, 0] = focal_length
    M[1, 1] = focal_length
    M[0, 2] = 479.5
    M[1, 2] = 269.5
    M[2, 2] = 1

    f = open(file_path, 'r')
    lines = f.readlines()
   
    N = []
    i=0
    
    while 4*i+2 < len(lines):
        
        line=lines[4*i+2].split()
        values=line[1:]
        # Get the frame ID from the line

        r_matrix = np.array(values).reshape(4,4)
        convertedArray = r_matrix.astype(float)
        # convertedArray=convertedArray[:3]
        N.append((convertedArray))
        i += 1
        
    return M, N