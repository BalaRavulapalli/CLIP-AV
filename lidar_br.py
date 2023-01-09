import os
import cv2
import re
import numpy as np


# Function to read calibration file
# Input: Calibration Text File Path
# Output: P2: 3D camera coordinates to 2D image pixels
# Output: vtc_mat: 3D Lidar coordinates to 3D camera coordinates 
def read_calib(calib_path):
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)

# Function to read lidar data
# Input: Path to lidar bin file
# Input: Camera 3D to Camera 2D Matrix
# Input: Lidar 3D to Camera 3D Matrix
# Output: Valid points in Lidar Coordinates
def read_velodyne(path, P, vtc_mat, IfReduce=True):
    max_row = 374  # y
    max_col = 1241  # x

    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        return lidar

    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros(shape=lidar.shape)
    lidar_copy[:, :] = lidar[:, :]

    velo_tocam = vtc_mat
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, velo_tocam.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.mat(velo_tocam).I
    velo_tocam = np.array(velo_tocam)
    normal = velo_tocam
    normal = normal[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

    return lidar_copy[mask]

# Function to convert 3D Camera coordinates to 3D Lidar coordinates
# Input: 3D Camera Points
# Input: Lidar 3D to Camera 3D Matrix
# Output: 3D Lidar Points

def cam_to_velo(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

# Function to convert 3D Lidar coordinates to 3D Camera coordinates
# Input: 3D Camera Points
# Input: Lidar 3D to Camera 3D Matrix
# Output: 3D Lidar Points
def velo_to_cam(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat)
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

# Function to read image
# Input: Image path
# Output: Image matrix
def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im

# Function to read labels
# Input: Input label path file
# Output: Array of Box coordinates
# Output: Array of label names
def read_detection_label(path):

    boxes = []
    names = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:],np.float32)
                boxes.append(line)
                names.append(this_name)

    return np.array(boxes),np.array(names)

# Function to read tracking label
# Input: Input label path file
# Output: Frame Dictionary
# Output: Frame Name dictionary
def read_tracking_label(path):

    frame_dict={}

    names_dict={}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])

            if this_name != "DontCare":
                line = np.array(line[10:17],np.float32).tolist()
                line.append(ob_id)


                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict,names_dict



# Detection Dataset Class
class KittiDetectionDataset:
    # Initialization Function to read all paths
    def __init__(self,root_path,label_path = None):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        if label_path is None:
            self.label_path = os.path.join(self.root_path, "label_2")
        else:
            self.label_path = label_path

        self.all_ids = os.listdir(self.velo_path)

    # Length Function
    def __len__(self):
        return len(self.all_ids)
    
    # Get index function
    def __getitem__(self, item):
        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names