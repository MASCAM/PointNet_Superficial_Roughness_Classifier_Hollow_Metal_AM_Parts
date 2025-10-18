import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
import urllib.request
import zipfile
import shutil

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile_name = os.path.basename(www)
    
    # Download the file
    urllib.request.urlretrieve(www, zipfile_name)
    
    # Unzip the file
    with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Move the extracted folder to DATA_DIR
    extracted_dir = zipfile_name[:-4]
    shutil.move(extracted_dir, DATA_DIR)
    
    # Remove the zip file
    os.remove(zipfile_name)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def get_rotation_matrix():
    """Get the last used rotation matrix for consistent point cloud and normal rotation"""
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                              [0, 1, 0],
                              [-sinval, 0, cosval]])
    return rotation_matrix


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_matrix = get_rotation_matrix()
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]  # Shape: (N, 2048, 3)
        label = f['label'][:]  # Shape: (N, 2048, 1)
        
        # Convert per-point labels to per-cloud labels
        # First remove the last dimension if it exists
        if label.ndim == 3:
            label = np.squeeze(label)  # Now shape is (N, 2048)
            
        # For each point cloud, get the most common label
        cloud_labels = []
        for i in range(label.shape[0]):
            # Count occurrences of each label in this point cloud
            unique, counts = np.unique(label[i], return_counts=True)
            # Get the most common label
            most_common_label = unique[np.argmax(counts)]
            cloud_labels.append(most_common_label)
            
        cloud_labels = np.array(cloud_labels, dtype=np.int32)
        
        # Debug information
        print(f"Data shape: {data.shape}")
        print(f"Original label shape: {label.shape}")
        print(f"Cloud labels shape: {cloud_labels.shape}")
        print(f"Unique cloud labels: {np.unique(cloud_labels)}")
        
        return data, cloud_labels

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_unified_h5(h5_filename):
    """Load unified H5 file containing multiple point clouds and labels"""
    with h5py.File(h5_filename, 'r') as f:
        points = f['points'][:]
        labels = f['labels'][:]
    return points, labels

def load_h5_data_label_unified(h5_filename):
    """Load data and labels from unified H5 file"""
    points, labels = load_unified_h5(h5_filename)
    
    # Ensure points are float32 and labels are int32
    points = points.astype(np.float32)
    labels = labels.astype(np.int32)
    
    return points, labels

def shuffle_unified_data(data, labels):
    """Shuffle data and labels while keeping corresponding pairs together"""
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx], labels[idx]
