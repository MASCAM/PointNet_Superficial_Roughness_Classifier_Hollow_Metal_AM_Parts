import logging
import json
import numpy as np
from datetime import datetime
import ast
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.field_path import FieldPath
from datetime import datetime
import pandas as pd
import numpy as np
import open3d as o3d
#import pyvista as pv
from numpy import genfromtxt

import random


import os

def count_files_with_substring(folder_path, substring):

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and substring in f]
    return len(files)


def generate_random_rugosity(position, normal, DEFECT, defect_chances):

    random_number = random.randint(1, 100)
    if DEFECT and random_number >= 100 - defect_chances[1]:

        position += defect_chances[2] * normal

    elif random_number >= defect_chances[0]:

        position += defect_chances[2] * normal
        DEFECT = True

    else:

        DEFECT = False

    
    return position, DEFECT

#FILENAME = 'Basic_Hollow_Cylinder_12cmDD_30mmHH_main'
FILENAME = 'Basic_Hollow_Cube_main'
#FILENAME = 'Basic_Hollow_Cylinder_main'
#FILENAME = 'Basic_Hollow_Cylinder_80mm_main'
FILE_NUMBER = 0
#CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [20, 30, 0.25], [30, 50, 0.25], [40, 60, 0.6], [60, 80, 0.8]]
#CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [30, 50, 0.25], [60, 80, 0.8]]
CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [30, 50, 0.25], [60, 80, 0.8]]
#CLASS_NAMES = ['Excelent', 'Good', 'Fair', 'Poor', 'Bad']
CLASS_NAMES = ['Good', 'Fair', 'Poor']
#CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [30, 50, 0.25], [60, 80, 0.25], [85, 90, 0.8]]
#CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [30, 50, 0.35], [65, 80, 0.5], [90, 95, 0.8]]
BATCH_SIZE = 240
VALIDATION_BASE_PATH = ''
#VALIDATION_BASE_PATH = 'Validation/'

# Create necessary directories
os.makedirs(f'./Positional_data/{VALIDATION_BASE_PATH}Original', exist_ok=True)
os.makedirs(f'./Positional_data/{VALIDATION_BASE_PATH}Simulated', exist_ok=True)
os.makedirs(f'./3D_meshes/{VALIDATION_BASE_PATH}Simulated', exist_ok=True)

original_mesh = o3d.io.read_triangle_mesh(f"./3D_meshes/{VALIDATION_BASE_PATH}Original/{FILENAME}.stl")
# Tensor TriangleMesh not supported this function yet.
#mesh.compute_vertex_normals()

pcd = original_mesh.sample_points_poisson_disk(49152)
print(np.asarray(pcd.points))

pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)



downpcd = pcd.voxel_down_sample(voxel_size=0.03)
downpcd.estimate_normals(max_nn=30, radius=20)



#print(downpcd.point)
positions = downpcd.point.positions
positions_np = positions.numpy()
normals = downpcd.point.normals
normals_np = normals.numpy()
if count_files_with_substring(f'./Positional_data/{VALIDATION_BASE_PATH}Original', f'{FILENAME}_positions.csv') == 0:

    df = pd.DataFrame(positions_np[1:])
    df = pd.concat([df, pd.DataFrame(normals_np[1:])], axis=1)
    df.columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z']
    df.to_csv(f'./Positional_data/{VALIDATION_BASE_PATH}Original/{FILENAME}_positions.csv', index=False)


FILE_NUMBER = count_files_with_substring(f'./Positional_data/{VALIDATION_BASE_PATH}Simulated', FILENAME)

for i in range(0, BATCH_SIZE):

    DEFECT = False
    if i % 3 == 0:

        new_positions_np = positions_np.copy()
        new_normals_np = normals_np.copy()

    #print(i%5)
    for j in range(0, len(new_positions_np)):

        
        new_positions_np[j], DEFECT = generate_random_rugosity(new_positions_np[j], new_normals_np[j], DEFECT, CLASS_DEFECT_CHANCES[i%3])
        #print(new_positions_np[i])

    #downpcd.estimate_normals(max_nn=30, radius=20)
    #positions = downpcd.point.positions# + downpcd.point.positions
    #print("Print first 5 positions of the downsampled point cloud.")
    #print(positions[:5], "\n")
    #print("Convert positions tensor into numpy array.")
    #new_positions_np = positions.numpy()
    #print(new_positions_np[:5])
    #print(new_positions_np)
    #print(downpcd.to_legacy())
    #o3d.visualization.draw_geometries([downpcd.to_legacy()],
    #                                zoom=0.3412,
    #                                front=[0.4257, -0.2125, -0.8795],
    #                                lookat=[2.6172, 2.0475, 1.532],
    #                                up=[-0.0694, -0.9768, 0.2024],
    #                                point_show_normal=True)
    FILE_NUMBER = count_files_with_substring(f'./Positional_data/{VALIDATION_BASE_PATH}Simulated', FILENAME)
    df = pd.DataFrame(new_positions_np[1:])
    df = pd.concat([df, pd.DataFrame(new_normals_np[1:])], axis=1)
    df.columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z']
    df['predicted_class'] = i%3
    if FILE_NUMBER > 0:

        df.to_csv(f'./Positional_data/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_simulated_positions ({FILE_NUMBER}).csv', index=False)

    else:

        df.to_csv(f'./Positional_data/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_simulated_positions.csv', index=False)
        
    #point_cloud = pv.PolyData(point_cloud)

    #FILE_NUMBER = count_files_with_substring('./3D_meshes/Simulated', FILENAME)

    #new_pcd = pv.PolyData(new_positions_np[1:])
    #surf = new_pcd.delaunay_2d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_positions_np[1:])
    pcd.normals = o3d.utility.Vector3dVector(new_normals_np[1:])
    print('run Poisson surface reconstruction')
    
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    #print(mesh)
    #o3d.visualization.draw_geometries([mesh],
    #                                  zoom=0.664,
    #                                  front=[-0.4761, -0.4698, -0.7434],
    #                                  lookat=[1.8900, 3.2596, 0.9284],
    #                                  up=[0.2304, -0.8825, 0.4101])
    mesh.compute_vertex_normals()
    if FILE_NUMBER > 0:

        o3d.io.write_triangle_mesh(f"./3D_meshes/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_mesh_simulated ({FILE_NUMBER}).stl", mesh)

    else:   

        o3d.io.write_triangle_mesh(f"./3D_meshes/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_mesh_simulated.stl", mesh) 
    

