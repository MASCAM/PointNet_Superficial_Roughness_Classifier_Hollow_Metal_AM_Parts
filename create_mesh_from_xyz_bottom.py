import pandas as pd
import open3d as o3d
import numpy as np
from numpy import genfromtxt
import random
import os
import time

FILENAME = 'Real_Hollow_Cylinder_30mm_2025_02_22_v2'
FILE_NUMBER = 0
CLASS_DEFECT_CHANCES = [[5, 20, 0.25], [20, 30, 0.25], [30, 50, 0.25], [40, 60, 0.6], [60, 80, 0.8]]
CLASS_NAMES = ['Excelent', 'Good', 'Fair', 'Poor', 'Bad']
BATCH_SIZE = 1000
VALIDATION_BASE_PATH = 'Validation/'

# Create necessary directories
os.makedirs(f'./Positional_data/{VALIDATION_BASE_PATH}Original', exist_ok=True)
os.makedirs(f'./3D_meshes/{VALIDATION_BASE_PATH}Simulated', exist_ok=True)

def generate_uniform_solid(pcd):
    # Get points and find z range
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    z_range = max_z - min_z
    
    # Calculate the height threshold (only fill bottom 10%)
    z_threshold = min_z + (z_range * 0.1)  # Only fill bottom 10%
    
    # Find points at the minimum z level
    bottom_points = points[z_values == min_z]
    
    # Generate new points
    new_points = []
    for point in bottom_points:
        x, y, z = point
        # Generate points only up to z_threshold instead of max_z
        z_steps = np.linspace(min_z, z_threshold, int((z_threshold - min_z) * 100) + 1)
        for new_z in z_steps:
            new_points.append([x, y, new_z])
    
    # Combine original and new points
    if new_points:
        new_points = np.array(new_points)
        all_points = np.vstack((points, new_points))
    else:
        all_points = points
    
    # Create new point cloud
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    return filled_pcd

def align_point_cloud_with_z(pcd):
    """
    Rotate point cloud to align the main axis with Z axis and flip Z to correct orientation.
    Args:
        pcd: open3d point cloud object
    Returns:
        aligned_pcd: rotated point cloud aligned with Z axis
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Create rotation matrix to swap Y and Z axes and flip Z
    rotation_matrix = np.array([
        [1, 0, 0],    # X stays the same
        [0, 0, -1],   # Y becomes -Z (flip Z)
        [0, 1, 0]     # Z becomes Y
    ])
    
    # Apply rotation to points and normals
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_normals = np.dot(normals, rotation_matrix.T)
    
    # Create new point cloud with rotated points
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(rotated_points)
    aligned_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)
    
    return aligned_pcd

def normalize_z_position(pcd):
    """
    Normalize point cloud so Z starts from 0.
    Args:
        pcd: open3d point cloud object
    Returns:
        normalized_pcd: point cloud with Z starting from 0
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Find minimum Z value
    min_z = np.min(points[:, 2])
    
    # Subtract minimum Z from all Z coordinates
    points[:, 2] = points[:, 2] - min_z
    
    # Create new point cloud with normalized points
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(points)
    normalized_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return normalized_pcd

# Replace 'path_to_your_file.xyz' with the actual file path
original_mesh = o3d.io.read_triangle_mesh(f"./3D_meshes/{VALIDATION_BASE_PATH}Original/{FILENAME}.stl")

# First convert mesh to point cloud with uniform sampling - increased number of points
pcd = original_mesh.sample_points_uniformly(20480)

# Fill the point cloud with additional points
filled_pcd = generate_uniform_solid(pcd)

# Print initial point count
print(f"Number of points after filling: {len(np.asarray(filled_pcd.points))}")

# Visualize the filled point cloud
o3d.visualization.draw_geometries([filled_pcd])

# Careful normal estimation with optimized parameters for 120mm diameter model
radius = 2.0  # Radius in mm for 120mm diameter model (2% of diameter)
max_nn = 50   # Good number of neighbors for normal estimation
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
filled_pcd.estimate_normals(search_param=search_param)
filled_pcd.orient_normals_consistent_tangent_plane(100)

# Create mesh using Ball Pivoting Algorithm (BPA) for better detail preservation
# Calculate appropriate radii for 120mm diameter, 30mm height model
# Scale: ~120mm diameter = ~60mm radius, so ball radii should be in mm range
# BPA preserves sharp features and surface details better than Poisson
print("Starting Ball Pivoting Algorithm reconstruction...")
print(f"Point cloud size: {len(filled_pcd.points)} points")
print(f"Point cloud bounds: {filled_pcd.get_axis_aligned_bounding_box()}")

radii = [0.5, 1.0, 2.0, 4.0, 8.0]  # Ball radii in mm (0.5mm to 8mm)
print(f"Using ball radii: {radii}")

try:
    print("Creating mesh with Ball Pivoting Algorithm...")
    start_time = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        filled_pcd, o3d.utility.DoubleVector(radii))
    end_time = time.time()
    print(f"BPA reconstruction successful! Generated {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    print(f"BPA reconstruction took {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"Ball Pivoting Algorithm failed: {e}")
    print("Falling back to Alpha Shapes reconstruction...")
    # Fallback to Alpha Shapes if BPA fails
    alpha = 5.0  # Alpha value in mm for 120mm diameter model
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            filled_pcd, alpha)
        print(f"Alpha shape reconstruction successful with alpha={alpha}")
    except Exception as e2:
        print(f"Alpha shape also failed: {e2}")
        print("Falling back to Poisson reconstruction...")
        # Final fallback to Poisson
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            filled_pcd, depth=8)
        print("Poisson reconstruction completed as fallback")

# Visualize normals
print("Visualizing mesh normals (blue lines show normal directions)")
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
normals_visualization = o3d.geometry.LineSet()

# Sample points and normals from the mesh for visualization
points = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)

# Sample a subset of points for clearer visualization
sample_size = min(1000, len(points))  # Limit to 1000 points for clarity
indices = np.random.choice(len(points), sample_size, replace=False)
sampled_points = points[indices]
sampled_normals = normals[indices]

# Create lines representing normals
normal_length = 0.02  # Adjust this value to change normal vector length
line_points = []
line_indices = []
for i, (point, normal) in enumerate(zip(sampled_points, sampled_normals)):
    line_points.append(point)
    line_points.append(point + normal * normal_length)
    line_indices.append([i*2, i*2+1])

normals_visualization.points = o3d.utility.Vector3dVector(line_points)
normals_visualization.lines = o3d.utility.Vector2iVector(line_indices)
normals_visualization.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in line_indices])  # Blue lines

# Visualize mesh with normals
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Mesh with Normal Vectors")
vis.add_geometry(mesh)
vis.add_geometry(normals_visualization)
vis.add_geometry(mesh_frame)

# Improve visualization settings
opt = vis.get_render_option()
opt.mesh_show_back_face = True
opt.background_color = np.array([0.8, 0.8, 0.8])
opt.point_size = 1.0
opt.line_width = 2.0  # Make normal lines more visible

# Run visualization
vis.run()
vis.destroy_window()

# Compute normals before saving
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# Align and normalize point cloud
downpcd = align_point_cloud_with_z(mesh.sample_points_uniformly(20480))
#downpcd = mesh.sample_points_uniformly(49152)
downpcd = normalize_z_position(downpcd)

radius = 1.0  # Radius in mm for downsampled point cloud
max_nn = 30   # Good number of neighbors for normal estimation
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
# Re-estimate normals after downsampling for better quality
downpcd.estimate_normals(search_param=search_param)
downpcd.orient_normals_consistent_tangent_plane(30)

# Visualize the downsampled point cloud
o3d.visualization.draw_geometries([downpcd])

# Create final mesh using Ball Pivoting Algorithm for detail preservation
# Use smaller radii for the downsampled point cloud to capture fine details
# These radii are optimized for the final point cloud density and model scale
print("\nStarting final Ball Pivoting Algorithm reconstruction...")
print(f"Downsampled point cloud size: {len(downpcd.points)} points")
print(f"Downsampled point cloud bounds: {downpcd.get_axis_aligned_bounding_box()}")

final_radii = [0.2, 0.5, 1.0, 2.0, 4.0]  # Smaller radii for fine details
print(f"Using final ball radii: {final_radii}")

try:
    print("Creating final mesh with Ball Pivoting Algorithm...")
    start_time = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        downpcd, o3d.utility.DoubleVector(final_radii))
    end_time = time.time()
    print(f"Final BPA reconstruction successful! Generated {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    print(f"Final BPA reconstruction took {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"Final Ball Pivoting Algorithm failed: {e}")
    print("Falling back to Alpha Shapes for final reconstruction...")
    # Fallback to Alpha Shapes if BPA fails
    final_alpha = 2.0  # Smaller alpha for fine details
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            downpcd, final_alpha)
        print(f"Final alpha shape reconstruction successful with alpha={final_alpha}")
    except Exception as e2:
        print(f"Final alpha shape also failed: {e2}")
        print("Falling back to Poisson for final reconstruction...")
        # Final fallback to Poisson
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            downpcd, depth=8)
        print("Final Poisson reconstruction completed as fallback")

mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

o3d.io.write_triangle_mesh(f"./3D_meshes/{VALIDATION_BASE_PATH}Simulated/{FILENAME}_mesh_simulated ({FILE_NUMBER}).stl", mesh)
o3d.visualization.draw_geometries([mesh])

positions_np = np.asarray(downpcd.points)
normals_np = np.asarray(downpcd.normals)

# Add visualization of point cloud with coordinate frame
print("Visualizing final point cloud with coordinate axes")
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud with Coordinate Axes")
vis.add_geometry(downpcd)
vis.add_geometry(coordinate_frame)


# Add text labels for axes
print("Coordinate system:")
print("Red axis: X")
print("Green axis: Y")
print("Blue axis: Z")

# Run visualization
vis.run()
vis.destroy_window()

# Continue with CSV saving
df = pd.DataFrame(positions_np[1:]*10)
df = pd.concat([df, pd.DataFrame(normals_np[1:])], axis=1)
df.columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z']
df.to_csv(f'./Positional_data/{VALIDATION_BASE_PATH}Original/{FILENAME}_positions.csv', index=False) 