import os
import numpy as np
import h5py
import glob
import re
import pandas as pd
import open3d as o3d
import random
from scipy.spatial.transform import Rotation as R

# Constants
NUM_POINT = 2048
VALIDATION_BASE_PATH = 'Validation/'
INPUT_DIR = f'Segmented_data/{VALIDATION_BASE_PATH}Original/Input_3_classes/'
OUTPUT_DIR = f'Pre_Processed_data/3_classes/{VALIDATION_BASE_PATH}Mixed/HDF5/'

# Global variables for flexible class support
NUM_CLASSES = 3  # Will be auto-detected

# Augmentation parameters - FLEXIBLE CLASS SUPPORT
DEFAULT_CLASS_AUGMENTATION_FACTORS = {
    0: 8,   # Class 0: REDUCED augmentation
    1: 80,  # Class 1: MASSIVELY INCREASED augmentation  
    2: 100, # Class 2: MASSIVELY INCREASED augmentation
    3: 8    # Class 3: REDUCED augmentation
}

THREE_CLASS_AUGMENTATION_FACTORS = {
    0: 36,   # Class 0: REDUCED augmentation
    1: 8,  # Class 1: MASSIVELY INCREASED augmentation
    2: 90  # Class 2: MASSIVELY INCREASED augmentation
}

ROTATION_RANGE = 30  # Degrees
SCALE_RANGE = (0.9, 1.1)  # Scale factor range
JITTER_STD = 0.01  # Standard deviation for position jittering

DEFAULT_CLASS_NAMES = ['Excellent', 'Good', 'Fair', 'Poor']
THREE_CLASS_NAMES = ['Good', 'Fair', 'Poor']

def detect_number_of_classes(csv_files):
    """Auto-detect the number of classes from CSV files"""
    print(f"Auto-detecting number of classes from {len(csv_files)} CSV files...")
    
    detected_labels = set()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'predicted_class' in df.columns:
                label = df['predicted_class'].iloc[0]
                if isinstance(label, (int, np.integer)) and 0 <= label < 10:
                    detected_labels.add(int(label))
        except Exception as e:
            print(f"Warning: Could not read predicted_class from {csv_file}: {e}")
    
    if detected_labels:
        num_classes = len(detected_labels)
        sorted_labels = sorted(detected_labels)
        print(f"Detected {num_classes} classes: {sorted_labels}")
        return num_classes
    else:
        print("Warning: Could not detect classes from CSV files, defaulting to 3 classes")
        return NUM_CLASSES

def save_h5_data_label_normal_metrics(h5_filename, data, label, normal, metrics,
        data_dtype='float32', label_dtype='uint8', normal_dtype='float32', metrics_dtype='float32'):
    """Save data to HDF5 format with compression including metrics channel"""
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('normal', data=normal, compression='gzip', compression_opts=4, dtype=normal_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.create_dataset('metrics', data=metrics, compression='gzip', compression_opts=4, dtype=metrics_dtype)
    h5_fout.close()

def save_h5_data_label_normal(h5_filename, data, label, normal,
        data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    """Save data to HDF5 format with compression without metrics channel"""
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('normal', data=normal, compression='gzip', compression_opts=4, dtype=normal_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()

def voxel_downsample(points, voxel_size):
    """Reduce point cloud using voxelization"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)

def detect_and_normalize_scale(points, normals):
    """Detect the scale of the data and normalize to consistent units"""
    point_ranges = np.ptp(points, axis=0)
    mean_range = np.mean(point_ranges)
    max_range = np.max(point_ranges)
    
    print(f"  Scale detection - Point ranges: X={point_ranges[0]:.6f}, Y={point_ranges[1]:.6f}, Z={point_ranges[2]:.6f}")
    print(f"  Scale detection - Mean range: {mean_range:.6f}, Max: {max_range:.6f}")
    
    # Scale detection logic
    if mean_range < 0.05 and max_range < 0.1:
        print("  Detected scale: METERS (converting to millimeters)")
        scale_factor = 1000.0
        scale_type = "meters_to_mm"
    elif 0.1 <= mean_range <= 1.0 and max_range <= 2.0:
        print("  Detected scale: DECIMETERS (converting to millimeters)")
        scale_factor = 100.0
        scale_type = "dm_to_mm"
    elif 0.5 < mean_range < 5.0 and max_range < 10.0:
        print("  Detected scale: CENTIMETERS (converting to millimeters)")
        scale_factor = 10.0
        scale_type = "cm_to_mm"
    elif mean_range > 10.0 or max_range > 20.0:
        print("  Detected scale: MILLIMETERS (keeping as is)")
        scale_factor = 1.0
        scale_type = "millimeters"
    else:
        print("  Assuming MILLIMETERS (default)")
        scale_factor = 1.0
        scale_type = "ambiguous"
    
    # Apply scale normalization
    normalized_points = points * scale_factor
    normalized_normals = normals  # Normals are unit vectors, no scale change needed
    
    print(f"  Scale factor applied: {scale_factor}")
    return normalized_points, normalized_normals, scale_type

def apply_data_augmentation(points, normals, metrics, augmentation_id):
    """Apply data augmentation: rotation, scaling, and jittering"""
    random.seed(augmentation_id * 42)
    np.random.seed(augmentation_id * 42)
    
    augmented_points = points.copy()
    augmented_normals = normals.copy()
    augmented_metrics = metrics.copy()
    
    # 1. Rotation augmentation
    if random.random() < 0.8:
        angle_z = random.uniform(-ROTATION_RANGE, ROTATION_RANGE) * np.pi / 180
        rotation_z = R.from_euler('z', angle_z)
        augmented_points = rotation_z.apply(augmented_points)
        augmented_normals = rotation_z.apply(augmented_normals)
        
        if random.random() < 0.3:
            angle_x = random.uniform(-10, 10) * np.pi / 180
            angle_y = random.uniform(-10, 10) * np.pi / 180
            rotation_xy = R.from_euler('xyz', [angle_x, angle_y, 0])
            augmented_points = rotation_xy.apply(augmented_points)
            augmented_normals = rotation_xy.apply(augmented_normals)
    
    # 2. Scaling augmentation
    if random.random() < 0.6:
        scale_factor = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
        augmented_points *= scale_factor
        augmented_metrics[:, 0] *= scale_factor
    
    # 3. Jittering augmentation
    if random.random() < 0.7:
        jitter = np.random.normal(0, JITTER_STD, augmented_points.shape)
        augmented_points += jitter
    
    return augmented_points, augmented_normals, augmented_metrics

def process_csv_to_h5_original(csv_file, h5_file, num_points):
    """Process a single CSV file WITHOUT augmentation and save to HDF5"""
    df = pd.read_csv(csv_file)
    
    # Check for basic required columns
    basic_required_columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z', 'predicted_class']
    missing_basic_columns = [col for col in basic_required_columns if col not in df.columns]
    
    if missing_basic_columns:
        print(f"Warning: Missing basic columns {missing_basic_columns} in {csv_file}, skipping...")
        return False, None
    
    # Check if metrics columns exist
    has_metrics = 'diameter' in df.columns and 'height' in df.columns
    
    # Extract points, normals, and label
    points = df[['pos_x', 'pos_y', 'pos_z']].values
    normals = df[['normal_x', 'normal_y', 'normal_z']].values
    label = df['predicted_class'].iloc[0]
    
    # Verify label is within expected range
    expected_labels = list(range(NUM_CLASSES))
    if label not in expected_labels:
        print(f"Warning: Label {label} is outside expected range {expected_labels} in {csv_file}")
        return False, None
    
    # Create label array with shape (2048, 1)
    labels = np.full((num_points, 1), label, dtype=np.uint8)
    
    # Create metrics array only if diameter and height columns exist
    if has_metrics:
        diameter = df['diameter'].values
        height = df['height'].values
        metrics = np.column_stack([diameter, height])
    else:
        metrics = None
    
    # Process point cloud to have exactly num_points points
    if len(points) > num_points:
        voxel_size = 0.02
        while len(points) > num_points:
            points = voxel_downsample(points, voxel_size)
            normals = normals[:len(points)]
            if metrics is not None:
                metrics = metrics[:len(points)]
            voxel_size *= 1.1
            
        if len(points) > num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
            normals = normals[idx]
            if metrics is not None:
                metrics = metrics[idx]
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
        points = points[idx]
        normals = normals[idx]
        if metrics is not None:
            metrics = metrics[idx]
    
    # Scale normalization
    print(f"Processing {os.path.basename(csv_file)} - Scale normalization...")
    normalized_points, normalized_normals, scale_type = detect_and_normalize_scale(points, normals)
    
    # Unit sphere normalization
    print(f"Processing {os.path.basename(csv_file)} - Unit sphere normalization...")
    centroid = np.mean(normalized_points, axis=0)
    normalized_points = normalized_points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(normalized_points)**2, axis=1)))
    normalized_points = normalized_points / furthest_distance
    
    # Normalize normals to unit length
    normalized_normals = normalized_normals / np.linalg.norm(normalized_normals, axis=1, keepdims=True)
    
    # Normalize metrics to unit sphere scale (only if metrics exist)
    if metrics is not None:
        scale_factor = furthest_distance
        normalized_metrics = metrics.copy()
        normalized_metrics[:, 0] = normalized_metrics[:, 0] / scale_factor
        
        save_h5_data_label_normal_metrics(h5_file, normalized_points, labels, normalized_normals, normalized_metrics)
    else:
        save_h5_data_label_normal(h5_file, normalized_points, labels, normalized_normals)
    
    return True, scale_type

def process_csv_to_h5_augmented(csv_file, h5_file, num_points, augmentation_id, total_augmentations):
    """Process a single CSV file with augmentation and save to HDF5"""
    df = pd.read_csv(csv_file)
    
    # Check for basic required columns
    basic_required_columns = ['pos_x', 'pos_y', 'pos_z', 'normal_x', 'normal_y', 'normal_z', 'predicted_class']
    missing_basic_columns = [col for col in basic_required_columns if col not in df.columns]
    
    if missing_basic_columns:
        print(f"Warning: Missing basic columns {missing_basic_columns} in {csv_file}, skipping...")
        return False, None
    
    # Check if metrics columns exist
    has_metrics = 'diameter' in df.columns and 'height' in df.columns
    
    # Extract points, normals, and label
    points = df[['pos_x', 'pos_y', 'pos_z']].values
    normals = df[['normal_x', 'normal_y', 'normal_z']].values
    label = df['predicted_class'].iloc[0]
    
    # Verify label is within expected range
    expected_labels = list(range(NUM_CLASSES))
    if label not in expected_labels:
        print(f"Warning: Label {label} is outside expected range {expected_labels} in {csv_file}")
        return False, None
    
    # Create label array with shape (2048, 1)
    labels = np.full((num_points, 1), label, dtype=np.uint8)
    
    # Create metrics array only if diameter and height columns exist
    if has_metrics:
        diameter = df['diameter'].values
        height = df['height'].values
        metrics = np.column_stack([diameter, height])
    else:
        metrics = None
    
    # Process point cloud to have exactly num_points points
    if len(points) > num_points:
        voxel_size = 0.02
        while len(points) > num_points:
            points = voxel_downsample(points, voxel_size)
            normals = normals[:len(points)]
            if metrics is not None:
                metrics = metrics[:len(points)]
            voxel_size *= 1.1
            
        if len(points) > num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
            normals = normals[idx]
            if metrics is not None:
                metrics = metrics[idx]
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
        points = points[idx]
        normals = normals[idx]
        if metrics is not None:
            metrics = metrics[idx]
    
    # Apply data augmentation
    print(f"  Applying data augmentation {augmentation_id + 1}/{total_augmentations}...")
    if metrics is not None:
        augmented_points, augmented_normals, augmented_metrics = apply_data_augmentation(
            points, normals, metrics, augmentation_id
        )
    else:
        dummy_metrics = np.zeros((len(points), 2), dtype=np.float32)
        augmented_points, augmented_normals, augmented_metrics = apply_data_augmentation(
            points, normals, dummy_metrics, augmentation_id
        )
        augmented_metrics = None
    
    # Scale normalization
    print(f"Processing {os.path.basename(csv_file)} - Scale normalization...")
    normalized_points, normalized_normals, scale_type = detect_and_normalize_scale(
        augmented_points, augmented_normals
    )
    
    # Unit sphere normalization
    print(f"Processing {os.path.basename(csv_file)} - Unit sphere normalization...")
    centroid = np.mean(normalized_points, axis=0)
    normalized_points = normalized_points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(normalized_points)**2, axis=1)))
    normalized_points = normalized_points / furthest_distance
    
    # Normalize normals to unit length
    normalized_normals = normalized_normals / np.linalg.norm(normalized_normals, axis=1, keepdims=True)
    
    # Normalize metrics to unit sphere scale (only if metrics exist)
    if augmented_metrics is not None:
        scale_factor = furthest_distance
        normalized_metrics = augmented_metrics.copy()
        normalized_metrics[:, 0] = normalized_metrics[:, 0] / scale_factor
        
        save_h5_data_label_normal_metrics(h5_file, normalized_points, labels, normalized_normals, normalized_metrics)
    else:
        save_h5_data_label_normal(h5_file, normalized_points, labels, normalized_normals)
    
    return True, scale_type

def unite_h5_files(h5_files, output_file):
    """Unite multiple H5 files into a single unified file"""
    data_list = []
    normals_list = []
    labels_list = []
    metrics_list = []
    has_metrics = None
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f_in:
            data = f_in['data'][:]
            normal = f_in['normal'][:]
            label = f_in['label'][:]
            
            if np.any(np.isnan(data)) or np.any(np.isnan(normal)):
                print(f"Warning: NaN values found in {h5_file}")
                continue
            
            file_has_metrics = 'metrics' in f_in.keys()
            
            if has_metrics is None:
                has_metrics = file_has_metrics
                print(f"Detected metrics channel: {'Yes' if has_metrics else 'No'}")
            
            if has_metrics == file_has_metrics:
                data_list.append(data)
                normals_list.append(normal)
                labels_list.append(label)
                
                if has_metrics:
                    metrics = f_in['metrics'][:]
                    if np.any(np.isnan(metrics)):
                        print(f"Warning: NaN values found in metrics of {h5_file}")
                        continue
                    metrics_list.append(metrics)
            else:
                print(f"Warning: Inconsistent metrics channel in {h5_file}, skipping...")
                continue
    
    combined_data = np.stack(data_list, axis=0)
    combined_normals = np.stack(normals_list, axis=0)
    combined_labels = np.stack(labels_list, axis=0)
    
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('data', data=combined_data)
        f_out.create_dataset('normal', data=combined_normals)
        f_out.create_dataset('label', data=combined_labels)
        
        if has_metrics:
            combined_metrics = np.stack(metrics_list, axis=0)
            f_out.create_dataset('metrics', data=combined_metrics)

def create_balanced_split(csv_files_by_class):
    """Create balanced train/test/validation split for each class"""
    train_files = []
    test_files = []
    validation_files = []
    
    for class_id, files in csv_files_by_class.items():
        num_files = len(files)
        print(f"Class {class_id}: {num_files} files")
        
        # Shuffle files for random distribution
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        # Calculate split sizes (70% train, 20% test, 10% validation)
        train_size = int(num_files * 0.7)
        test_size = int(num_files * 0.2)
        validation_size = num_files - train_size - test_size
        
        print(f"  Train: {train_size}, Test: {test_size}, Validation: {validation_size}")
        
        # Split files
        train_files.extend(shuffled_files[:train_size])
        test_files.extend(shuffled_files[train_size:train_size + test_size])
        validation_files.extend(shuffled_files[train_size + test_size:])
    
    print(f"\nTotal split:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Test: {len(test_files)} files")
    print(f"  Validation: {len(validation_files)} files")
    
    return train_files, test_files, validation_files

def create_unified_mixed_h5_files(input_dir, output_dir):
    """Main function to create mixed train/test/validation datasets"""
    global NUM_CLASSES
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    # Auto-detect number of classes
    NUM_CLASSES = detect_number_of_classes(csv_files)
    print(f"Using {NUM_CLASSES} classes for processing")
    
    # Group files by class
    csv_files_by_class = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'predicted_class' in df.columns:
                file_class = df['predicted_class'].iloc[0]
                if file_class not in csv_files_by_class:
                    csv_files_by_class[file_class] = []
                csv_files_by_class[file_class].append(csv_file)
        except Exception as e:
            print(f"Warning: Could not read predicted_class from {csv_file}: {e}")
    
    # Print class distribution
    print(f"\n=== Class Distribution ===")
    for class_id in sorted(csv_files_by_class.keys()):
        print(f"Class {class_id}: {len(csv_files_by_class[class_id])} files")
    
    # Create balanced split
    print(f"\n=== Creating Balanced Split ===")
    train_files, test_files, validation_files = create_balanced_split(csv_files_by_class)
    
    # Determine class configuration
    if NUM_CLASSES == 3:
        class_names = THREE_CLASS_NAMES
        augmentation_factors = THREE_CLASS_AUGMENTATION_FACTORS
    elif NUM_CLASSES == 4:
        class_names = DEFAULT_CLASS_NAMES
        augmentation_factors = DEFAULT_CLASS_AUGMENTATION_FACTORS
    else:
        class_names = [f'Class_{i}' for i in range(NUM_CLASSES)]
        augmentation_factors = {}
        for i in range(NUM_CLASSES):
            if i == 0 or i == NUM_CLASSES - 1:
                augmentation_factors[i] = 8
            else:
                augmentation_factors[i] = 80
    
    # Update output directory to reflect class count
    if NUM_CLASSES != 3:
        output_dir = output_dir.replace('3_classes', f'{NUM_CLASSES}_classes')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Updated output directory to: {output_dir}")
    
    # Process datasets
    all_h5_files = []
    
    # Process TRAINING dataset (with augmentation)
    print(f"\n=== Processing TRAINING Dataset (with augmentation) ===")
    train_h5_files = []
    for i, csv_file in enumerate(train_files):
        print(f"Processing training file {i+1}/{len(train_files)}: {os.path.basename(csv_file)}")
        
        # Get class from file
        df = pd.read_csv(csv_file)
        file_class = df['predicted_class'].iloc[0]
        augmentation_factor = augmentation_factors[file_class]
        
        # Create augmented versions
        for aug_id in range(augmentation_factor):
            h5_file = os.path.join(output_dir, f'train_{i:04d}_aug_{aug_id:02d}.h5')
            success, scale_type = process_csv_to_h5_augmented(
                csv_file, h5_file, NUM_POINT, aug_id, augmentation_factor
            )
            if success:
                train_h5_files.append(h5_file)
    
    # Process TESTING dataset (with augmentation)
    print(f"\n=== Processing TESTING Dataset (with augmentation) ===")
    test_h5_files = []
    for i, csv_file in enumerate(test_files):
        print(f"Processing testing file {i+1}/{len(test_files)}: {os.path.basename(csv_file)}")
        
        # Get class from file
        df = pd.read_csv(csv_file)
        file_class = df['predicted_class'].iloc[0]
        augmentation_factor = augmentation_factors[file_class]
        
        # Create augmented versions
        for aug_id in range(augmentation_factor):
            h5_file = os.path.join(output_dir, f'test_{i:04d}_aug_{aug_id:02d}.h5')
            success, scale_type = process_csv_to_h5_augmented(
                csv_file, h5_file, NUM_POINT, aug_id, augmentation_factor
            )
            if success:
                test_h5_files.append(h5_file)
    
    # Process VALIDATION dataset (NO augmentation)
    print(f"\n=== Processing VALIDATION Dataset (NO augmentation) ===")
    validation_h5_files = []
    for i, csv_file in enumerate(validation_files):
        print(f"Processing validation file {i+1}/{len(validation_files)}: {os.path.basename(csv_file)}")
        
        h5_file = os.path.join(output_dir, f'validation_{i:04d}.h5')
        success, scale_type = process_csv_to_h5_original(csv_file, h5_file, NUM_POINT)
        if success:
            validation_h5_files.append(h5_file)
    
    # Create unified files
    print(f"\n=== Creating Unified Files ===")
    
    # Train unified file
    train_output = os.path.join(output_dir, 'mixed_train.h5')
    unite_h5_files(train_h5_files, train_output)
    all_h5_files.append(train_output)
    print(f"Created unified training file: {train_output}")
    
    # Test unified file
    test_output = os.path.join(output_dir, 'mixed_test.h5')
    unite_h5_files(test_h5_files, test_output)
    all_h5_files.append(test_output)
    print(f"Created unified testing file: {test_output}")
    
    # Validation unified file
    validation_output = os.path.join(output_dir, 'mixed_validation.h5')
    unite_h5_files(validation_h5_files, validation_output)
    all_h5_files.append(validation_output)
    print(f"Created unified validation file: {validation_output}")
    
    # Write file lists
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        f.write(train_output + '\n')
    
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        f.write(test_output + '\n')
    
    with open(os.path.join(output_dir, 'validation_files.txt'), 'w') as f:
        f.write(validation_output + '\n')
    
    print(f"\n=== Mixed Processing Complete ===")
    print(f"Number of classes detected: {NUM_CLASSES}")
    print(f"Class names: {class_names}")
    print(f"Files saved in: {output_dir}")
    print(f"Augmentation factors: {augmentation_factors}")
    print(f"\nDataset split:")
    print(f"  Training: {len(train_files)} original files -> {len(train_h5_files)} augmented files")
    print(f"  Testing: {len(test_files)} original files -> {len(test_h5_files)} augmented files")
    print(f"  Validation: {len(validation_files)} original files -> {len(validation_h5_files)} files (no augmentation)")

if __name__ == '__main__':
    create_unified_mixed_h5_files(INPUT_DIR, OUTPUT_DIR)
