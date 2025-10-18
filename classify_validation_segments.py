#!/usr/bin/env python3
"""
Validation Segment Classifier
Uses K-means and KNN to classify validation segments based on roughness features
Classifies segments into 3 or 4 quality classes (0-2 or 0-3) based on surface roughness characteristics
"""

import os
import numpy as np
import pandas as pd
import glob
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

import shutil
from datetime import datetime

# Configuration
NUM_CLASSES = 3  # Change this to 4 for 4-class classification

def load_point_cloud(file_path):
    """Load point cloud data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data.values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def longest_diagonal(points, axis=2):
    """Compute the longest diagonal across a hollow cylindrical structure"""
    if len(points) < 2:
        return 0.0
    
    # Sort points by their z-values
    sorted_points = points[np.argsort(points[:, axis])]
    
    # Select the lowest point
    min_z_point = sorted_points[0]
    
    # Find the point with maximum radial distance (opposite side)
    radial_distances = np.linalg.norm(points[:, :2] - min_z_point[:2], axis=1)
    max_radial_idx = np.argmax(radial_distances)
    max_radial_point = points[max_radial_idx]
    
    # Compute Euclidean distance
    longest_diagonal = np.linalg.norm(max_radial_point - min_z_point)
    return longest_diagonal

def thickness_analysis(points, num_points=2048):
    """Compute thickness statistics of the structure"""
    if len(points) < 10:
        return 0.0, 0.0
    
    # Sort points by z-value
    sorted_points = points[np.argsort(points[:, 2])[:min(num_points, len(points))]]
    
    thickness_values = []
    z_range = np.ptp(sorted_points[:, 2]) * 0.1
    
    # Determine central axis
    center_x, center_y = np.mean(sorted_points[:, :2], axis=0)
    
    # Sample points across z-range
    z_samples = np.linspace(np.min(sorted_points[:, 2]), np.max(sorted_points[:, 2]), 
                           min(20, len(sorted_points)//10))
    
    for z in z_samples:
        z_min = z - z_range / 2
        z_max = z + z_range / 2
        
        z_points = sorted_points[(sorted_points[:, 2] >= z_min) & (sorted_points[:, 2] <= z_max)]
        
        if len(z_points) > 1:
            radii = np.linalg.norm(z_points[:, :2] - np.array([center_x, center_y]), axis=1)
            if len(radii) > 0:
                inner_radius = np.min(radii)
                outer_radius = np.max(radii)
                thickness_values.append(outer_radius - inner_radius)
    
    if len(thickness_values) > 0:
        return np.mean(thickness_values), np.std(thickness_values)
    else:
        return 0.0, 0.0

def fiedler_number(points, num_points=2048, k_neighbors=6):
    """Compute Fiedler number for surface connectivity analysis using k-NN graph"""
    if len(points) < 10:
        return 0.0

    try:
        # Use ALL points in the segment (no sampling)
        all_points = points[:, :3]  # Use only x, y, z coordinates
        
        # For very large point clouds, we might need to sample for computational efficiency
        # but try to use as many points as possible
        if len(all_points) > 5000:
            # Sample points but keep more than the original num_points
            step = len(all_points) // 3000  # Keep ~3000 points
            sampled_points = all_points[::step]
        else:
            sampled_points = all_points

        # Build k-NN graph with adaptive k based on point density
        adaptive_k = min(k_neighbors, len(sampled_points) - 1)
        if adaptive_k < 2:
            return 0.0
            
        nbrs = NearestNeighbors(n_neighbors=adaptive_k, algorithm='auto').fit(sampled_points)
        adjacency = nbrs.kneighbors_graph(mode='connectivity')

        # Compute normalized Laplacian
        L = laplacian(adjacency, normed=True)

        # Compute smallest eigenvalues (Fiedler number is the second-smallest)
        try:
            # Try to get at least 2 eigenvalues, but not more than the matrix size
            k_eigenvals = min(2, L.shape[0] - 1)
            if k_eigenvals < 2:
                return 0.0
                
            eigenvalues, _ = eigsh(L, k=k_eigenvals, which='SM', tol=1e-3)
            eigenvalues = np.sort(eigenvalues)
            return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        except:
            # Fallback to dense computation
            try:
                eigenvalues = np.linalg.eigvalsh(L.toarray())
                eigenvalues = np.sort(eigenvalues)
                return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            except:
                return 0.0

    except ImportError:
        print("Warning: Required libraries not available for Fiedler number calculation.")
        return 0.0
    except Exception as e:
        print(f"Warning: Error computing Fiedler number: {e}")
        return 0.0


def surface_roughness_metrics(points):
    """Compute surface roughness metrics"""
    if len(points) < 10:
        return 0.0, 0.0, 0.0, 0.0
    
    # Extract coordinates
    x = points[:, 0] if points.shape[1] > 0 else np.zeros(len(points))
    y = points[:, 1] if points.shape[1] > 1 else np.zeros(len(points))
    z = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
    
    # Compute radial distances from center
    center_x, center_y = np.median(x), np.median(y)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Roughness metrics
    r_std = np.std(r)
    r_rms = np.sqrt(np.mean(r**2))
    r_ra = np.mean(np.abs(r - np.mean(r)))
    
    # Height variation
    height_std = np.std(z)
    
    return r_std, r_rms, r_ra, height_std

def normal_variation(points):
    """Compute normal vector variation (if normals available)"""
    if points.shape[1] < 6:
        return 0.0
    
    try:
        # Extract normals (assuming columns 3, 4, 5 are normal_x, normal_y, normal_z)
        nx = points[:, 3]
        ny = points[:, 4] 
        nz = points[:, 5]
        
        # Compute normal magnitudes
        normal_magnitudes = np.sqrt(nx**2 + ny**2 + nz**2)
        return np.std(normal_magnitudes)
    except:
        return 0.0

def extract_fiedler_feature(points):
    """Extract only Fiedler number for classification"""
    if points is None or len(points) == 0:
        return np.array([0.0])
    
    fiedler = fiedler_number(points)
    return np.array([fiedler])

def load_all_validation_segments(validation_dir):
    """Load all validation segments and extract features"""
    print(f"Loading validation segments from: {validation_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(validation_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    features_list = []
    file_paths = []
    
    for i, file_path in enumerate(csv_files):
        print(f"Processing {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        
        points = load_point_cloud(file_path)
        if points is not None:
            features = extract_fiedler_feature(points)
            features_list.append(features)
            file_paths.append(file_path)
        else:
            print(f"  Skipped due to loading error")
    
    if len(features_list) == 0:
        print("No valid segments found!")
        return None, None
    
    features_array = np.array(features_list)
    print(f"Extracted features from {len(features_array)} segments")
    
    return features_array, file_paths

def find_optimal_clusters(features, max_clusters=8):
    """Find optimal number of clusters using silhouette analysis"""
    print("Finding optimal number of clusters...")
    
    silhouette_scores = []
    K_range = range(2, min(max_clusters + 1, len(features)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  k={k}: silhouette score = {silhouette_avg:.3f}")
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k

def classify_segments(features, file_paths, output_dir):
    """Classify segments using K-means and organize into folders"""
    print(f"Classifying segments into {NUM_CLASSES} classes...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # For 3-class classification, use reference-based clustering to maintain consistency
    if NUM_CLASSES == 3:
        # Load reference 4-class results to maintain consistency
        reference_path = "Segmented_data/Validation/Original/Input_4_classes_fiedler/classification_results.csv"
        if os.path.exists(reference_path):
            print("Using 4-class results as reference for consistent 3-class mapping...")
            ref_df = pd.read_csv(reference_path)
            
            # Create mapping from 4-class to 3-class based on Fiedler number ranges
            # Map 4-class results to 3-class: 0→0, 1→0, 2→1, 3→2
            ref_fiedler_by_class = {}
            for class_id in range(4):
                class_data = ref_df[ref_df['quality_class'] == class_id]['fiedler_number']
                ref_fiedler_by_class[class_id] = {
                    'min': class_data.min(),
                    'max': class_data.max(),
                    'mean': class_data.mean()
                }
            
            # Define 3-class boundaries based on 4-class reference
            # Class 0: combine 4-class 0 and 1 (lowest Fiedler)
            # Class 1: 4-class 2 (middle Fiedler)  
            # Class 2: 4-class 3 (highest Fiedler)
            boundary_01 = (ref_fiedler_by_class[0]['max'] + ref_fiedler_by_class[1]['max']) / 2
            boundary_12 = (ref_fiedler_by_class[2]['max'] + ref_fiedler_by_class[3]['min']) / 2
            
            print(f"3-class boundaries based on 4-class reference:")
            print(f"  Class 0: Fiedler < {boundary_01:.6f}")
            print(f"  Class 1: {boundary_01:.6f} ≤ Fiedler < {boundary_12:.6f}")
            print(f"  Class 2: Fiedler ≥ {boundary_12:.6f}")
            
            # Assign classes based on Fiedler number boundaries
            cluster_labels = np.zeros(len(features), dtype=int)
            for i, fiedler in enumerate(features[:, 0]):
                if fiedler < boundary_01:
                    cluster_labels[i] = 0
                elif fiedler < boundary_12:
                    cluster_labels[i] = 1
                else:
                    cluster_labels[i] = 2
        else:
            print("Reference 4-class results not found, using standard K-means...")
            # Fallback to standard K-means
            kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10, init='k-means++')
            cluster_labels = kmeans.fit_predict(features_scaled)
    else:
        # For other class counts, use standard K-means
        kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10, init='k-means++')
        cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze cluster characteristics
    print("\nCluster Analysis:")
    for i in range(NUM_CLASSES):
        cluster_mask = cluster_labels == i
        cluster_features = features[cluster_mask]
        
        print(f"\nClass {i} ({np.sum(cluster_mask)} segments):")
        print(f"  Mean Fiedler: {np.mean(cluster_features[:, 0]):.6f}")
        print(f"  Min Fiedler: {np.min(cluster_features[:, 0]):.6f}")
        print(f"  Max Fiedler: {np.max(cluster_features[:, 0]):.6f}")
    
    # Sort clusters by Fiedler number to assign quality classes
    cluster_fiedler = []
    for i in range(NUM_CLASSES):
        cluster_mask = cluster_labels == i
        mean_fiedler = np.mean(features[cluster_mask, 0])  # Fiedler number
        cluster_fiedler.append((i, mean_fiedler))
    
    # Sort by Fiedler number (ascending: 0=good, 1=fair, 2=poor)
    cluster_fiedler.sort(key=lambda x: x[1])
    
    print(f"\nQuality Class Assignment (based on Fiedler number):")
    for quality_class, (cluster_id, fiedler) in enumerate(cluster_fiedler):
        print(f"  Quality Class {quality_class}: Cluster {cluster_id} (Fiedler={fiedler:.6f})")
    
    # Create output directories
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(output_dir, f"Class_{i}")
        os.makedirs(class_dir, exist_ok=True)
    
    # Copy files to appropriate class directories
    print(f"\nCopying files to class directories...")
    for i, (file_path, cluster_id) in enumerate(zip(file_paths, cluster_labels)):
        # Find quality class for this cluster
        quality_class = next(quality for quality, (cid, _) in enumerate(cluster_fiedler) if cid == cluster_id)
        
        # Copy file
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, f"Class_{quality_class}", filename)
        shutil.copy2(file_path, dest_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(file_paths)} files")
    
    # Save classification results
    results_df = pd.DataFrame({
        'file_path': [os.path.basename(f) for f in file_paths],
        'cluster_id': cluster_labels,
        'quality_class': [next(quality for quality, (cid, _) in enumerate(cluster_fiedler) if cid == cluster_id) 
                         for cluster_id in cluster_labels],
        'fiedler_number': features[:, 0]
    })
    
    results_path = os.path.join(output_dir, 'classification_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Classification results saved to: {results_path}")
    
    return cluster_labels, cluster_fiedler

def update_predicted_class_in_folders(output_dir):
    """Update predicted_class column in all CSV files to match folder number (0 to NUM_CLASSES-1)"""
    print(f"\nUpdating predicted_class column in all CSV files...")
    
    for class_id in range(NUM_CLASSES):
        class_folder = os.path.join(output_dir, f"Class_{class_id}")
        
        if not os.path.exists(class_folder):
            print(f"  Class_{class_id} folder not found, skipping...")
            continue
            
        # Get all CSV files in this class folder
        csv_files = glob.glob(os.path.join(class_folder, "*.csv"))
        print(f"  Processing Class_{class_id}: {len(csv_files)} files")
        
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Update predicted_class column to match folder number
                df['predicted_class'] = class_id
                
                # Save back to the same file
                df.to_csv(csv_file, index=False)
                
            except Exception as e:
                print(f"    Error updating {os.path.basename(csv_file)}: {e}")
    
    print("  Finished updating predicted_class columns")

def generate_classification_summary(results_df, output_dir):
    """Generate comprehensive summary of classification results"""
    print(f"\nGenerating classification summary...")
    
    # Create summary report
    summary_lines = []
    summary_lines.append("CLASSIFICATION SUMMARY REPORT")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total segments classified: {len(results_df)}")
    summary_lines.append(f"Number of classes: {NUM_CLASSES}")
    summary_lines.append("")
    
    # Class distribution
    class_counts = results_df['quality_class'].value_counts().sort_index()
    summary_lines.append("CLASS DISTRIBUTION:")
    for class_id in sorted(class_counts.index):
        count = class_counts[class_id]
        percentage = (count / len(results_df)) * 100
        summary_lines.append(f"  Class {class_id}: {count} segments ({percentage:.1f}%)")
    summary_lines.append("")
    
    # Fiedler number statistics
    fiedler_stats = results_df['fiedler_number'].describe()
    summary_lines.append("FIEDLER NUMBER STATISTICS:")
    summary_lines.append(f"  Min: {fiedler_stats['min']:.6f}")
    summary_lines.append(f"  Max: {fiedler_stats['max']:.6f}")
    summary_lines.append(f"  Mean: {fiedler_stats['mean']:.6f}")
    summary_lines.append(f"  Std: {fiedler_stats['std']:.6f}")
    summary_lines.append(f"  25%: {fiedler_stats['25%']:.6f}")
    summary_lines.append(f"  50%: {fiedler_stats['50%']:.6f}")
    summary_lines.append(f"  75%: {fiedler_stats['75%']:.6f}")
    summary_lines.append("")
    
    # Fiedler statistics by class
    summary_lines.append("FIEDLER NUMBER BY CLASS:")
    for class_id in sorted(results_df['quality_class'].unique()):
        class_data = results_df[results_df['quality_class'] == class_id]['fiedler_number']
        summary_lines.append(f"  Class {class_id} ({len(class_data)} segments):")
        summary_lines.append(f"    Min: {class_data.min():.6f}")
        summary_lines.append(f"    Max: {class_data.max():.6f}")
        summary_lines.append(f"    Mean: {class_data.mean():.6f}")
        summary_lines.append(f"    Std: {class_data.std():.6f}")
        summary_lines.append(f"    Median: {class_data.median():.6f}")
    summary_lines.append("")
    
    # Model-specific analysis
    results_df['model'] = results_df['file_path'].str.extract(r'(Real_Hollow_Cylinder_30mm_\d{4}_\d{2}_\d{2}_v2)')
    models = results_df['model'].value_counts().index.tolist()
    
    summary_lines.append("MODEL-SPECIFIC ANALYSIS:")
    for model in models:
        model_data = results_df[results_df['model'] == model]
        summary_lines.append(f"  {model}: {len(model_data)} segments")
        
        # Class distribution for this model
        class_counts_model = model_data['quality_class'].value_counts().sort_index()
        for class_id in sorted(class_counts_model.index):
            count = class_counts_model[class_id]
            percentage = (count / len(model_data)) * 100
            summary_lines.append(f"    Class {class_id}: {count} segments ({percentage:.1f}%)")
        
        # Fiedler range for this model
        fiedler_min = model_data['fiedler_number'].min()
        fiedler_max = model_data['fiedler_number'].max()
        fiedler_mean = model_data['fiedler_number'].mean()
        summary_lines.append(f"    Fiedler range: {fiedler_min:.6f} to {fiedler_max:.6f}")
        summary_lines.append(f"    Fiedler mean: {fiedler_mean:.6f}")
        summary_lines.append("")
    
    # Classification quality analysis
    class_fiedler_means = results_df.groupby('quality_class')['fiedler_number'].mean().sort_index()
    summary_lines.append("CLASSIFICATION QUALITY:")
    summary_lines.append("  Mean Fiedler Number by Class:")
    for class_id, mean_fiedler in class_fiedler_means.items():
        summary_lines.append(f"    Class {class_id}: {mean_fiedler:.6f}")
    
    # Check if classes are properly ordered
    is_properly_ordered = all(class_fiedler_means.iloc[i] <= class_fiedler_means.iloc[i+1] 
                             for i in range(len(class_fiedler_means)-1))
    if is_properly_ordered:
        summary_lines.append("  Status: Classes properly ordered (Lower Fiedler = Better Quality)")
    else:
        summary_lines.append("  Status: Classes not properly ordered")
    
    # File listing by class
    summary_lines.append("")
    summary_lines.append("FILES BY CLASS:")
    for class_id in sorted(results_df['quality_class'].unique()):
        class_files = results_df[results_df['quality_class'] == class_id]['file_path'].tolist()
        summary_lines.append(f"  Class {class_id} ({len(class_files)} files):")
        for file_name in sorted(class_files):
            summary_lines.append(f"    {file_name}")
        summary_lines.append("")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'classification_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Classification summary saved to: {summary_path}")
    
    # Print summary to console
    for line in summary_lines:
        print(line)

def analyze_model_classification(results_df, output_dir):
    """Analyze classification results by model and generate summary report"""
    
    # Define the models to analyze
    models = [
        'Real_Hollow_Cylinder_30mm_2025_02_13_v2',
        'Real_Hollow_Cylinder_30mm_2025_02_20_v2', 
        'Real_Hollow_Cylinder_30mm_2025_02_22_v2',
        'Real_Hollow_Cylinder_30mm_2025_02_26_v2'
    ]
    
    # Create quality class descriptions based on NUM_CLASSES
    if NUM_CLASSES == 3:
        quality_desc = "Quality Classes: 0=Excellent, 1=Good, 2=Poor"
    elif NUM_CLASSES == 4:
        quality_desc = "Quality Classes: 0=Excellent, 1=Good, 2=Fair, 3=Poor"
    else:
        quality_desc = f"Quality Classes: 0=Excellent, {NUM_CLASSES-1}=Poor (intermediate classes)"
    
    # Create summary report
    summary_lines = []
    summary_lines.append("=== VALIDATION SEGMENT CLASSIFICATION SUMMARY ===")
    summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("Classification based on: Fiedler Number (surface connectivity)")
    summary_lines.append(quality_desc)
    summary_lines.append("")
    
    # Analyze each model
    total_segments = len(results_df)
    summary_lines.append(f"Total segments analyzed: {total_segments}")
    summary_lines.append("")
    
    for model in models:
        # Find segments belonging to this model
        model_segments = results_df[results_df['file_path'].str.contains(model, na=False)]
        
        if len(model_segments) == 0:
            summary_lines.append(f"Model: {model}")
            summary_lines.append("  No segments found for this model")
            summary_lines.append("")
            continue
            
        # Count segments per class
        class_counts = model_segments['quality_class'].value_counts().sort_index()
        
        summary_lines.append(f"Model: {model}")
        summary_lines.append(f"  Total segments: {len(model_segments)}")
        
        for class_id in range(NUM_CLASSES):
            count = class_counts.get(class_id, 0)
            percentage = (count / len(model_segments)) * 100 if len(model_segments) > 0 else 0
            summary_lines.append(f"  Class {class_id}: {count} segments ({percentage:.1f}%)")
        
        # Fiedler number statistics for this model
        fiedler_stats = model_segments['fiedler_number'].describe()
        summary_lines.append(f"  Fiedler number range: {fiedler_stats['min']:.6f} - {fiedler_stats['max']:.6f}")
        summary_lines.append(f"  Fiedler number mean: {fiedler_stats['mean']:.6f}")
        summary_lines.append("")
    
    # Overall statistics
    summary_lines.append("=== OVERALL STATISTICS ===")
    overall_class_counts = results_df['quality_class'].value_counts().sort_index()
    for class_id in range(NUM_CLASSES):
        count = overall_class_counts.get(class_id, 0)
        percentage = (count / total_segments) * 100 if total_segments > 0 else 0
        summary_lines.append(f"Class {class_id}: {count} segments ({percentage:.1f}%)")
    
    # Fiedler number distribution
    fiedler_stats = results_df['fiedler_number'].describe()
    summary_lines.append("")
    summary_lines.append("Fiedler Number Distribution:")
    summary_lines.append(f"  Min: {fiedler_stats['min']:.6f}")
    summary_lines.append(f"  Max: {fiedler_stats['max']:.6f}")
    summary_lines.append(f"  Mean: {fiedler_stats['mean']:.6f}")
    summary_lines.append(f"  Std: {fiedler_stats['std']:.6f}")
    summary_lines.append(f"  25%: {fiedler_stats['25%']:.6f}")
    summary_lines.append(f"  50%: {fiedler_stats['50%']:.6f}")
    summary_lines.append(f"  75%: {fiedler_stats['75%']:.6f}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'classification_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Classification summary saved to: {summary_path}")
    
    # Print summary to console
    for line in summary_lines:
        print(line)

def main():
    print("=== VALIDATION SEGMENT CLASSIFIER ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Classify validation segments based on Fiedler number')
    parser.add_argument('--validation_dir', type=str, 
                       default=r"C:\Users\jvara\Documents\Tese_Mestrado_Backup\Digital_Twin_3D_Meshes\Segmented_data\Validation\Original",
                       help='Directory containing validation CSV files')
    parser.add_argument('--output_dir', type=str, 
                       default=r"C:\Users\jvara\Documents\Tese_Mestrado_Backup\Digital_Twin_3D_Meshes\Segmented_data\Validation\Original\Input_3_classes",
                       help='Output directory for classified segments')
    
    args = parser.parse_args()
    
    validation_dir = args.validation_dir
    output_dir = args.output_dir
    
    print(f"Using {NUM_CLASSES} classes for classification")
    print(f"Validation directory: {validation_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and extract features from all validation segments
    features, file_paths = load_all_validation_segments(validation_dir)
    
    if features is None:
        print("Failed to load validation segments!")
        return
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Features: [Fiedler Number]")
    
    # Classify segments
    cluster_labels, cluster_fiedler = classify_segments(features, file_paths, output_dir)
    
    # Update predicted_class column in all CSV files to match folder numbers
    update_predicted_class_in_folders(output_dir)
    
    # Load results for analysis
    results_path = os.path.join(output_dir, 'classification_results.csv')
    results_df = pd.read_csv(results_path)
    
    # Generate comprehensive classification summary
    generate_classification_summary(results_df, output_dir)
    
    # Generate model-specific analysis and summary
    analyze_model_classification(results_df, output_dir)
    
    print(f"\n=== CLASSIFICATION COMPLETE ===")
    print(f"Classified {len(file_paths)} segments into {NUM_CLASSES} quality classes")
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    for i in range(NUM_CLASSES):
        class_count = np.sum([1 for _, (cid, _) in enumerate(cluster_fiedler) if cid == cluster_labels[i]])
        print(f"Class {i}: {class_count} segments")

if __name__ == "__main__":
    main()
