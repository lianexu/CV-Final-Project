import os  
import numpy as np  
import open3d as o3d  
from tqdm import tqdm  
import glob  
import random  
import shutil  
  
  
# Function to read STL file and convert to point cloud  
def stl_to_point_cloud(stl_file, num_points=2048):  
    # Read the STL file as a mesh  
    mesh = o3d.io.read_triangle_mesh(stl_file)  
      
    # Ensure the mesh has triangles  
    if len(mesh.triangles) == 0:  
        print(f"Warning: {stl_file} has no triangles")  
        return None  
      
    # Sample points uniformly from the mesh surface  
    point_cloud = mesh.sample_points_uniformly(num_points)  
      
    return np.asarray(point_cloud.points)  
  
# Function to normalize point cloud to [-range, range]  
def normalize_point_cloud(point_cloud, target_range=0.35):  
    # Find the max absolute value across all axes  
    max_abs = np.max(np.abs(point_cloud), axis=0)  
      
    # Scale each dimension independently  
    normalized_points = np.zeros_like(point_cloud)  
    normalized_points[:, 0] = point_cloud[:, 0] * (target_range / max_abs[0])  
    normalized_points[:, 1] = point_cloud[:, 1] * (target_range / max_abs[1])  
    normalized_points[:, 2] = point_cloud[:, 2] * (target_range / max_abs[2])  
      
    return normalized_points  
  
# Function to apply random rotation to point cloud  
def random_rotate_point_cloud(point_cloud):  
    # Generate random rotation matrix  
    theta = np.random.uniform(0, 2 * np.pi)  
    phi = np.random.uniform(0, 2 * np.pi)  
    z = np.random.uniform(0, 2 * np.pi)  
      
    # Rotation matrices  
    Rx = np.array([  
        [1, 0, 0],  
        [0, np.cos(theta), -np.sin(theta)],  
        [0, np.sin(theta), np.cos(theta)]  
    ])  
      
    Ry = np.array([  
        [np.cos(phi), 0, np.sin(phi)],  
        [0, 1, 0],  
        [-np.sin(phi), 0, np.cos(phi)]  
    ])  
      
    Rz = np.array([  
        [np.cos(z), -np.sin(z), 0],  
        [np.sin(z), np.cos(z), 0],  
        [0, 0, 1]  
    ])  
      
    # Combined rotation matrix  
    R = np.dot(Rz, np.dot(Ry, Rx))  
      
    # Apply rotation  
    rotated_point_cloud = np.dot(point_cloud, R)  
      
    return rotated_point_cloud  
  
# Function to save point cloud as PLY file  
def save_as_ply(point_cloud, output_file):  
    # Create Open3D point cloud  
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(point_cloud)  
      
    # Save as PLY  
    o3d.io.write_point_cloud(output_file, pcd)  
  
# Main processing function  
def process_stl_files(input_dir, output_dir, num_points=2048, target_range=0.35, augment=True, num_augmentations=3):  
    # Create output directory if it doesn't exist  
    os.makedirs(output_dir, exist_ok=True)  
      
    # Find all STL files  
    stl_files = glob.glob(os.path.join(input_dir, "*.stl"))  
      
    print(f"Found {len(stl_files)} STL files")  
      
    # Process each STL file  
    for stl_file in tqdm(stl_files):  
        # Get base filename without extension  
        base_name = os.path.splitext(os.path.basename(stl_file))[0]  
          
        # Convert STL to point cloud  
        point_cloud = stl_to_point_cloud(stl_file, num_points)  
          
        if point_cloud is None:  
            continue  
          
        # Normalize point cloud  
        normalized_point_cloud = normalize_point_cloud(point_cloud, target_range)  
          
        # Save the normalized point cloud  
        output_file = os.path.join(output_dir, f"{base_name}.ply")  
        save_as_ply(normalized_point_cloud, output_file)  
          
        # Create augmented versions with random rotations  
        if augment:  
            for i in range(num_augmentations):  
                # Apply random rotation  
                rotated_point_cloud = random_rotate_point_cloud(normalized_point_cloud)  
                  
                # Save the rotated point cloud  
                output_file = os.path.join(output_dir, f"{base_name}_aug{i+1}.ply")  
                save_as_ply(rotated_point_cloud, output_file)  
  



# # Example usage  
# if __name__ == "__main__":  
#     input_directory = "femurleft_stl"  # Replace with your input directory  
#     output_directory = "femurleft_ply"  # Replace with your output directory  
      
#     process_stl_files(  
#         input_dir=input_directory,  
#         output_dir=output_directory,  
#         num_points=2048,  # Number of points to sample (required by ShapeInversion)  
#         target_range=0.35,  # Normalization range [-0.35, 0.35]  
#         augment=True,  # Apply random rotations  
#         num_augmentations=2  # Number of augmented versions per file  
#     )


def split_dataset(input_dir, output_base_dir, train_ratio=0.8, class_name="femur"):  
    """  
    Split a folder of .ply files into training and testing sets.  
      
    Args:  
        input_dir: Directory containing .ply files  
        output_base_dir: Base directory for output  
        train_ratio: Ratio of files to use for training (default: 0.8)  
        class_name: Name of the class (default: "femur")  
    """  
    # Create output directories  
    train_dir = os.path.join(output_base_dir, "train", class_name)  
    test_dir = os.path.join(output_base_dir, "test", class_name)  
      
    os.makedirs(train_dir, exist_ok=True)  
    os.makedirs(test_dir, exist_ok=True)  
      
    # Get all .ply files  
    ply_files = glob.glob(os.path.join(input_dir, "*.ply"))  
      
    # Shuffle the files to ensure random split  
    random.shuffle(ply_files)  
      
    # Calculate split indices  
    num_files = len(ply_files)  
    num_train = int(num_files * train_ratio)  
      
    # Split files  
    train_files = ply_files[:num_train]  
    test_files = ply_files[num_train:]  
      
    print(f"Total files: {num_files}")  
    print(f"Training files: {len(train_files)}")  
    print(f"Testing files: {len(test_files)}")  
      
    # Copy files to respective directories  
    print("Copying training files...")  
    for file in tqdm(train_files):  
        filename = os.path.basename(file)  
        shutil.copy(file, os.path.join(train_dir, filename))  
      
    print("Copying testing files...")  
    for file in tqdm(test_files):  
        filename = os.path.basename(file)  
        shutil.copy(file, os.path.join(test_dir, filename))  
      
    print(f"Dataset split complete. Files organized in {output_base_dir}")  
  
# # Example usage  
# if __name__ == "__main__":  
#     input_directory = "femurleft_ply"  # Replace with your input directory  
#     output_base_directory = "split_femur_dataset"    # Replace with your output directory  
      
#     split_dataset(  
#         input_dir=input_directory,  
#         output_base_dir=output_base_directory,  
#         train_ratio=0.8,  
#         class_name="femur"  
#     )