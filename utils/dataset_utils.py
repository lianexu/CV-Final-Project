# Import necessary libraries
import numpy as np
import torch
import os
from utils.io import read_ply_xyz
from utils.pc_transform import scale_numpy
from loss import farthest_point_sample  # Import the FPS function
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.plot import draw_any_set

def normalize_point_cloud(pc, target_range=0.35):
    """
    Normalize a point cloud to fit within a target range centered at origin

    Args:
        pc: Point cloud as numpy array of shape (N, 3)
        target_range: Target range for normalization (default: 0.35)

    Returns:
        Normalized point cloud
    """
    # Find the center of the point cloud
    center = np.mean(pc, axis=0)

    # Center the point cloud
    centered_pc = pc - center

    # Find the maximum distance from center
    max_dist = np.max(np.abs(centered_pc))

    # Scale the point cloud to fit within target range
    scale_factor = target_range / max_dist
    normalized_pc = centered_pc * scale_factor

    return normalized_pc

def create_partial_from_complete(complete_pc, num_points=2048):
    """
    Randomly sample points from a complete point cloud to create a partial point cloud

    Args:
        complete_pc: Complete point cloud (N x 3)
        num_points: Number of points in the partial point cloud

    Returns:
        Partial point cloud (num_points x 3)
    """
    import numpy as np

    # Ensure we don't try to sample more points than available
    num_points = min(num_points, complete_pc.shape[0])

    # Randomly sample indices
    indices = np.random.choice(complete_pc.shape[0], num_points, replace=False)

    # Create partial point cloud
    partial_pc = complete_pc[indices]

    return partial_pc

def load_ply(file_path):
    """Load PLY file as numpy array"""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def save_pcd(points, file_path):
    """Save points as PCD file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)

def rotate_point_cloud(point_cloud, rotation_angle=np.pi/2):
    """
    Rotate the point cloud around X axis by rotation_angle radians
    """
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
        [0, np.sin(rotation_angle), np.cos(rotation_angle)]
    ])

    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return rotated_point_cloud