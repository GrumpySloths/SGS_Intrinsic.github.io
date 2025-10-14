import numpy as np
from typing import Tuple
from utils.stepfun import sample_np
import json
import scipy
# import torch
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

class CameraPoseInterpolator:
    """
    A system for interpolating between sets of camera poses with visualization capabilities.
    """
    
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        """
        Initialize the interpolator with weights for pose distance computation.
        
        Args:
            rotation_weight: Weight for rotational distance in pose matching
            translation_weight: Weight for translational distance in pose matching
        """
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def compute_pose_distance(self, pose1, pose2):
        """
        Compute weighted distance between two camera poses.
        
        Args:
            pose1, pose2: 4x4 transformation matrices
            
        Returns:
            Combined weighted distance between poses
        """
        # Translation distance (Euclidean)
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)
        
        # Rotation distance (angular distance between quaternions)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1 = R1.as_quat()
        q2 = R2.as_quat()
        
        # Ensure quaternions are in the same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        rotation_dist = np.arccos(2 * np.dot(q1, q2)**2 - 1)
        
        return (self.translation_weight * translation_dist + 
                self.rotation_weight * rotation_dist)
    
    def find_nearest_assignments(self, training_poses, testing_poses):
        """
        Find the nearest training camera pose for each testing camera pose.
        
        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            
        Returns:
            assignments: list of closest training pose indices for each testing pose
        """
        M = len(testing_poses)
        assignments = []

        for j in range(M):
            # Compute distance from each training pose to this testing pose
            distances = [self.compute_pose_distance(training_pose, testing_poses[j])
                         for training_pose in training_poses]
            # Find the index of the nearest training pose
            nearest_index = np.argmin(distances)
            assignments.append(nearest_index)
        
        return assignments
    
    def interpolate_rotation(self, R1, R2, t):
        """
        Interpolate between two rotation matrices using SLERP.
        """
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        # Clamp dot product to avoid invalid values in arccos
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        theta = np.arccos(dot_product)
        
        if np.abs(theta) < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1-t)*theta) * q1 + np.sin(t*theta) * q2) / np.sin(theta)
        
        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix()
    
    def interpolate_poses(self, training_poses, testing_poses, num_steps=20):
        """
        Interpolate between camera poses using nearest assignments.
        
        Args:
            training_poses: [N, 4, 4] array of training poses
            testing_poses: [M, 4, 4] array of testing poses
            num_steps: number of interpolation steps
            
        Returns:
            interpolated_sequences: list of lists of interpolated poses
        """
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        interpolated_sequences = []
        
        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]
            sequence = []
            
            for t in np.linspace(0, 1, num_steps):
                # Interpolate rotation
                R_interp = self.interpolate_rotation(
                    train_pose[:3, :3],
                    test_pose[:3, :3],
                    t
                )
                
                # Interpolate translation
                t_interp = (1-t) * train_pose[:3, 3] + t * test_pose[:3, 3]
                
                # Construct interpolated pose
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                
                sequence.append(pose_interp)
            
            interpolated_sequences.append(sequence)
        
        return interpolated_sequences
    
    def interpolate_training_pose_neighbors(self, training_poses, num_steps=20):
        """
        For each training pose, find its nearest neighbor (excluding itself) among training_poses,
        and interpolate between them.

        Args:
            training_poses: [N, 4, 4] array of training poses
            num_steps: number of interpolation steps

        Returns:
            interpolated_sequences: list of lists of interpolated poses for each training pose
            neighbor_indices: list of nearest neighbor indices for each training pose
        """
        N = len(training_poses)
        interpolated_sequences = []
        neighbor_indices = []

        for i in range(N):
            # Compute distances to all other training poses
            distances = [
                self.compute_pose_distance(training_poses[i], training_poses[j]) if i != j else np.inf
                for j in range(N)
            ]
            nearest_idx = np.argmin(distances)
            neighbor_indices.append(nearest_idx)
            pose_a = training_poses[i]
            pose_b = training_poses[nearest_idx]
            sequence = []
            for t in np.linspace(0, 1, num_steps):
                R_interp = self.interpolate_rotation(
                    pose_a[:3, :3],
                    pose_b[:3, :3],
                    t
                )
                t_interp = (1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                sequence.append(pose_interp)
            interpolated_sequences.append(sequence)
        return interpolated_sequences, neighbor_indices


    def interpolate_training_pose_neighbors_with_second(self, training_poses, num_steps=20):
        """
        For each training pose, find its nearest and second nearest neighbors (excluding itself) among training_poses,
        and interpolate between them.

        Args:
            training_poses: [N, 4, 4] array of training poses
            num_steps: number of interpolation steps

        Returns:
            interpolated_sequences: list of lists of interpolated poses for each training pose (to nearest neighbor)
            neighbor_indices: list of nearest neighbor indices for each training pose
            interpolated_sequences_second: list of lists of interpolated poses for each training pose (to second nearest neighbor)
            second_neighbor_indices: list of second nearest neighbor indices for each training pose
        """
        N = len(training_poses)
        interpolated_sequences = []
        neighbor_indices = []
        interpolated_sequences_second = []
        second_neighbor_indices = []

        for i in range(N):
            # Compute distances to all other training poses
            distances = [
                self.compute_pose_distance(training_poses[i], training_poses[j]) if i != j else np.inf
                for j in range(N)
            ]
            sorted_indices = np.argsort(distances)
            nearest_idx = sorted_indices[0]
            second_nearest_idx = sorted_indices[1]
            neighbor_indices.append(nearest_idx)
            second_neighbor_indices.append(second_nearest_idx)

            pose_a = training_poses[i]
            pose_b = training_poses[nearest_idx]
            pose_c = training_poses[second_nearest_idx]

            # Interpolate to nearest neighbor
            sequence = []
            for t in np.linspace(0, 1, num_steps):
                R_interp = self.interpolate_rotation(
                    pose_a[:3, :3],
                    pose_b[:3, :3],
                    t
                )
                t_interp = (1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                sequence.append(pose_interp)
            interpolated_sequences.append(sequence)

            # Interpolate to second nearest neighbor
            sequence_second = []
            for t in np.linspace(0, 1, num_steps):
                R_interp = self.interpolate_rotation(
                    pose_a[:3, :3],
                    pose_c[:3, :3],
                    t
                )
                t_interp = (1 - t) * pose_a[:3, 3] + t * pose_c[:3, 3]
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                sequence_second.append(pose_interp)
            interpolated_sequences_second.append(sequence_second)

        return interpolated_sequences, neighbor_indices, interpolated_sequences_second, second_neighbor_indices
        
    def shift_poses(self, training_poses, testing_poses, distance=0.1, threshold=0.1):
        """
        Shift nearest training poses toward testing poses by a specified distance.
        
        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            distance: float, the step size to move training pose toward testing pose
            
        Returns:
            novel_poses: [M, 4, 4] array of shifted poses
        """
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []

        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]

            if self.compute_pose_distance(train_pose, test_pose) <= distance:
                novel_poses.append(test_pose) #这应该意味着两者之间的差距很小了,故这里的步进选择了直接将current_pose步进到了test pose
                continue

            # Calculate translation step if shifting is necessary
            t1, t2 = train_pose[:3, 3], test_pose[:3, 3]
            translation_direction = t2 - t1
            translation_norm = np.linalg.norm(translation_direction)
            
            if translation_norm > 1e-6:
                translation_step = (translation_direction / translation_norm) * distance #这里直接就是小步直线步进的方式了
                new_translation = t1 + translation_step
            else:
                # If translation direction is too small, use testing pose translation directly
                new_translation = t2

            # Check if the new translation would overshoot the testing pose translation,这里存个疑
            if np.dot(new_translation - t1, t2 - t1) <= 0 or np.linalg.norm(new_translation - t2) <= distance:
                new_translation = t2

            # Update rotation
            R1 = train_pose[:3, :3]
            R2 = test_pose[:3, :3]
            if translation_norm > 1e-6:
                R_interp = self.interpolate_rotation(R1, R2, min(distance / translation_norm, 1.0))
            else:
                R_interp = R2  # Use testing rotation if too close

            # Construct shifted pose
            shifted_pose = np.eye(4)
            shifted_pose[:3, :3] = R_interp
            shifted_pose[:3, 3] = new_translation

            novel_poses.append(shifted_pose)

        return np.array(novel_poses)