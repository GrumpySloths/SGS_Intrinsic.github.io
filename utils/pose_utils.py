import numpy as np
from typing import Tuple
from utils.stepfun import sample_np
import json
import scipy
# import torch
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir) # z
  vec0 = normalize(np.cross(up, vec2)) # u
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    #该函数实现的是一个线性代数问题，即求一个点使得其到所有直线的距离都最小
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  transform = np.linalg.inv(pad_poses(cam2world))  #这里transform表示的是w2c矩阵
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform

def generate_spiral_path(poses_arr,
                         n_frames: int = 180,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  poses = poses_arr[:, :-2].reshape([-1, 3, 5])
  bounds = poses_arr[:, -2:]
  fix_rotation = np.array([
      [0, -1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)
  poses = poses[:, :3, :4] @ fix_rotation

  scale = 1. / (bounds.min() * .75)
  poses[:, :3, 3] *= scale
  bounds *= scale
  poses, transform = recenter_poses(poses)

  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_pose = np.eye(4)
    render_pose[:3] = viewmatrix(z_axis, up, position)
    render_pose = np.linalg.inv(transform) @ render_pose
    render_pose[:3, 1:3] *= -1
    render_pose[:3, 3] /= scale
    render_poses.append(np.linalg.inv(render_pose))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses

def generate_spiral_path_dtu(poses_arr, n_frames=180, n_rots=2, zrate=.5, perc=60): 
    """Calculates a forward facing spiral path for rendering for DTU."""
  
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o[:, :3, :4] @ fix_rotation

    c_poses, transform = recenter_poses(poses)

    scale = np.max(np.abs(c_poses[:, :3, -1]))
    c_poses[:, :3, -1] /= scale

    # Get radii for spiral path using 60th percentile of camera positions.
    positions = c_poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(c_poses)
    up = c_poses[:, :3, 1].mean(0)
    z_axis = focus_point_fn(c_poses)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
        
    render_poses = np.stack(render_poses, axis=0)

    render_poses[:, :3, -1] *= scale
    def backcenter_poses(_poses, _pose_ref):
        """Recenter poses around the origin."""
        cam2world = poses_avg(_pose_ref)
        _poses = pad_poses(cam2world) @ pad_poses(_poses)
        return unpad_poses(_poses)
    render_poses = backcenter_poses(render_poses, poses)
    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)
    
    def convert_poses(_poses):
        _poses = np.concatenate([_poses[:, 1:2], _poses[:, 0:1], -_poses[:, 2:3], _poses[:, 3:4], _poses[:, 4:5]], 1)
        bottom = np.tile(np.array([0,0,0,1.]).reshape([1,1,4]), (_poses.shape[0], 1, 1))

        H, W, fl = _poses[0, :, -1]

        _poses = np.concatenate([_poses[..., :4], bottom], 1)
        _poses = np.linalg.inv(_poses)
        Rs = _poses[:, :3, :3]
        tvecs = _poses[:, :3, -1]
        # print(Rs.shape, tvecs.shape, H, W, fl)
        return Rs, tvecs, H, W, fl

    Rs, tvecs, height, width, focal_length_x = convert_poses(render_poses)

    from utils.graphics_utils import focal2fov
    from scene.dataset_readers import CameraInfo

    
    cam_infos = []
    
    for idx, _ in enumerate(Rs):

        uid = idx
        # R = np.transpose(Rs[idx])
        R_ori = np.transpose(Rs[idx])
        R = np.zeros_like(R_ori)
        R[0, 0] = - R_ori[1, 1]
        R[0, 1] = R_ori[1, 0]
        R[0, 2] = R_ori[1, 2]
        R[1, 0] = - R_ori[0, 1]
        R[1, 1] = R_ori[0, 0]
        R[1, 2] = R_ori[0, 2]
        R[2, 0] = R_ori[2, 1]
        R[2, 1] = - R_ori[2, 0]
        R[2, 2] = - R_ori[2, 2]
        
        T_ori = tvecs[idx]
        T = np.zeros_like(T_ori)
        T[0] = - T_ori[1]
        T[1] = T_ori[0]
        T[2] = T_ori[2]

        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, image_path=None, image_name=None, width=width, height=height, mask=None, bounds=bounds)
        cam_infos.append(cam_info)
    
    return cam_infos

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def generate_ellipse_path(views, n_frames=600, const_speed=True, z_variation=0., z_phase=0.):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses

def generate_random_poses_llff_annealing_view(views, n_frames=10000):
    """
    Generates random poses using LLFF annealing view method.
    Args:
        views (list): A list of view objects, each containing rotation matrix (R), translation vector (T), and bounds.
        n_frames (int, optional): Number of frames to generate. Defaults to 10000.
    Returns:
        tuple: A tuple containing:
            - random_poses (list): A list of generated random poses.
            - closest_poses (list): A list of closest poses from the input views.这里的closest_poses指的是距离对应的random_poses最近的gt_poses
    The function performs the following steps:
        1. Converts input views to homogeneous transformation matrices and inverts them.
        2. Re-centers and scales the poses.
        3. Computes focal length based on the bounds.
        4. Generates random poses by adding noise to the selected poses.
        5. Ensures the generated poses are within the specified bounds.
    """
    # 初始化存储姿态和边界的列表
    poses, bounds = [], []
    
    # 将输入视图转换为齐次变换矩阵并对其进行求逆
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        bounds.append(view.bounds) #这里的bounds应该代表的是near和far的值,每个near和far不一样是因为对于真实
        #场景来说其相机参数是通过colmap来估计出来的，这里的near和far其实代表着场景点距离相机中心的最近距离和最远距离
    # 将姿态和边界堆叠成numpy数组
    poses = np.stack(poses, 0)  #(3,4,4)
    bounds = np.stack(bounds)

    # 缩放姿态和边界
    scale = 1. / (bounds.min() * .75) #这里让scale等于near的75%是为了让相机的视野范围更大一些
    poses[:, :3, 3] *= scale #这里实际上修改的是相机在世界坐标系下的位置
    bounds *= scale
    
    # 重新中心化姿态,这里是因为原来的相机位姿在直接经过缩放之后它的几何中心或者说重心不在世界坐标系原点，需要经过一个变换让其中心是在坐标系原点的
    #这里进行recenter的主要目的应该是稀疏视角采样的相机看向的是中心物体
    poses, transform = recenter_poses(poses)

    # 根据边界计算焦距，混合近远平面，计算一个虚拟的焦距focal，平衡近处和远处的细节。
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))
    
    # 获取用于生成随机姿态的位置和半径,这里构建的radii主要是用于nerf的螺旋相机路径生成，在本函数中实际并没有用到
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # 计算平均的相机到世界的变换矩阵和上向量
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    
    # 初始化噪声生成的参数
    poses_num = len(poses)
    t_std_start = 0.2 * n_frames
    z_std_start = 0.2 * n_frames
    t_std_max = 0.05 #这里指的是添加噪声的最大值，t代表的是相机的位置
    z_std_max = 0.05 #这里z指的实际上是相机的z轴朝向所添加噪声的最大值
    
    # 初始化存储随机姿态和最近姿态的列表
    random_poses = []
    closest_poses = []

    # 生成随机姿态,这里说穿了在实质上就是原本的相机位姿空间下直接添加噪声并不是一个容易的事，所以先将其归一化到标准空间下，这样就可以比较方便的添加噪声了
    #只需要简单的标准高斯噪声，再还原到原来的相机空间就可以了，而且在标准空间下生成随机位姿可以保证这个位姿不会很偏
    for idx in range(n_frames):
        # 选择一个随机姿态
        pose_idx = np.random.randint(poses_num)
        selected_pose = poses[pose_idx]
        selected_position = selected_pose[:3, 3]
        
        # 对选择的位置添加噪声
        selected_t = selected_position
        assert (selected_t < -1).sum() == 0 and (selected_t > 1).sum() == 0
        t_std = max(t_std_start, float(idx+1)) / n_frames * t_std_max
        t_noise = np.random.randn(3) * t_std
        noisy_t = selected_t + t_noise
        noisy_position = noisy_t
        noisy_position = np.clip(noisy_position, -1, 1) #这里直接进行clip应该是有合理性的，毕竟是在标准化的空间下
        
        # 计算注视点并添加噪声
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_std = max(z_std_start, float(idx+1)) / n_frames * z_std_max
        z_noise = np.random.randn(3) * z_std
        noisy_lookat = lookat + z_noise
        
        # 计算z轴并生成随机姿态
        noisy_z = noisy_position - noisy_lookat
        random_pose = np.eye(4)
        random_pose[:3] = viewmatrix(noisy_z, up, noisy_position)
        random_pose = np.linalg.inv(transform) @ random_pose
        random_pose[:3, 1:3] *= -1
        random_pose[:3, 3] /= scale
        random_poses.append(np.linalg.inv(random_pose))

        # 存储最近的姿态
        closest_poses.append(views[pose_idx])
    
    return random_poses, closest_poses

def generate_random_poses_dtu_annealing_view(views, n_frames=10000):
    """Generates random poses."""
    init_poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        init_poses.append(tmp_view)
    init_poses = np.stack(init_poses, 0)
    poses, transform = transform_poses_pca(init_poses)

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    
    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    
    poses_num = len(poses)
    t_std_start = 0.2 * n_frames
    t_std_max = 0.05
    z_std_start = 0.2 * n_frames
    z_std_max = 0.05
    
    random_poses = []
    closest_poses = []

    for idx in range(n_frames):
        
        pose_idx = np.random.randint(poses_num)
        selected_pose = poses[pose_idx]
        
        selected_position = selected_pose[:3, 3]
        selected_t = selected_position
        
        assert (selected_t < -1).sum() == 0 and (selected_t > 1).sum() == 0
        
        t_std = max(t_std_start, float(idx+1)) / n_frames * t_std_max
        t_noise = np.random.randn(3) * t_std
        noisy_t = selected_t + t_noise
        noisy_position = noisy_t
        noisy_position = np.clip(noisy_position, -1, 1)
        
        lookat = center
        z_std = max(z_std_start, float(idx+1)) / n_frames * z_std_max
        z_noise = np.random.randn(3) * z_std
        noisy_lookat = lookat + z_noise
        
        noisy_z = noisy_position - noisy_lookat
        
        random_pose = np.eye(4)
        random_pose[:3] = viewmatrix(noisy_z, up, noisy_position)
        random_pose = np.linalg.inv(transform) @ random_pose
        random_pose[:3, 1:3] *= -1
        random_pose = np.linalg.inv(random_pose)
        random_poses.append(random_pose)

        closest_poses.append(views[pose_idx])
    
    return random_poses, closest_poses

def generate_random_poses_360_annealing_view(views, n_frames=10000, z_variation=0.1, z_phase=0):
    #这里的views是一个list，维护了一个camera类 list
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)  #这里poses明确表示的是相机的位姿
    # 保存原始poses到本地npy文件
    np.save("poses_360.npy", poses)
    poses, transform = transform_poses_pca(poses)
    # 保存transform后的poses到本地npy文件
    np.save("poses_360_transformed.npy", poses)

    # Calculate the focal point for the path (cameras point toward this).下面的代码应该是致力于构建一个椭圆的轨迹，方便后续进行采样
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    # 保存关键参数到字典并写入本地文件
    ellipse_params = {
        "center": center.tolist(),
        "offset": offset.tolist(),
        "sc": sc.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "z_low": z_low.tolist(),
        "z_high": z_high.tolist(),
        "transform": transform.tolist(),
    }
    with open("ellipse_params_360.json", "w") as f:
        json.dump(ellipse_params, f, indent=4)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.random.rand(n_frames) * 2. * np.pi  #这里是从[0,2pi]中随机采样n_frames个点
    positions = get_positions(theta)
    #保存随机采样的椭圆轨迹到本地文件
    np.save("random_positions_360.npy", positions)
    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    # random_poses = []
    closest_poses = []

    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
        
        # Find the closest pose in views based on Euclidean distance
        min_distance = float('inf')
        closest_pose = None
        for view in views:
            view_position = view.T  # Assuming view.T is the position of the view
            distance = np.linalg.norm(p - view_position)
            if distance < min_distance:
                min_distance = distance
                closest_pose = view
        closest_poses.append(closest_pose)
    # 保存render_poses到本地npy文件，便于后续分析
    np.save("render_poses_360.npy", np.array(render_poses))
     
    return render_poses,closest_poses

def generate_random_poses_360(views, n_frames=10000, z_variation=0.1, z_phase=0):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.random.rand(n_frames) * 2. * np.pi
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses

def interpolate_poses_spline(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]  #这里的lookat指的是z轴朝向看向的方向，应该是负z轴朝向
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]  #这里的up是指的是y轴朝向看向的方向，应该是正y轴朝向
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape  #(2,3,3)
        pts = np.reshape(points, (sh[0], -1)) #reshape为(2,9)
        k = min(k, sh[0] - 1)  #这里应该是样条插值的阶数是有一个上限存在的
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck)) #shape: (9,25)
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    def viewmatrix(lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    points = poses_to_points(poses, dist=rot_weight) #rot_weight=0.1 (pos,lookat,up)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree, #5
                        s=smoothness)
    new_poses = points_to_poses(new_points) 
    extra_row = np.tile(np.array([[0, 0, 0, 1]]), (new_poses.shape[0], 1, 1))
    poses_final = np.concatenate([new_poses, extra_row], axis=1)

    return poses_final

#该函数用于对views进行插值生成对应的虚拟伪视角，这里对views的选取是采用最近邻的方式来进行选取的
def generate_interpolate_poses_360(views):

    # n_interp=25
    n_interp=5

    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        poses.append(tmp_view)

    c2ws = np.stack(poses, 0)
    
    # c2ws = poses  # poses shape: (N, 4, 4)
    positions = c2ws[:, :3, 3]
    sorted_indices = np.argsort(positions[:, 2])
    remaining_indices = list(sorted_indices)
    interpolated_poses = []
    closest_poses = []

    while len(remaining_indices) > 1:
        current_idx = remaining_indices[0]
        current_pos = positions[current_idx].reshape(1, 3)
        other_indices = remaining_indices[1:]
        other_positions = positions[other_indices]
        dists = cdist(current_pos, other_positions)[0]
        nearest_idx_in_other = np.argmin(dists)
        nearest_idx = other_indices[nearest_idx_in_other]
        interp = interpolate_poses_spline(
            c2ws[[current_idx, nearest_idx], :3, :], n_interp=n_interp
        )
        # 对interp中的每个c2w矩阵进行求逆，得到对应的w2c矩阵
        interp = np.linalg.inv(interp)
        interpolated_poses.append(interp[1:])  # append as numpy array
        # 记录最近邻view
        for _ in range(interp.shape[0] - 1):
            closest_poses.append(views[current_idx])
        remaining_indices.pop(0)
        remaining_indices = sorted(remaining_indices, key=lambda i: positions[i, 2])

    # 可选：最后一个相机直接加入
    # interpolated_poses.append(c2ws[remaining_indices, :3, :])
    # for _ in range(len(remaining_indices)):
    #     closest_poses.append(views[remaining_indices[_]])

    final_interpolated_poses = np.concatenate(interpolated_poses, axis=0)

    return final_interpolated_poses, closest_poses

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