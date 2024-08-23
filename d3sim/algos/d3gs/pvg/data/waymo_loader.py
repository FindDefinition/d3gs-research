import dataclasses
import math
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from d3sim.algos.d3gs.origin.data.scene.dataset_readers import getNerfppNorm
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import cv2 
import kornia
import torch 
from torch.nn import functional as F
from d3sim.algos.d3gs.origin.data.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from d3sim.constants import D3SIM_DEFAULT_DEVICE
def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fx, fy, w, h):
    top = cy / fy * znear
    bottom = -(h-cy) / fy * znear
    
    left = -(w-cx) / fx * znear
    right = cx / fx * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class BasicPointCloud(NamedTuple):
    points : np.ndarray
    colors : np.ndarray
    normals : np.ndarray | None
    time : np.ndarray | None = None

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int
    sky_mask: np.ndarray | None = None
    timestamp: float = 0.0
    FovY: float | None = None
    FovX: float | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    pointcloud_camera: np.ndarray | None = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_interval: float = 0.02
    time_duration: list = [-0.5, 0.5]

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)


def storePly(path, xyz, rgb, timestamp=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('time', 'f4')]

    normals = np.zeros_like(xyz)
    if timestamp is None:
        timestamp = np.zeros_like(xyz[:, :1])

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, timestamp), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius: float=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
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
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor

@dataclasses.dataclass
class PVGArgs:
    source_path: str
    model_path: str
    resolution_scales: list[float] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8, 16])
    cam_num: int = 3
    num_pts: int = 600000
    frame_interval: float = 0.02
    time_duration: list = dataclasses.field(default_factory=lambda: [-0.5, 0.5])
    fix_radius: float = 0.0
    testhold: int = 4
    eval: bool = False
    debug_cuda: bool = False
    resolution: int = -1
    data_device: str = D3SIM_DEFAULT_DEVICE

def readWaymoInfo(args: PVGArgs):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    points = []
    points_time = []

    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval*(frame_num-1)/2,args.frame_interval*(frame_num-1)/2]
    else:
        time_duration = args.time_duration

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt'))

        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3]
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4)
        lidar2cam = pad_poses(lidar2cam)

        cam2lidar = np.linalg.inv(lidar2cam)
        c2w = ego_pose @ cam2lidar
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []
        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:args.cam_num]:
            image_path = os.path.join(args.source_path, subdir, car_id + '.png')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:args.cam_num]:
            sky_data = np.array(Image.open(os.path.join(args.source_path, subdir, car_id + '.png')))
            sky_mask = sky_data>0
            sky_masks.append(sky_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6)
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
        point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3]
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)
        for j in range(args.cam_num):
            point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3]
            R = np.transpose(w2c[j, :3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovX = FovY = -1.0
            cam_infos.append(CameraInfo(uid=idx * 5 + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[j], 
                                        image_path=image_paths[j], image_name=car_id,
                                        width=HWs[j][1], height=HWs[j][0], timestamp=timestamp,
                                        pointcloud_camera = point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy, 
                                        sky_mask=sky_masks[j]))

        if args.debug_cuda:
            break

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        
        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num)>0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1/nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0],3]), normals=None, time=pointcloud_timestamp)
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class Camera(torch.nn.Module):
    def __init__(self, colmap_id, R, T, FoVx=None, FoVy=None, cx=None, cy=None, fx=None, fy=None, 
                 image=None,
                 image_name=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", timestamp=0.0, 
                 resolution=None, image_path=None,
                 pts_depth=None, sky_mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.resolution = resolution
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.sky_mask = sky_mask.to(self.data_device) > 0 if sky_mask is not None else sky_mask
        self.pts_depth = pts_depth.to(self.data_device) if pts_depth is not None else pts_depth

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(D3SIM_DEFAULT_DEVICE)
        if cx is not None:
            self.FoVx = 2 * math.atan(0.5*self.image_width / fx)
            self.FoVy = 2 * math.atan(0.5*self.image_height / fy)
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fx, fy,
                                                                    self.image_width, self.image_height).transpose(0, 1).to(D3SIM_DEFAULT_DEVICE)
        else:
            self.cx = self.image_width / 2
            self.cy = self.image_height / 2
            self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                         fovY=self.FoVy).transpose(0, 1).to(D3SIM_DEFAULT_DEVICE)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device=D3SIM_DEFAULT_DEVICE)[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u-self.cx+torch.rand_like(u))/self.fx,
                                        (v-self.cy+torch.rand_like(v))/self.fy,
                                        torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u-self.cx+0.5)/self.fx,
                                        (v-self.cy+0.5)/self.fy,
                                        torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_intrinsic(self):
        res = np.eye(3, dtype=np.float32)
        focal_x = fov2focal(self.FoVx, self.image_width)
        focal_y = fov2focal(self.FoVy, self.image_height)
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        res[0, 0] = focal_x
        res[1, 1] = focal_y
        res[0, 2] = center_x
        res[1, 2] = center_y
        return res


def loadCam(args: PVGArgs, id, cam_info: CameraInfo, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height  # cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 8, 16, 32]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if cam_info.cx:
        cx = cam_info.cx / scale
        cy = cam_info.cy / scale
        fy = cam_info.fy / scale
        fx = cam_info.fx / scale
    else:
        cx = None
        cy = None
        fy = None
        fx = None
    
    if cam_info.image.shape[:2] != resolution[::-1]:
        image_rgb = cv2.resize(cam_info.image, resolution)
    else:
        image_rgb = cam_info.image
    image_rgb = torch.from_numpy(image_rgb).float().permute(2, 0, 1)
    gt_image = image_rgb[:3, ...]

    if cam_info.sky_mask is not None:
        if cam_info.sky_mask.shape[:2] != resolution[::-1]:
            sky_mask = cv2.resize(cam_info.sky_mask, resolution)
        else:
            sky_mask = cam_info.sky_mask
        if len(sky_mask.shape) == 2:
            sky_mask = sky_mask[..., None]
        sky_mask = torch.from_numpy(sky_mask).float().permute(2, 0, 1)
    else:
        sky_mask = None

    if cam_info.pointcloud_camera is not None:
        h, w = gt_image.shape[1:]
        K = np.eye(3)
        if cam_info.cx:
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        else:
            K[0, 0] = fov2focal(cam_info.FovX, w)
            K[1, 1] = fov2focal(cam_info.FovY, h)
            K[0, 2] = cam_info.width / 2
            K[1, 2] = cam_info.height / 2
        pts_depth = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ K.T
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2]
        uv = uv.astype(int)
        # TODO: may need to consider overlap
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        pts_depth = torch.from_numpy(pts_depth).float()
    else:
        pts_depth = None

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        image=gt_image,
        image_name=cam_info.image_name,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        image_path=cam_info.image_path,
        pts_depth=pts_depth,
        sky_mask=sky_mask,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_from_camInfos_gen(cam_infos, resolution_scale, args):
    for id, c in enumerate(cam_infos):
        yield loadCam(args, id, c, resolution_scale)

class Scene:
    def __init__(self, args: PVGArgs, shuffle=True):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = readWaymoInfo(args)
        
        self.time_interval = args.frame_interval
        print("time duration: ", scene_info.time_duration)
        print("frame interval: ", self.time_interval)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            

    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

