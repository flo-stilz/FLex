import numpy as np
import os
from PIL import Image
import json
import sys
import tifffile as tiff
import torch
import imageio
from lietorch import SE3
from rectification import StereoRectifier
import cv2
import shutil
sys.path.append(os.getcwd())
from flex.render.util.util import draw_poses
from local.util import read_freiburg, read_poses
from local.eval_poses import get_cam2world
from torchvision import transforms as T
from tqdm.auto import tqdm, trange
import imageio.v2 as iio
import argparse


def generate_gif(file_paths):
    transform = T.ToTensor()
    all_images = []
    for file_path in file_paths:
        img = Image.open(os.path.join(basedir+'/images', file_path))

        img = transform(img)  # (3, h, w)
        img = img.permute(1, 2, 0)  # (h*w, 3) RGB
        img = (img*255).numpy().astype(np.uint8)

        all_images += [img]


    tqdm.write("DEMO|Generate rendering gif...")
    imageio.mimwrite(
        f"{basedir}/video.mp4",
        np.stack(np.array(all_images[::2])),
        fps=35,
        format="FFMPEG",
        quality=8,
    )
    fps = 35
    if len(all_images)>2000:
        fps = (len(all_images)/2000) * fps
    with iio.get_writer(f"{basedir}/video.gif",mode="I", duration=1/fps) as writer:
        for frame in all_images:
            writer.append_data(frame) 

def adjust_depth(poses, scale, basedir, dict=dict):
    # fix depth images (scale depth images like poses):
    if scale!=1.0:
        os.makedirs(os.path.join(basedir, "depth_adj"), exist_ok=True)
        depth_file_paths_original = sorted(os.listdir(os.path.join(basedir, 'depth_original')))
        max_depth = 0
        # find max depth value for normalization
        for j in range(len(poses)):
            depths = np.array(Image.open(os.path.join(basedir, 'depth_original/') + depth_file_paths_original[j]), dtype=np.float64)
            if np.max(depths) > max_depth:
                max_depth = np.max(depths)

        # rescale depth and save!
        new_max = 0
        for j in range(len(depth_file_paths_original)):
            depths = np.array(Image.open(os.path.join(basedir, 'depth_original/') + depth_file_paths_original[j]), dtype=np.float64)
            
            depths /= scale

            # rescale poses translation accordingly
            if np.max(depths)>new_max:
                new_max = np.max(depths)
            tiff.imsave(os.path.join(basedir, 'depth_adj/') + depth_file_paths_original[j][:-3]+"tiff", depths)

        print("Old Max. Depth:"+str(max_depth))
        print("New Max. Depth: "+str(new_max))
        print(poses[:10, :, 3])
        print(f"Poses Trans. Max: {np.max(poses[:,:,3])}")
        print(f"Poses Trans. Min: {np.min(poses[:,:,3])}")

    return poses


def create_transforms(i_train, poses, basedir, tool_mask, flow_data, file_paths, depth_file_paths, tool_mask_paths, fwd_paths, bwd_paths, all_frames, i_test, split='train', dict=dict):
    frames = []
    for i in i_train:
        item = {}
        item['file_path'] = 'images/' + file_paths[i][:-4]
        item['transform_matrix'] = poses[i, :, :4].tolist()
        item['transform_matrix'].append([0, 0, 0, 1])
        item['time'] = (float(i)/(len(file_paths)-1)) # correct version
        
        item['depth_file_path'] = basedir + '/depth_adj/' + depth_file_paths[i]
        
        if tool_mask:
            item['tool_mask_path'] = basedir + '/masks/' + tool_mask_paths[i]

        if flow_data:
            if i<(len(all_frames)-1):
                item['fwd_file_path'] = basedir + '/flow_ds/fwd/' + fwd_paths[i+1]
            else:
                item['fwd_file_path'] = basedir + '/flow_ds/fwd/' + fwd_paths[0]
            item['bwd_file_path'] = basedir + '/flow_ds/bwd/' + bwd_paths[i]

            item['fwd_mask'] = 1 if i<(len(all_frames)-1) else 0
            item['bwd_mask'] = 1 if i!=0 else 0 
            item['prev_test'] = 1 if (i-1 in i_test) else 0
            item['post_test'] = 1 if (i+1 in i_test) else 0

        frames.append(item)

    dict['frames'] = frames

    with open(os.path.join(basedir, "transforms_"+split+".json"), "w") as outfile:
        json.dump(dict, outfile)


def create_hexd(scene='23', factor=1, flow_data=False, data_type='stereomis', stereomis_gt=False, tool_mask=True, own_poses=False, generate_gif=False, calib_img_size=None):

    if data_type=='miti':
        #basedir = '/media/storage/stilz/HexPlane/data/miti_slam_nerf/' + str(scene)
        basedir = 'data/miti_slam_nerf/'+str(scene)
    elif data_type=='stereomis':
        basedir = 'data/StereoMIS_FLex/'+str(scene)
    else:
        AssertionError('Unknown Data Type!')

    # load poses
    if data_type=='stereomis':
        if stereomis_gt:
            poses_arr = read_poses(pose_path=os.path.join(basedir, 'groundtruth.txt'))
        else:
            poses_arr = read_poses(pose_path=os.path.join(basedir, 'trajectory.freiburg'))
            #poses_arr = poses_arr[1:] # fix duplicate of first entry
        poses = poses_arr
        
    else:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        
    length = len(poses_arr)

    # Select the data split.
    i_test = np.arange(length)[::8]
    i_train = np.array(
        [i for i in np.arange(int(length)) if i not in i_test])

    all_frames = np.arange(length)

    if not (data_type=='stereomis'):
        file_paths = sorted(os.listdir(os.path.join(basedir, 'images')))
    else:
        file_paths = sorted(os.listdir(os.path.join(basedir, 'video_frames')))
        # use only left images 
        keep_files = [item for item in file_paths if "l" in item]
        if len(keep_files)!=0:
            file_paths = keep_files
            # move files to new folder
            os.makedirs(os.path.join(basedir, 'images'), exist_ok=True)
            current_root = os.path.join(basedir, 'video_frames')
            for file_path in file_paths:
                shutil.move(os.path.join(current_root, file_path), os.path.join(basedir, 'images'))
        else:
            file_paths = sorted(os.listdir(os.path.join(basedir, 'images')))

        if generate_gif:
            generate_gif(file_paths)

    if flow_data:
        fwd_paths = sorted(os.listdir(os.path.join(basedir, 'flow_ds/fwd')))
        bwd_paths = sorted(os.listdir(os.path.join(basedir, 'flow_ds/bwd')))
    if tool_mask:
        tool_mask_paths = sorted(os.listdir(os.path.join(basedir, 'masks')))

    dict = {}

    # Set camera intrinsics
    dim = np.array(Image.open(os.path.join(basedir, 'images/') + file_paths[0])).shape
    dict['w'], dict['h'] = dim[1], dim[0]
    print(dict['w'], dict['h'])
    dict['camera_model'] = 'OPENCV'

    if data_type=='miti':
        dict['fl_x'], dict['fl_y'], dict['cx'], dict[
            'cy'] = 1.5709233828878441e+03/2, 1.5709233828878441e+03/2, 8.0491165252677206e+02/2, 5.2790839350304282e+02/2
    elif data_type=='stereomis':
        rect_mode = 'conventional'
        img_size = [dict['w'], dict['h']]
        if calib_img_size is not None:
            img_size = calib_img_size

        calib_file = os.path.join(basedir, 'StereoCalibration.ini')
        rect = StereoRectifier(calib_file, img_size_new=img_size, mode=rect_mode)
        calib = rect.get_rectified_calib()
        print(calib)

        dict['fl_x'], dict['fl_y'] = calib['intrinsics']['left'][0,0], calib['intrinsics']['left'][1,1]
        dict['cx'] = calib['intrinsics']['left'][0,2]
        dict['cy'] = calib['intrinsics']['left'][1,2]

    if data_type=='stereomis' or data_type=="miti":
        
        # Convert from OpenCV to OpenGL Format
        print(f"Poses Trans. Max. Before: {np.max(poses[:,:,3])}")
        print(f"Poses Trans. Min. Before: {np.min(poses[:,:,3])}")
        init_max_pose_t = np.max(poses[:,:,3])
        flip = np.eye(4)
        flip[1,1] = -1
        flip[2,2] = -1
        poses = np.matmul(poses, flip)

        if data_type!="stereomis":
            poses = poses[:,:3,:4]

        if data_type=='stereomis':

            bds = np.array([0.01, 12.0]).reshape(1,2)
            bds = np.repeat(bds, len(poses), axis=0)
            hwf = np.array([dict["h"], dict["w"], dict['fl_x']]).reshape(1,3,1)
            hwf = np.repeat(hwf, len(poses), axis=0)

            scale = 1.0
            poses, scale, center_shift = center_poses_0(poses)
            print(f"ReScale: {scale}")
            print(f"ReCenter: {center_shift}")

            poses_3x5 = np.concatenate((poses[:,:3,:4], hwf), axis=-1)
            poses_flat = poses_3x5.reshape(-1, 15)
            poses_bounds = np.concatenate((poses_flat, bds), axis=-1)

            np.save(basedir + '/poses_bounds.npy', poses_bounds)
            print('saved poses bounds!')

            near_fars = poses_arr[:, -2:]
            poses = poses_bounds[:, :-2].reshape(-1, 3, 5).transpose([1, 2, 0])
            bds = poses_bounds[:, -2:].transpose([1, 0])
            bds = np.moveaxis(bds, -1, 0).astype(np.float32)

            poses = np.moveaxis(poses, -1, 0).astype(np.float32)

            if own_poses:
                poses_rot = torch.load(os.path.join(basedir, "own_poses/poses_rot.th"),map_location=torch.device('cpu')).detach()
                poses_t = torch.load(os.path.join(basedir, "own_poses/poses_t.th"),map_location=torch.device('cpu')).detach()
                poses = get_cam2world(poses_rot, poses_t).numpy()
                # 0-center poses!
                poses, new_scale, new_center_shift = center_poses_0(poses)
                scale *= new_scale
                center_shift += new_center_shift

            
    print(f"Poses Shape: {poses.shape}")
    dict["depth_scale_factor"] = scale.item()
    dict["recenter_trans"] = center_shift.tolist()
    poses = adjust_depth(poses, scale, basedir, dict=dict)
    depth_file_paths = sorted(os.listdir(os.path.join(basedir, 'depth_adj')))
        

    # create dict for each frame:
    create_transforms(i_train, poses, basedir, tool_mask, flow_data, file_paths, depth_file_paths, tool_mask_paths, fwd_paths, bwd_paths, all_frames, i_test, split='train', dict=dict)
    create_transforms(i_test, poses, basedir, tool_mask, flow_data, file_paths, depth_file_paths, tool_mask_paths, fwd_paths, bwd_paths, all_frames, i_test, split='test', dict=dict)
    create_transforms(all_frames, poses, basedir, tool_mask, flow_data, file_paths, depth_file_paths, tool_mask_paths, fwd_paths, bwd_paths, all_frames, i_test, split='', dict=dict)



def center_poses_0(poses):

    center = poses[..., 3].mean(0)
    poses[..., 3] = poses[..., 3] - center
    max_poses = abs(poses[..., 3]).max(0)
    scale = max_poses.max(0)
    print(f"Max poses along translation axis: {max_poses} and scale value: {scale}")
    poses[..., 3] = (poses[..., 3]) / scale
    
    return poses, scale, center

def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def _generate_spherical_poses(poses, bds):
    """Generate a 360 degree spherical path for rendering."""
    # pylint: disable=g-long-lambda
    p34_to_44 = lambda p: np.concatenate([
      p,
      np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
      a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
      b_i = -a_i @ rays_o
      pt_mindist = np.squeeze(-np.linalg.inv(
        (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
      return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
            np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
      camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
      up = np.array([0, 0, -1.])
      vec2 = normalize(camorigin)
      vec0 = normalize(np.cross(vec2, up))
      vec1 = normalize(np.cross(vec2, vec0))
      pos = camorigin
      p = np.stack([vec0, vec1, vec2, pos], 1)
      new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
      new_poses,
      np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
      poses_reset[:, :3, :4],
      np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    return poses_reset


if __name__ == "__main__":
    print('Start!')
    # console arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--data_type', type=str, help='Dataset name')
    parser.add_argument('--factor', type=int, default=1 ,help='Size ratio to initial images')
    parser.add_argument('--flow_data', action='store_true')
    parser.add_argument('--tool_mask', action='store_true')
    parser.add_argument('--stereomis_gt', action='store_true')
    parser.add_argument('--own_poses', action='store_true')
    parser.add_argument('--generate_gif', action='store_true')
    parser.add_argument('--calib_img_size', nargs='+', type=int, default=None, help='set image dimensions, epsecially needed when changing image via cropping etc. to fix calibration file')
    args = parser.parse_args()
    create_hexd(scene=args.scene, factor=args.factor, flow_data=args.flow_data, data_type=args.data_type, stereomis_gt=args.stereomis_gt, tool_mask=args.tool_mask, own_poses=args.own_poses, generate_gif=args.generate_gif, calib_img_size=args.calib_img_size)
    print('Done!')

    # Example
    # python flex/dataloader/preprocess_data.py --scene 'P2_8_1' --data_type 'stereomis' --flow_data --tool_mask
