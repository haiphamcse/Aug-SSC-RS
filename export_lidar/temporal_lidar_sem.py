
# import open3d as o3d
import numpy as np
import pickle
import torch
# import monoscene.data.semantic_kitti.io_data as SemanticKittiIO
from numpy.linalg import inv
from PIL import Image
import os
from tqdm import tqdm

def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses


def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib

def process_pc(f):
    ref_num = int(f[:-4])
    ref = poses[int(ref_num)] # reference frame with GT semantic voxel
    path = f"data/dataset/unidepth/accumulated_lidar_seg/sequences/{scene_id}/"
    os.makedirs(path, exist_ok=True)
    id = r"{:06d}.npy".format(ref_num)
    f_path = os.path.join(path, id)
    if os.path.isfile(f_path):
        pass
    else:
        ## Load Target Point Clouds ##
        targets = [ref_num]
        min_num = max(ref_num-20, 0)
        max_num = min(ref_num, len(poses))
        for i in range(min_num, ref_num-1, 5):
            targets.append(i)
        voxels_ = []
        target2refs = []
        for i in targets:
            target = poses[i]
            target2ref = np.matmul(inv(ref), target) # both for lidar
            target2refs.append(target2ref)
            folder = "data/dataset/unidepth/lidar_seg/{}/{:06d}.npy".format(scene_id, i)
            semantics = np.load(folder)
            voxels_.append(semantics)

        accum_pts = []
        for i in range(len(voxels_)):
            vox = voxels_[i]
            tar = target2refs[i]

            points_ = np.ones((vox[:, :4].shape))
            points_[:, 0:3] = vox[:, 0:3]
            new = np.matmul(tar, points_.T)
            
            if i != 0:
                mask = (vox[:, 2] < 6) & (vox[:, 0] < 30.0) & (vox[:, 0] > 1.0) & (vox[:, 4] != 255) & (vox[:, 4] != 0)
            else:
                mask = (vox[:, 2] < 4.4) & (vox[:, 4] != 255) & (vox[:, 4] != 0)
            vox[:, :3] = new.T[:, :3]
            accum_pts.append(vox[mask])
        
        accum_pts_np = np.concatenate(accum_pts, axis=0)
        np.save(f_path, accum_pts_np)
        print(f'Done: {f_path}')

if __name__ == "__main__":
    ids = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # ids = ["08"]
    out_path = "/home/ubuntu/Workspace/hai-dev/Occupancy/UniDepth/assets/depth/sequences"
    for scene_id in ids:
        print(f"Current Split: {scene_id}")
        
        calibration = parse_calibration(f"data/dataset/sequences/{scene_id}/calib.txt")
        poses = parse_poses(f"data/dataset/sequences/{scene_id}/poses.txt", calibration)
        print(len(poses))

        folder = f"data/dataset/unidepth/lidar_seg/{scene_id}" # change this to the file path on your machine
        files = sorted(os.listdir(folder))
        
        for f in tqdm(files):
            process_pc(f)
