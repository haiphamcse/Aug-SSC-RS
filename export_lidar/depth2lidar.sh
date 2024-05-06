set -e
exeFunc(){
    num_seq=$1
    python depth2lidar.py --calib_dir  ../data/dataset/sequences/$num_seq \
    --depth_dir ../data/dataset/unidepth/depth/sequences/$num_seq/depth \
    --save_dir ../data/dataset/unidepth/lidar_bin_single/$num_seq \
    # --img_dir ../data/raw_semantic/seem_focal_L_sem_kitti/$num_seq/image_2_np
}
# Change data_path to your own specified path
# And make sure there is enough space under data_path to store the generated data
# data_path=/mnt/NAS/data/yiming/segformer3d_data

for i in {00..10}
do
    exeFunc $i
done

    # python scripts/depth2lidar.py --calib_dir  /home/ubuntu/Workspace/datasets/SemanticKITTI/sequences/$num_seq --depth_dir /home/ubuntu/Workspace/hai-dev/Occupancy/UniDepth/assets/depth/sequences/$num_seq/depth --save_dir /home/ubuntu/Workspace/hai-dev/Occupancy/UniDepth/assets/lidar/sequences/$num_seq --img_dir /home/ubuntu/Workspace/datasets/SemanticKITTI/sequences/$num_seq/image_2

