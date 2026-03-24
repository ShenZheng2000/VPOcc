# SemanticKITTI 
# Ref: https://github.com/astra-vision/MonoScene#semantickitti 

# download the zip files
    # wget https://www.semantic-kitti.org/assets/data_odometry_voxels.zip
    # wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
    # wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip

# Use VoxFormer to generate gt labels
    # https://github.com/NVlabs/VoxFormer/tree/main/preprocess
    # Below for depth estimation!
    # /longdata/anurag_storage/Shen_Projects/VoxFormer/preprocess/image2depth.sh


# # SSCBench-KITTI360 (TODO later)
# # Ref: https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360
# huggingface-cli download ai4ce-drive/SSCBench --repo-type dataset --include "sscbench-kitti/*" --local-dir sscbench
# cd sscbench
# cat sscbench-kitti-part_* > sscbench-kitti.sqfs
# unsquashfs -no-xattrs sscbench-kitti.sqfs