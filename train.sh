# pip install "lightning-cloud<=0.5.38"
# `import lightning as L` for "tools/test_semantickitti.py"

# export PYTHONPATH=`pwd`:$PYTHONPATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# models -> yaml file
# +model_name -> code folders.

run_train () {
    LOG_NAME=$1
    MODEL_CFG=$2   # vpocc / vpocc_skip_original

    python tools/train.py \
        --config-name config.yaml trainer.devices=4 \
        +data_root=./data/SemanticKITTI \
        +label_root=./data/SemanticKITTI/labels \
        +depth_root=./data/SemanticKITTI/depth \
        +log_name=${LOG_NAME} \
        models=${MODEL_CFG} \
        +model_name=vpocc \
        +seed=53
}

# run_train train_semantickitti vpocc
# run_train train_semantickitti_skip_original vpocc_skip_original
# run_train train_semantickitti_skip_warped vpocc_skip_warped

run_train train_semantickitti_skip_original_warp_gaussian vpocc_skip_original_warp_gaussian