# pip install "lightning-cloud<=0.5.38"
# `import lightning as L` for "tools/test_semantickitti.py"

# export PYTHONPATH=`pwd`:$PYTHONPATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# models -> yaml file
# +model_name -> code folders.

run_eval () {
    LOG_NAME=$1
    MODEL_CFG=$2   # e.g. vpocc or vpocc_skip_original

    python tools/evaluate.py \
        --config-name config.yaml trainer.devices=1 \
        +ckpt_path=./ckpts/semantickitti.ckpt \
        +data_root=./data/SemanticKITTI \
        +label_root=./data/SemanticKITTI/labels \
        +depth_root=./data/SemanticKITTI/depth \
        +log_name=${LOG_NAME} \
        models=${MODEL_CFG} \
        +model_name=vpocc \
        +seed=53
}

run_eval eval_semantickitti vpocc
# run_eval eval_semantickitti_skip_original vpocc_skip_original
# run_eval eval_semantickitti_skip_warped vpocc_skip_warped