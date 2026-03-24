# pip install "lightning-cloud<=0.5.38"
# export PYTHONPATH=`pwd`:$PYTHONPATH

python tools/train.py --config-name config.yaml trainer.devices=4 \
    +data_root=./data/SemanticKITTI \
    +label_root=./data/SemanticKITTI/labels \
    +depth_root=./data/SemanticKITTI/depth \
    +log_name=train_semantickitti \
    +model_name=vpocc \
    +seed=53

# python tools/evaluate.py \
#     --config-name config.yaml trainer.devices=1 \
#     +ckpt_path=./ckpts/semantickitti.ckpt \
#     +data_root=./data/SemanticKITTI \
#     +label_root=./data/SemanticKITTI/labels \
#     +depth_root=./data/SemanticKITTI/depth \
#     +log_name=eval_semantickitti \
#     +model_name=vpocc \
#     +seed=53

# Pending approval here. 
# https://www.codabench.org/competitions/13814/