# python tools/generate_outputs.py --config-name config.yaml trainer.devices=1 \
#     +ckpt_path=./ckpts/semantickitti.ckpt \
#     +data_root=./data/SemanticKITTI \
#     +label_root=./data/SemanticKITTI/labels \
#     +depth_root=./data/SemanticKITTI/depth \
#     +log_name=vis_semantickitti \
#     +model_name=vpocc

# python tools/visualize.py --config-name config.yaml \
#     +path=outputs/vis_semantickitti \
#     +output_dir=outputs/vis_semantickitti_png \
#     +save_mode=auto

# NOTE: a simple bev alterative for quick debug visualization
python tools/visualize_simple.py \
    --path outputs/vis_semantickitti \
    --output_dir outputs/vis_semantickitti_png