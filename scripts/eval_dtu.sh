export CUDA_VISIBLE_DEVICES=$1
conda activate slotlifter
dataset=dtu

# eval all scenes
python trainer/train.py \
    cfg=${dataset} \
    cfg.job_type='test' \
    cfg.exp_name=eval_dtu_${scene} \
    cfg.test_percent=1.0 \
    cfg.num_workers=16 \
    cfg.precision=32 \
    cfg.num_slots=8 \
    cfg.chunk=16384 \
    cfg.scene_id=-1 \
    cfg.ckpt_path=/home/yuliu/Projects/OR3D/runs/dtu/f32_8s_64+64_01282054/checkpoints/best.ckpt \
    # cfg.logger=wandb 

# eval specific scenes

# ids=(1)
# # ids=(6 7)
# # ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# for scene in ${ids[@]};do
#     python trainer/train.py \
#         cfg=${dataset} \
#         cfg.job_type='test' \
#         cfg.exp_name=eval_dtu_${scene} \
#         cfg.test_percent=1.0 \
#         cfg.num_workers=16 \
#         cfg.precision=32 \
#         cfg.num_slots=8 \
#         cfg.chunk=16384 \
#         cfg.ckpt_path=checkpoints/dtu.ckpt \
#         cfg.scene_id=${scene} \
#         # cfg.logger=wandb
# done
    