export CUDA_VISIBLE_DEVICES=7
conda activate slotlifter
dataset=scannet
seed=0
python trainer/train.py \
    cfg=${dataset} \
    cfg.job_type='train' \
    cfg.exp_name="${dataset}_${seed}" \
    cfg.batch_size=1 \
    cfg.ray_batchsize=1024 \
    cfg.val_check_interval=3 \
    cfg.num_workers=16 \
    cfg.num_slots=8 \
    cfg.num_src_view=4 \
    cfg.seed=${seed} \
    # cfg.logger=wandb \
    