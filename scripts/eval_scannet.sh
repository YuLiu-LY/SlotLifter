export CUDA_VISIBLE_DEVICES=$1
conda activate slotlifter
dataset=scannet
seed=0
python trainer/train.py \
    cfg=${dataset} \
    cfg.job_type='test' \
    cfg.exp_name="eval_scannet" \
    cfg.num_workers=8 \
    cfg.chunk=16384 \
    cfg.test_percent=1.0 \
    cfg.ckpt_path=checkpoints/scannet.ckpt \
    cfg.num_slots=8 \
    cfg.seed=${seed} \
    # cfg.logger=wandb \
