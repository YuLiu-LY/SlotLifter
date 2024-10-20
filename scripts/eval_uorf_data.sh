export CUDA_VISIBLE_DEVICES=$1
conda activate slotlifter
dataset=uorf
subset=clevr_567 # change the num_slots to 8 for clevr_567
subset=room_chair
# subset=room_diverse
# subset=room_texture
# subset=kitchen_matte
# subset=kitchen_shiny
seed=0
python trainer/train.py \
    cfg=${dataset} \
    cfg.job_type='test' \
    cfg.exp_name="eval_${subset}_seed1" \
    cfg.group=${subset} \
    cfg.subset=${subset} \
    cfg.num_workers=8 \
    cfg.chunk=16384 \
    cfg.test_percent=1.0 \
    cfg.ckpt_path=checkpoints/${subset}/seed1.ckpt \
    cfg.num_slots=5 \
    cfg.val_subsample_frames=1 \
    cfg.seed=${seed} \
    cfg.logger=wandb \

