export CUDA_VISIBLE_DEVICES=6,7
conda activate slotlifter
dataset=uorf
subset=clevr_567 # change the num_slots to 8 for clevr_567
# subset=room_chair
# subset=room_diverse
# subset=room_texture
# subset=kitchen_matte
# subset=kitchen_shiny
seed=0
python trainer/train.py \
    cfg=${dataset} \
    cfg.job_type='train' \
    cfg.exp_name="${subset}_${seed}" \
    cfg.group=${subset} \
    cfg.subset=${subset} \
    cfg.num_workers=8 \
    cfg.batch_size=2 \
    cfg.ray_batchsize=1024 \
    cfg.val_check_interval=10 \
    cfg.num_slots=8 \
    cfg.seed=${seed} \
    cfg.logger=wandb \
    # cfg.monitor=psnr \  # using for kitchen_shiny and kitchen_matte because these datasets do not have ground truth masks