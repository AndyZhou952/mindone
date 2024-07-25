python opensora/train/train_causalvae.py \
    --exp_name "9x256x256" \
    --batch_size 1 \
    --precision bf16 \
    --amp_level "O2" \
    --max_steps 100000 \
    --save_steps 2000 \
    --output_dir results/causalvae \
    --video_path /remote-home1/dataset/data_split_tt \
    --video_num_frames 9 \
    --resolution 256 \
    --sample_rate 1 \
    --dataloader_num_workers 8 \
    --load_from_checkpoint pretrained/causal_vae_488_init.ckpt \
    --start_learning_rate 1e-5 \
    --lr_scheduler constant \
    --optim adam \
    --betas 0.5 0.9 \
    --clip_grad True \
    --weight_decay 0.0 \
    --mode 0 \
    --init_loss_scale 128 \
    --jit_level "O0" \