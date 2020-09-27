# Warm up
python train.py --gpu_id 0 --isL2Loss --root ../datasets --scale 2 --nEpochs 60 --lr 1e-3 --patch_size 192 > ../training/log_warm_up.txt

# Fine-tuning
python train.py --gpu_id 0 --isL2Loss --root ../datasets --scale 2 --nEpochs 120 --lr 2e-4 --patch_size 512 --pretrained  ../training/epoch_60.pth --save_epoch_bias 60 > ../training/log_fine_tuning.txt
