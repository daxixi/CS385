


python3 train.py --auxiliary --cutout --arch RL1 --gpu 3 --batch_size 64

python3 train_search.py

python3 test.py --gpu 1 --auxiliary --model_path final/warmup_6k_lr3e-3_best/weights.pt --arch RL14 --batch_size 64
