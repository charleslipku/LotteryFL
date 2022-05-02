# LotteryFL: Empower Edge Intelligence with Personalized and Communication-Efficient Federated Learning

This repository is the official implementation of "[LotteryFL: Empower Edge Intelligence with Personalized and Communication-Efficient Federated Learning](https://ieeexplore.ieee.org/abstract/document/9708944)". 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset supported

All the datasets should be downloaded in "./data/"

```
MNIST (--mnist_extr_noniid for sample limited unbalanced non-iid settings)
CIFAR10 (--cifar10_extr_noniid for sample limited unbalanced non-iid settings)
FEMNIST (--femnist)
Fashion-MNIST (--fmnist)
UCI-HAR (--HAR)
USC-HAD (--HAD)
```


## Training

To train the model(s) with Non-iid MNIST Dataset in the paper, run this command:

```train
python lotteryFL.py --model=cnn --dataset=mnist_extr_noniid --gpu=1 --iid=0 --epochs=400 --prune_percent=20 --prune_start_acc=0.5 --prune_end_rate=0.1 --lr=0.01 --local_bs=32 --num_users=400 --frac=0.05 --nclass=2 --nsamples=20 --rate_unbalance=1.0
```

To train the model(s) with Non-iid CIFAR10 Dataset in the paper, run this command:

```train
python lotteryFL.py --model=cnn --dataset=cifar10_extr_noniid --gpu=1 --iid=0 --epochs=2000 --prune_percent=20 --prune_start_acc=0.1 --prune_end_rate=0.5 --lr=0.01 --local_bs=32 --num_users=400 --frac=0.05 --nclass=2 --nsamples=20 --rate_unbalance=1.0
```

To train the model(s) with EMNIST Dataset in the paper, run this command:

```train
python lotteryFL.py --model=cnn --dataset=femnist --gpu=1 --iid=0 --epochs=2000 --prune_percent=20 --prune_start_acc=0.5 --prune_end_rate=0.1 --lr=0.01 --local_bs=32 --num_users=2424 --frac=0.05
```

To train the model(s) with HAR Dataset, run this command:
```train
python lotteryFL.py --dataset=HAR --prune_percent=30 --prune_start_acc=0.7 --model=mlp_general --gpu=0 --num_users=30 --frac=0.33
```

To run FedAvg, use fedavg.py with the same arguments(The useless arguments for FedAvg would be ignored).

<u>**important options:**</u>
```
--prune_percent:    fixed pruning rate r_p
--prune_start_acc:  acc_threshold
--prune_end_rate:   target pruning rate r_target
--rate_unbalance:   unbalance rate of non-iid MNIST and CIFAR10 dataset
--nsamples:         number of samples per class distributed to clients (for mnist_extr_noniid and cifar10_extr_noniid)
--num_samples:      number of samples per class distributed to clients (for HAR)
--num_users:        number of users (not larger than 30 for HAR)
```

**Comments:**

Current version of code might prune the local models slightly sparser than "prune_end_rate" because of the logic in line 138 of lotteryFL.py. But it does not hurt significantly.

