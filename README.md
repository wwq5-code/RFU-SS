# RFU-SS

# Representation forgetting unlearning

### Overview
This repository contains the implementation of RFU-SS for [Machine Unlearning via Representation Forgetting with Parameter Self-Sharing (TIFS 2023)](https://ieeexplore.ieee.org/document/10312776)

### Prerequisites

```
python = 3.10.10
torch==2.0.0
torchvision==0.15.1
matplotlib==3.7.1
numpy==1.23.5
```

### Running the experiments

1. To run the RFU and RFU-SS on MNIST
```
python /RFU-SS/VIBU_with_backdoor/On_MNIST/temp.py
```

2. To run the RFU and RFU-SS on CIFAR10
```
python /RFU-SS/VIBU_with_backdoor/On_CIFAR10/cifar10_test.py
```

3. To run our reproduced and improved HFU and VBU on MNIST
```
python /RFU-SS/VIBU_with_backdoor/On_MNIST/temp.py
```

4. To run our reproduced and improved HFU and VBU on CIFAR
```
python /RFU-SS/VIBU_with_backdoor/On_CIFAR10/cifar10_test.py
```


### Citation
```
@ARTICLE{10312776,
  author={Wang, Weiqi and Zhang, Chenhan and Tian, Zhiyi and Yu, Shui},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Machine Unlearning via Representation Forgetting With Parameter Self-Sharing}, 
  year={2024},
  volume={19},
  number={},
  pages={1099-1111},
  keywords={Data models;Training;Degradation;Optimization;Computational modeling;Mutual information;Task analysis;Machine unlearning;representation forgetting;multi-objective optimization;machine learning},
  doi={10.1109/TIFS.2023.3331239}
}
```