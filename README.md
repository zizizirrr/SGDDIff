# 🎆 SGDDiff


## Abstract

Salient object detection (SOD) in optical remote sensing images (ORSIs) presents unique challenges due to low spatial resolution, diverse object scales, and complex, class-confusable backgrounds. While diffusion models have achieved remarkable success, their slow and large model size substantially limit their adaptability and practicality in real-world applications. In this work, we propose SGDDiff (Semantic-Guided Diffusion Distillation Network), a novel framework that combines the structural advantages of diffusion models with the efficiency of a compact student network through knowledge distillation. SGDDiff distills a powerful conditional diffusion teacher into a lightweight student guided by pixel-level semantic supervision and spatial feature alignment, enabling progressive saliency prediction from global structure to fine boundaries. A hybrid encoder, VD, is introduced to capture both spatial and temporal dependencies by integrating Vision Mamba with an LSTM architecture enhanced by a Gated Depthwise Block (GDB) module. Additionally, we employ Dynamic Token Normalization(DyT) to improve training stability and convergence. A two-stage training strategy is adopted, involving encoder-decoder pretraining followed by feature-aligned distillation. Extensive experiments on multiple ORSI-SOD benchmarks demonstrate that SGDDiff outperforms 18 state-of-the-art methods while significantly reducing computational overhead, offering a favorable trade-off between accuracy and efficiency.


## Network Architecture
<img width="1680" alt="image" src="">

## Requirements
PyTorch: 2.4.1+cu118 + TorchVision: 0.19.1+cu11 + MMSegmentation: 0.29.0+


## Data preparation
This project uses three datasets for salient object detection (SOD): EORSSD, ORSSD, and ORSI-4199.  
Download each dataset and organize them in the `datasets` directory as follows:  

EORSSD: train-images, train-labels, test-images, test-labels  
ORSSD: train-images, train-labels, test-images, test-labels  
ORSI-4199: train-images, train-labels, test-images, test-labels  

Notes:
1. Filenames of images and corresponding masks must match exactly.
2. Mask files should be in PNG, representing salient regions.
3. Place the `datasets` folder in the project root, and ensure that the `data_root` paths in the configuration files point to the correct dataset directories.

4. Download this dataset and put it into [datasets](https://pan.baidu.com/s/1x7RPfl-gO5K7vRG3f-CA5g?pwd=GZRD). 

## Training
Please download the pre-trained model weights and dataset first. Next, generate the path of the training set and the test set, and change the dataset path in the code to the path of the dataset you specified.

Single GPU training
~~~python
python tools/train_distill.py \
    configs/EORSSD/sgd_eorssd_distill.py

~~~
Multi-GPU training
~~~python
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
  tools/train_distill.py \
  configs/EORSSD/sgd_eorssd_distill.py

~~~


## Inference
~~~python
python tools/images_demo.py \
  --img datasets/EORSSD/test-images \
  --config configs/EORSSD/eorssd_student.py \
  --checkpoint work_dirs/sgd_eorssd_distill/best_mIoU_iter_xxxx.pth \
  --out-dir outputs/eorssd_student

~~~

## Trained Weights of SGDDiff for Inference
We provide Trained Weights of our SGDDiff.
[Download](https://pan.baidu.com/s/1qy5xVyGuj9-RIFDkrFnaYA?pwd=GZRD)

## Saliency maps
We provide saliency maps of our SGDDiff on ORSSD，EORSSD and ORSI-4199 datasets.
[Download](https://pan.baidu.com/s/1nq2OD2je_jqae9LOkpe0Zg)

## Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.

## Acknowledgement
We would like to thank the contributors to the [SGDDiff]().

