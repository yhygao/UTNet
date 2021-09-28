# UTNet
Official implementation of UTNet: A Hybrid Transformer Architecture for Medical Image Segmentation

## Introduction 

Transformer architecture has emerged to be successful in a
number of natural language processing tasks. However, its applications
to medical vision remain largely unexplored. In this study, we present
UTNet, a simple yet powerful hybrid Transformer architecture that in-
tegrates self-attention into a convolutional neural network for enhancing
medical image segmentation. UTNet applies self-attention modules in
both encoder and decoder for capturing long-range dependency at dif-
ferent scales with minimal overhead. To this end, we propose an efficient
self-attention mechanism along with relative position encoding that re-
duces the complexity of self-attention operation significantly from O(n2)
to approximate O(n). A new self-attention decoder is also proposed to
recover fine-grained details from the skipped connections in the encoder.
Our approach addresses the dilemma that Transformer requires huge
amounts of data to learn vision inductive bias. Our hybrid layer de-
sign allows the initialization of Transformer into convolutional networks
without a need of pre-training. We have evaluated UTNet on the multi-
label, multi-vendor cardiac magnetic resonance imaging cohort. UTNet
demonstrates superior segmentation performance and robustness against
the state-of-the-art approaches, holding the promise to generalize well on
other medical image segmentations.

![image](https://user-images.githubusercontent.com/55367673/134997310-69c3576d-bbf2-40c8-ad5a-9b3bf3e9e97d.png)
![image](https://user-images.githubusercontent.com/55367673/134997347-a581cda7-7050-48ef-9af3-d4628fefac9a.png)

## Supportting models
UTNet

TransUNet

ResNet50-UTNet

ResNet50-UNet

To be continue ...

## Getting Started
### Prerequisites
```
Python >= 3.6
pytorch = 1.8.1
SimpleITK = 2.0.2
numpy = 1.19.5
einops = 0.3.2
```

### Preprocess
Currently, we only support [M&Ms dataset](https://www.ub.edu/mnms/).

Resample all data to spacing of 1.2x1.2 mm in x-y plane. We don't change the spacing of z-axis, as UTNet is a 2D network. Then put all data into 'dataset/'

### Training

#### UTNet
For default UTNet setting, training with:
```
python train_deep.py -m UTNet -u EXP_NAME --data_path YOUR_OWN_PATH --reduce_size 8 --block_list 1234 --num_blocks 1,1,1,1 --gpu 0 --aux_loss
```
To optimize UTNet in your own task, there are several hyperparameters to tune:

'--block_list': indicates apply transformer blocks in which resolution. The number means the number of downsamplings, e.g. 3,4 means apply transformer blocks in features after 3 and 4 times downsampling. Apply transformer blocks in higher resolution feature maps will introduce much more computation.

'--num_blocks': indicates the number of transformer blocks applied in each level. e.g. block_list='3,4', num_blocks=2,4 means apply 2 transformer blocks in 3-times downsampling level and apply 4 transformer blocks in 4-time downsampling level.

'--reduce_size': indicates the size of downsampling for efficient attention. In our experiments, reduce_size 8 and 16 don't have much difference, but 16 will introduce more computation, so we choost 8 as our default setting. 16 might have better performance in other applications.

'--aux_loss': applies deep supervision in training, will introduce some computation overhead but has slightly better performance.

Here are some recomended parameter setting:
```
--block_list 1234 --num_blocks 1,1,1,1
```
Our default setting, most efficient setting. Suitable for tasks with limited training data, and most errors occur in the boundary of ROI where high resolution information is important.

```
--block_list 1234 --num_blocks 1,1,4,8
```
Similar to the previous one. The model capacity is larger as more transformer blocks are including, but needs larger dataset for training.

```
--block_list 234 --num_blocks 2,4,8
```
Suitable for tasks that has complex contexts and errors occurs inside ROI. More transformer blocks can help learn higher-level relationship.


Feel free to try other combinations of the hyperparameter like base_chan, reduce_size and num_blocks in each level etc. to trade off between capacity and efficiency to fit your own tasks and datasets.

#### TransUNet
We borrow code from the original TransUNet repo and fit it into our training framework. The configuration is not parsed by command line, so if you want change the configuration of TransUNet, you need change it inside the train_deep.py.
```
python train_deep.py -m TransUNet -u EXP_NAME --data_path YOUR_OWN_PATH --gpu 0
```

#### ResNet50-UTNet
For fair comparison with TransUNet, we implement the efficient attention proposed in UTNet into ResNet50 backbone, which is basically append transformer blocks into specified level after ResNet blocks.
```
python train_deep.py -m ResNet_UTNet -u EXP_NAME --data_path YOUR_OWN_PATH --reduce_size 8 --block_list 123 --num_blocks 1,1,1 --gpu 0
```
Similar to UTNet, this is the most efficient setting, suitable for tasks with limited training data.
```
--block_list 23 --num_blocks 2,4
```
Suitable for tasks that has complex contexts and errors occurs inside ROI. More transformer blocks can help learn higher-level relationship.


#### ResNet50-UNet
If you don't use Transformer blocks in ResNet50-UTNet, it is actually ResNet50-UNet. So you can use this as the baseline to compare the performance improvement from Transformer for fair comparision with TransUNet and our UTNet.

```
python train_deep.py -m ResNet_UTNet -u EXP_NAME --data_path YOUR_OWN_PATH --block_list ''  --gpu 0
```

## Citation
If you find this repo helps, please kindly cite our paper, thanks!
```
@inproceedings{gao2021utnet,
  title={UTNet: a hybrid transformer architecture for medical image segmentation},
  author={Gao, Yunhe and Zhou, Mu and Metaxas, Dimitris N},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={61--71},
  year={2021},
  organization={Springer}
}
```

