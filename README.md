# EVA6_Session9_resnets_and_higher_receptive_fields

## Learning Time

### One Cycle Policy

In the paper [“A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay”](https://arxiv.org/abs/1803.09820), Leslie Smith describes the approach to set hyper-parameters (namely learning rate, momentum and weight decay) and batch size. In particular, he suggests 1 Cycle policy to apply learning rates.

The author recommends doing one cycle of learning rate of 2 steps of equal length. We choose the maximum learning rate using a range test. We use a lower learning rate as 1/5th or 1/10th of the maximum learning rate. We go from a lower learning rate to a higher learning rate in step 1 and back to a lower learning rate in step 2. We pick this cycle length slightly lesser than the total number of epochs to be trained. And in the last remaining iterations, we annihilate the learning rate way below the lower learning rate value(1/10 th or 1/100 th).

The motivation behind this is that, during the middle of learning when the learning rate is higher, the learning rate works as a regularisation method and keep the network from overfitting. This helps the network to avoid steep areas of loss and land better flatter minima.

![image1](/images_and_logs/one_cycle_lr_image1.png)

As in the figure, We start at a learning rate 0.08 and make a step of 41 epochs to reach a learning rate of 0.8, then make another step of 41 epochs where we go back to a learning rate 0.08. Then we make another 13 epochs to reach 1/10th of lower learning rate bound(0.08).With CLR 0.08–0.8, batch size 512, momentum 0.9 and Resnet-56, we got ~91.30% accuracy in 95 epochs on CIFAR-10.

Momentum and learning rate are closely related. It can be seen in the weight update equation for SGD that the momentum has a similar impact as the learning rate on weight updates. The author found in their experiments that reducing the momentum when the learning rate is increasing gives better results. This supports the intuition that in that part of the training, we want the SGD to quickly go in new directions to find a better minima, so the new gradients need to be given more weight.

![image2](/images_and_logs/one_cycle_lr_image2.png)

In practice, we choose 2 values for momentum. As in One Cycle, we do 2 step cycle of momentum, wherein step 1 we reduce momentum from higher to lower bound, and in step 2 we increase momentum from lower to higher bound. According to the paper, this cyclic momentum gives the same final results, but this saves time and effort of running multiple full cycles with different momentum values. With One Cycle Policy and cyclic momentum, I could replicate the results mentioned in the paper. Where the model achieved 91.54% accuracy in 9310 iterations while using one cycle with learning rates 0.08–0.8 and momentum 0.95–0.80 with resnet-56 and batch size of 512, while without CLR it requires around 64k iterations to achieve this accuracy. ( Paper achieved 92.0 ± 0.2 accuracies). 

Does it provide us higher accuracy in practice? The answer, sadly, is NO.

## Objectives

* [x] To write a custom ResNet architecture for CIFAR10 that has the following architecture:
  * [x] PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  * [x] Layer1 -
    * [x] X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    * [x] R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    * [x] Add(X, R1)
  * [x] Layer 2 -
    * [x] Conv 3x3 [256k]
    * [x] MaxPooling2D
    * [x] BN
    * [x] ReLU
  * [x] Layer 3 -
    * [x] X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    * [x] R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    * [x] Add(X, R2)
  * [x] MaxPooling with Kernel Size 4
  * [x] FC Layer 
  * [x] SoftMax
* [x] Uses One Cycle Policy such that:
  * [x] Total Epochs = 24
  * [x] Max at Epoch = 5
  * [x] LRMIN = _0_
  * [x] LRMAX = _0.05501043841336117_
  * [x] NO Annihilation
* [x] Uses the following transforms, in order: 
  * [x] RandomCrop 32, 32 (after padding of 4) 
  * [x] FlipLR 
  * [x] CutOut(8, 8)
* [x] Batch size = 512
* [ ] Target Accuracy: 93% _(Achieved 92.09%)_

## Results

1. Model: Custom ResNet18
  2. Total Train data: 60,000 | Total Test Data: 10,000
  3. Total Parameters: 6,573,130
  4. Test Accuracy: 92.09%
  5. Epochs: Run till 24 epochs
  6. Normalization: Batch Normalization
  7. Regularization: L2 with factor 0.002
  8. Optimizer: SGD
  9. Loss criterion: Cross Entropy
  10. Scheduler: OneCycleLR
  11. Albumentations: 
      1. RandomCrop(32, padding=4)
      2. HorizontalFlip
      3. CoarseDropout(8x8)
      6. Normalization 
   12. Misclassified Images: 791 images were misclassified out of 10,000
  
## Code Structure

* [custom_resnet.py](https://github.com/RohinSequeira/pytorch_cifar10/blob/main/model/custom_resnet.py): Our version of a custom Resnet Architecture.


* [utils](https://github.com/RohinSequeira/pytorch_cifar10/blob/main/utils/utils.py): Utils code contains the following components:-  
  1. Data Loaders  
  2. Albumentations  
  3. Accuracy Plots
  4. Misclassification Image Plots
  5. Seed

* [lr_finder](https://github.com/RohinSequeira/pytorch_cifar10/blob/main/utils/lr_finder.py): Return the best max_lr for OneCycleLR.

* [main.py](https://github.com/RohinSequeira/pytorch_cifar10/blob/main/main.py): Main code contains the following functions:-  
  1. Train code
  2. Test code
  3. Main function for training and testing the model  

* [Colab file](/pytroch_custom_resnet.ipynb): This Google Colab file contains rundown of the execution of our code. Check it out for more information. 
 

## Model Summary

![model_summary](/images_and_logs/model_summary.JPG)

## Train and Test Logs

```------------------------------------------

Epoch 1 : 
Train set: Average loss: 1.3702, Accuracy: 32.88

Test set: Average loss: 0.003, Accuracy: 45.80

Epoch 2 : 
Train set: Average loss: 1.2244, Accuracy: 52.07

Test set: Average loss: 0.002, Accuracy: 60.82

Epoch 3 : 
Train set: Average loss: 1.0451, Accuracy: 61.94

Test set: Average loss: 0.002, Accuracy: 69.26

Epoch 4 : 
Train set: Average loss: 0.8514, Accuracy: 70.11

Test set: Average loss: 0.002, Accuracy: 72.19

Epoch 5 : 
Train set: Average loss: 0.6583, Accuracy: 75.31

Test set: Average loss: 0.002, Accuracy: 67.19

Epoch 6 : 
Train set: Average loss: 0.4510, Accuracy: 78.86

Test set: Average loss: 0.002, Accuracy: 72.43

Epoch 7 : 
Train set: Average loss: 0.4529, Accuracy: 81.01

Test set: Average loss: 0.001, Accuracy: 78.88

Epoch 8 : 
Train set: Average loss: 0.5118, Accuracy: 83.11

Test set: Average loss: 0.001, Accuracy: 78.03

Epoch 9 : 
Train set: Average loss: 0.4669, Accuracy: 84.90

Test set: Average loss: 0.001, Accuracy: 81.38

Epoch 10 : 
Train set: Average loss: 0.4460, Accuracy: 86.06

Test set: Average loss: 0.001, Accuracy: 81.36

Epoch 11 : 
Train set: Average loss: 0.4558, Accuracy: 87.04

Test set: Average loss: 0.001, Accuracy: 81.88

Epoch 12 : 
Train set: Average loss: 0.3746, Accuracy: 87.90

Test set: Average loss: 0.001, Accuracy: 82.37

Epoch 13 : 
Train set: Average loss: 0.3672, Accuracy: 88.37

Test set: Average loss: 0.001, Accuracy: 82.19

Epoch 14 : 
Train set: Average loss: 0.3021, Accuracy: 88.95

Test set: Average loss: 0.001, Accuracy: 86.21

Epoch 15 : 
Train set: Average loss: 0.3010, Accuracy: 89.92

Test set: Average loss: 0.001, Accuracy: 84.96

Epoch 16 : 
Train set: Average loss: 0.2863, Accuracy: 90.75

Test set: Average loss: 0.001, Accuracy: 83.70

Epoch 17 : 
Train set: Average loss: 0.2100, Accuracy: 91.34

Test set: Average loss: 0.001, Accuracy: 85.48

Epoch 18 : 
Train set: Average loss: 0.2264, Accuracy: 92.79

Test set: Average loss: 0.001, Accuracy: 85.13

Epoch 19 : 
Train set: Average loss: 0.1590, Accuracy: 94.05

Test set: Average loss: 0.001, Accuracy: 89.68

Epoch 20 : 
Train set: Average loss: 0.1448, Accuracy: 95.31

Test set: Average loss: 0.001, Accuracy: 89.41

Epoch 21 : 
Train set: Average loss: 0.1201, Accuracy: 96.59

Test set: Average loss: 0.001, Accuracy: 90.90

Epoch 22 : 
Train set: Average loss: 0.0752, Accuracy: 97.69

Test set: Average loss: 0.000, Accuracy: 91.91

Epoch 23 : 
Train set: Average loss: 0.0556, Accuracy: 98.30

Test set: Average loss: 0.000, Accuracy: 92.01

Epoch 24 : 
Train set: Average loss: 0.0630, Accuracy: 98.48

Test set: Average loss: 0.000, Accuracy: 92.09
```

## Plots

  1. Sample Transformed Input Images

  ![sample_input_images](/images_and_logs/sample_transformed_input_images.png)

  2. Max LR plot
  
  ![max_lr](/images_and_logs/max_lr_graph.png)

  3. Train & Test Loss, Train & Test Accuracy  
  
  ![loss_and_accuracy](/images_and_logs/loss_and_accuracy_graphs.png)

  4. Misclassified Images  
  
  ![loss_and_accuracy](/images_and_logs/misclassified_images.png)
  

## Collaborators
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira
