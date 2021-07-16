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

* [ ] To write a custom ResNet architecture for CIFAR10 that has the following architecture:
  * [ ] PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  * [ ] Layer1 -
    * [ ] X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    * [ ] R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    * [ ] Add(X, R1)
  * [ ] Layer 2 -
    * [ ] Conv 3x3 [256k]
    * [ ] MaxPooling2D
    * [ ] BN
    * [ ] ReLU
  * [ ] Layer 3 -
    * [ ] X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    * [ ] R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    * [ ] Add(X, R2)
  * [ ] MaxPooling with Kernel Size 4
  * [ ] FC Layer 
  * [ ] SoftMax
* [ ] Uses One Cycle Policy such that:
  * [ ] Total Epochs = 24
  * [ ] Max at Epoch = 5
  * [ ] LRMIN = _To be calculated_
  * [ ] LRMAX = _To be calculated_
  * [ ] NO Annihilation
* [ ] Uses the following transforms, in order: 
  * [ ] RandomCrop 32, 32 (after padding of 4) 
  * [ ] FlipLR 
  * [ ] CutOut(8, 8)
* [ ] Batch size = 512
* [ ] Target Accuracy: 93%

## Results

  
## Code Structure

* [custom_resnet.py](https://github.com/RohinSequeira/pytorch_cifar10/blob/main/model/custom_resnet.py): Our version of a custom Resnet Architecture.


* [utils](https://github.com/Arijit-datascience/pytorch_cifar10/blob/main/utils/utils.py): Utils code contains the following components:-  
  1. Data Loaders  
  2. Albumentations  
  3. Accuracy Plots
  4. Misclassification Image Plots
  5. Seed

* [main.py](https://github.com/Arijit-datascience/pytorch_cifar10/blob/main/main.py): Main code contains the following functions:-  
  1. Train code
  2. Test code
  3. Main function for training and testing the model  

* [Colab file](/pytroch_custom_resnet.ipynb): This Google Colab file contains rundown of the execution of our code. Check it out for more information. 
 

## Model Summary

![model_summary](/images_and_logs/model_summary.JPG)

## Plots

  1. Train & Test Loss, Train & Test Accuracy  
  
  ![loss_and_accuracy](/images_and_logs/loss_and_accuracy_graphs.png)

  2. Misclassified Images  
  
  ![loss_and_accuracy](/images_and_logs/misclassified_images.png)
  

## Collaborators
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira
