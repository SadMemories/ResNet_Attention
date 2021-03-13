# ResNet_Attention(CBAM，SE)

Official instruction: [CBAM](https://arxiv.org/abs/1807.06521)，[SE](https://arxiv.org/abs/1709.01507)

# Required Environment
Ubuntu20.04  
GTX 1080Ti  
Python3.7  
PyTorch1.7.0  
CUDA10.2  
CuDNN7.0

# Usage Method(trian with CIFAR10)
The model's backbone is ResNet. In our training, we use CIFAR10 as our dataset.  
```
# To train with Se
 python train_CIFAR10.py --prefix 4 --device 1 --epoch 160 --att_type se
# To trian with CBAM  
 python train_CIFAR10.py --prefix 5 --device 1 --epoch 160 --att_type cbam
```
# Validation Result
* ResNet50         (trained for 160 epochs) ACC@1=93.41% ACC@5=99.84%
  * ResNet50+SE (trained for 160 epochs) ACC@1=94.01% ACC@5=99.89%

# Result Graph

**Blue：ResNet50**

**Red：ResNet50+SE**

![image-20210313162513690](C:\Users\云之遥\AppData\Roaming\Typora\typora-user-images\image-20210313162513690.png)

![image-20210313162527033](C:\Users\云之遥\AppData\Roaming\Typora\typora-user-images\image-20210313162527033.png)