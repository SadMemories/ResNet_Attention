# ResNet_Attention(CBAM)

Official instruction: [CBAM](https://arxiv.org/abs/1807.06521)

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
# To trian with CBAM  
python train_CIFAR10.py --prefix 10 --att-type cbam  
```
# Validation Result
* ResNet50      (trained for 160 epochs) ACC@1=77.88% ACC@5=98.4%
* ResNet50+CBAM (trained for 160 epochs) ACC@1=77.81% ACC@5=98.55%
