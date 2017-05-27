# SelfCriticalSequenceTrainingforImageCaptioning
<b> TensorFlow implementation of [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563).  
General framework for reinforcement learning image caption tasks.


## References

This work is highly based on @yunjey 's implementation of [show-attend-and-tell](https://github.com/yunjey/show-attend-and-tell)  

Pytorch implementation [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)


## Getting Started

### Prerequisites

- python2.7
- tensorflow 0.12 
- numpy  
- matplotlib  
- scipy  
- scikit-image  
- hickle  
- Pillow  
- [pycocoevalcap](https://github.com/tylin/coco-caption.git)

### data

- In this work, we used [MSCOCO data set](http://mscoco.org/home/)
- Data download & process is the same as  @yunjey 's work. Please check [show-attend-and-tell](https://github.com/yunjey/show-attend-and-tell) "Getting Started" secton for details.



### Train the model 

To train the image captioning model, run command below. 

```bash
$ python train.py
```

For reference, the RL model is in ./core/model.py


## Results

After 5 epochs, our result has 5% improvement. Note that we are using different data set to test and we used different feature net.