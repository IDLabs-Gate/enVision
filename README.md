# enVision
Deep Learning Models for Vision Tasks on iOS

![sample] (https://github.com/IDLabs-Gate/enVision/blob/master/sample2.jpg)

## Usage
Download [dependencies] folder tf
[dependencies]:https://drive.google.com/open?id=0B7JMhWoJ8WpUNW9wYS1tRVI0dlk
Extract all archives in tf/models and tf/lib

Put tf folder in same directory level as enVision project folder

Build and Run

Press screen to change running model

Tap a data slot below to select, then tap a detection box to snap

Tap a data slot with two fingers to remove last snap

Press a data slot to clear

## Models

#### YOLO:
https://arxiv.org/abs/1506.02640

![sample2](https://github.com/IDLabs-Gate/enVision/blob/master/sample1.jpg)

YOLO 1 tiny (VOC): Best performance on basic classes

YOLO 1 small (VOC): Better accuracy for basic classes

YOLO 1.1 tiny (COCO): Fast on extended classes

YOLO 2 (COCO): Best accuracy on extended classes

YOLO detector + Jetpac feature extractor from snaps + kNN classifier with Euclidean distance
.

#### FaceNet:
https://arxiv.org/abs/1503.03832

![sample3](https://github.com/IDLabs-Gate/enVision/blob/master/sample3.jpg)

Inception-Resnet-v1 (FaceScrub and CASIA-Webface)

Native iOS Face detector + FaceNet feature extractor from snaps + kNN classifier with Euclidean distance
.

#### Inception: 
https://arxiv.org/abs/1512.00567

Inception v3 (ImageNet)

Can run retrained models too

.

#### Jetpac:
https://github.com/jetpacapp/DeepBeliefSDK

Jetpac network (ImageNet)

DeepBeliefSDK framework


## License
####MIT License

Owner: ID Labs L.L.C.

Original Contributor: Muhammad Hilal


