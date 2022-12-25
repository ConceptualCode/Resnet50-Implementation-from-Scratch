# Resnet50-Implementation-from-Scratch
An implementation of ResNet50 from scratch using Tensorflow

The purpose of this project is to implement the ResNet architecture, trained it on hand sign dataset and compare its result with a model pretrained on ImageNet dataset

In this project:

Dataset:


I implemented the ResNet model architecture and its building block which are:
1. Identity block
2. Convolution block and
3. The idea of skip connection or shortcut

Things to keep in mind:
1. The hand signs dataset was saved in .h5 format. So I extracted and convert to numpy to be able use it.
2. In the transfer learning path, I resized the images to (224, 224) because the original model architecture accepts (224, 224)


Reference:
Official Paper
Deeplearning.AI
