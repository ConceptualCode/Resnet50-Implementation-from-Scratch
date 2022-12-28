# Resnet50-Implementation-from-Scratch
An implementation of ResNet50 from scratch using Tensorflow

The purpose of this project is to implement the ResNet architecture from scratch, train it on hand sign dataset and compare its result with a model pretrained on ImageNet dataset

In this project:

- I used the hand sign dataset, here is the link https://drive.google.com/drive/folders/1ZKXsUcl_rUXQtRqmMCY2VWC_d1bUXY4f?usp=share_link
- I implemented the ResNet model architecture and its building block which are:
  - Identity block
  - Convolution block and
  - The idea of skip connection or shortcut
  
In the transfer learning path, I froze the base layer, removed the top layers, and added some custom top layers to suit my problem statement.
Initially, I noticed that the model was underfitting the train data because I used only few layers to train, I had 30% correct prediction on the train data. Then I was just using a GlobalAveragePooling2D, a Dense layer of 1024 units, and the prediction layer of 6 units.
Then I added more features to the model with more layers: I added 4 more Dense layers with 3024, 2024, 1024, and 512 units respectively before the prediction layer.
For the resnet50 model implemented from scratch, I trained for 10 epochs while the pretrained model for 15 epochs. My goal is to show the implementation from scratch, compaare the result with the pretrained model, and
show how to solve underfitting. I am not working to achieve a high accuracy model.

However, I could do that in the future.

In this project, I practically show that adding more layers to model can help to solve underfitting on train data. 
Reference:
- Official Paper @ https://arxiv.org/abs/1512.03385
- Deeplearning.AI Course on Coursera
