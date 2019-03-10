Faster-RCNN using PyTorch
====
  This code contains the training and test of Faster-RCNN

## Contents
* [Train](#Train)
* [Test](#Test)


## Train
Training process is implemented in train_frcnn.py
* train
  - You need to realize your own data_loader in roidb. deepdrive_data_loader.py is implemented for DeepDrive of AIChallenge dataset. You can also change it into VOC data loader or COCO data loader by overwriting get_minibatch()
  - More details in train_frcnn.py. The default number of epochs is 50, and the learning rate decays by 0.1 at epoch 30. You can change it in train()
  - The first 100 images containing the foreground anchors are visualized iin 'vis_anchors'. This is a test for generating anchors.

* Test
  - Put the testing image in to folder 'images' and modify inference.py
  - You can visualize proposals output by RPN by uncommenting draw_proposals at line 181

* Results
  - Detection
![Load failed](https://github.com/yfji/pytorch-faster-rcnn/blob/master/detect.jpg?raw=true)
  - Proposals
![Load failed](https://github.com/yfji/pytorch-faster-rcnn/blob/master/proposals.jpg?raw=true)