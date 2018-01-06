# Vehicle-Tracking-and-Lane-Detection
Vehicle Detection using Mask R-CNN and Computer Vision based Lane Detection

Implemented the [Mask R-CNN](https://arxiv.org/abs/1703.06870) using Keras and TensorFlow.
The model detects vehicles in the image frame using segmentation masks with the pretrained weights trained on COCO dataset; the lane detection is done using sobel filter.
This project is a part of Udacity Self-Driving Car Engineer program

Result image frames:

Mask R-CNN output on a road scene:

<img src="./output/output_1.png" width = "420" height = "250" align=center />

Land Identification output:

<img src="./input/test2.jpg" width = "420" height = "250" align=center />  <img src="./output/output_lane.png" width = "420" height = "250" align=center />

Final combined output:

<img src="./input/test6.jpg" width = "420" height = "250" align=center />  <img src="./output/output.png" width = "420" height = "250" align=center />

Download the pretrained weights [here](https://drive.google.com/open?id=1dNi9Ny1h9KBMj_3mocMQNapeEwgeERlz) and place it in the current working directory. Run main.py specifying the input image path. It can be applied with video files using moviepy and calling process_video() function.

Final Video GIF:

![](https://github.com/tj27-vkr/RCNN-Vehicle-Tracking-Lane-Detection/blob/master/output/ouput.gif)
