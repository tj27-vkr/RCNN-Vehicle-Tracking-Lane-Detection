import cv2
import numpy as np
from keras.preprocessing import image

import visualize_car_detection
import process_image as ip
import model

class InferenceConfig():
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE_SHAPES=np.array([[256, 256], [128, 128],  [ 64,  64],  [ 32,  32],  [ 16,  16]])
    BACKBONE_STRIDES=[4, 8, 16, 32, 64]
    BATCH_SIZE=1
    BBOX_STD_DEV=[ 0.1,  0.1,  0.2,  0.2]
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=0.6 #0.5
    DETECTION_NMS_THRESHOLD=0.3
    IMAGE_MAX_DIM=1024
    IMAGE_MIN_DIM=800
    IMAGE_PADDING=True
    IMAGE_SHAPE=np.array([1024, 1024,    3])
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE =0.002
    MASK_POOL_SIZE=14
    MASK_SHAPE    =[28, 28]
    MAX_GT_INSTANCES=100
    MEAN_PIXEL      =[ 123.7,  116.8,  103.9]
    MINI_MASK_SHAPE =(56, 56)
    NAME            ="coco"
    NUM_CLASSES     =81
    POOL_SIZE       =7
    POST_NMS_ROIS_INFERENCE =1000
    POST_NMS_ROIS_TRAINING  =2000
    ROI_POSITIVE_RATIO=0.33
    RPN_ANCHOR_RATIOS =[0.5, 1, 2]
    RPN_ANCHOR_SCALES =(32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE =2
    RPN_BBOX_STD_DEV  =np.array([ 0.1,  0.1,  0.2 , 0.2])
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    RPN_NMS_THRESHOLD = 0.3
    STEPS_PER_EPOCH            =1000
    TRAIN_ROIS_PER_IMAGE       =128
    USE_MINI_MASK              =True
    USE_RPN_ROIS               =True
    VALIDATION_STPES           =50
    WEIGHT_DECAY               =0.0001

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def findLane(img):
    cropped_img = ip.area_of_interest(img, [ip.crop_points.astype(np.int32)])
    trans_img  = ip.applyTransformation(cropped_img)
    masked_image = ip.applyMasks(trans_img)
    left_fit, right_fit, _ = ip.slidingWindow(masked_image)
    lane_mask = ip.applyBackTrans(img, left_fit, right_fit)
    img_result = cv2.addWeighted(img, 1, lane_mask, 1, 0)
    return img_result

def process_video(neural_net, input_img):

    img = cv2.resize(input_img, (1024, 1024))
    img = image.img_to_array(img)
    results = neural_net.detect([img], verbose=0)
    r = results[0]
    final_img = visualize_car_detection.display_instances2(img, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
    inp_shape = image.img_to_array(input_img).shape
    final_img = cv2.resize(final_img, (inp_shape[1], inp_shape[0]))

    return final_img

if __name__ == "__main__":
    img_arr    = cv2.imread('input/test2.jpg')
    config = InferenceConfig()

    #NN = model.MaskRCNN(mode="inference", model_dir='logs', config=config)
    #NN.load_weights('mask_rcnn_coco.h5', by_name=True)
    #img_result = process_video(NN,img_arr)
    #cv2.imwrite('output/output1.png', img_result)

    img_result = findLane(img_arr)
    cv2.imwrite('output/output_lane.png', img_result)
