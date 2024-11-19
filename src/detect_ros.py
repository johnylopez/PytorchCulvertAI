#!/usr/bin/python3
from datetime import datetime
import os
from typing import Tuple, Union, List
import torch
import cv2
import numpy as np
import rospy
import torch.nn.functional as F

from skimage.transform import resize
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from models.culvertAI import attempt_load
from visualizer import createOutputImage

#MODEL CLASS
class CulvertAI:
    # Parameters
    # Weights: path to Pytorch pretained pth model [String]
    # device: Device where to load the model, either "cuda" or "cpu" [String]

    def __init__(self, weights,
                 device: str = "cuda"):
        self.device = device
        #Load of the model using the load function, Parameters: weight path, width_mult
        self.model = attempt_load(weights, width_mult=3.0, device = self.device) 
        self.model.eval() #Set model to evaluation mode improving inference time
    
    
    #INFERENCE FUNCTION
    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        # PARAMETERS
        # img: the image feed to the network for predictions. [torch tensor]

        img = img.unsqueeze(0) #(H, W, C) to (1, H, W, C)
        pred_results = self.model(img) #Feed image to the model, will return the prediction results
        pred_results = F.interpolate(pred_results, size=[128, 128], mode='bilinear', align_corners=False) #Resize output size 
        detections = torch.argmax(pred_results, dim=1) #Obtain predicted class
        
        return detections


#CulvertAIPublisher Class
class CulvertAIPublisher:
    def __init__(self, img_topic: str, weights: str, pub_topic: str = "culvert_ai/visualization",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (128, 128),
                 queue_size: int = 1, visualize: bool = False,
                 time: datetime = datetime.now()):
        
        # Paremeters:
        # img_topic: path to ros image topic [String]
        # weights: path to Pytorch pretained pth model [String]
        # pub_topic: ros topic where to publish output [String]
        # device: Device where to load tensors, either "cuda" or "cpu" [String]
        # img_size: Size of images accepted by the Pytorch model. Callback function will resize to this size before inference. [(int, int)]
        # queue_size: Queue size for publishing [int]
        # visualize: Flag indicating wheter we want the results to be published. [Boolean]
       
        self.time = datetime.now()
        self.img_size = img_size
        self.device = device

        #Creation and initialization of ROS Publisher if visualize flag is true
        self.visualization_publisher = rospy.Publisher(
            pub_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.detections_publisher = rospy.Publisher('culvert_ai/detections', String, queue_size=10)

        #Bridge betwwen ROS image message and OpenCV image. Will convert ros images to cv2 images
        self.bridge = CvBridge()

        #Initialization of the Model 
        self.model = CulvertAI(
            weights=weights, device=device
        )

        """
        Initialization of the ROS Subscriber, it subscribes to the ROS image topic and it will call the callback function
        every time a new image is being published by the image topic
        """
        
        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )

        rospy.loginfo("CulvertAI initialization complete. Ready to start inference")

    #Callback function for the subscriber
    def process_img_msg(self, img_msg: Image):
        #Bridge the image message to a CV2 image
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        # Resize the image to the desired state
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # Conversion to torch tensor, normalization, and loading to GPU device
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img)) #Create torch tensor from numpy array
        img = img.float()  # uint8 to fp16/32
        img /= 255  #Normalization 0 - 255 to 0.0 - 1. 
        img = img.to(self.device) #Load torch tensor to GPU device
        
        # Run the inference, create output image and merging with original image
        detections = self.model.inference(img)
        output, detections = createOutputImage(detections[0], np_img_resized)
        
        # FPS AND CONFIDENCE SCORE
        # os.system('clear')
        # timediff = datetime.now() - self.time
        # self.time = datetime.now()
        # fps = round(1.0 / timediff.total_seconds(),2)
        # print("FPS: " + str(fps))
        
        msg = String()
        msg.data = ', '.join(detections)
        self.detections_publisher.publish(msg)

        # Visualization if visualization parameter is true
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
            self.visualization_publisher.publish(vis_msg)
        



if __name__ == "__main__":

    #Initialization of the Culvert AI ROS Node
    rospy.init_node("culvertAI_node")

    #Creation of the ros package path
    ns = rospy.get_name() + "/"

    #Setting the paths of different parameters
    weights_path = rospy.get_param(ns + "weights_path")
    img_topic = rospy.get_param(ns + "img_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    img_size = rospy.get_param(ns + "img_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    #Check if model weights exists
    if not os.path.isfile(weights_path):
        raise FileExistsError(f"Weights not found ({weights_path}).")

    #Set device to CUDA if GPU is present
    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    #Creation and Initialization of ROS Publisher
    publisher = CulvertAIPublisher(
        img_topic=img_topic,
        pub_topic=out_topic,
        weights=weights_path,
        device=device,
        visualize=visualize,
        img_size=(img_size, img_size),
        queue_size=queue_size,
    )

    #Mantain the ROS Node running
    rospy.spin()
