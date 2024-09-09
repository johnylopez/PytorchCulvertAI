#!/usr/bin/python3
from datetime import datetime
import os
from typing import Tuple, Union, List
import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy
import math
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from models.culvertAI import attempt_load
import torch.nn.functional as F
from visualizer import createOutputImage


class CulvertAI:
    def __init__(self, weights,
                 device: str = "cuda"):
        self.device = device
        self.model = attempt_load(weights, width_mult=1.7, device = self.device) 
        self.model.eval()
    
    
    @torch.no_grad()
    def inference(self, img: torch.Tensor):

        img = img.unsqueeze(0)
        pred_results = self.model(img)
        
        pred_results = F.interpolate(pred_results, size=[128, 128], mode='bilinear', align_corners=False)
        detections = torch.argmax(pred_results, dim=1)
        
        return detections

class CulvertAIPublisher:
    def __init__(self, img_topic: str, weights: str, pub_topic: str = "culvertAI_detections",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 queue_size: int = 1, visualize: bool = False,
                 time: datetime = datetime.now()):
        
        self.time = datetime.now()
        self.img_size = img_size
        self.device = device

        vis_topic = pub_topic + "visualization" if pub_topic.endswith("/") else \
            pub_topic + "/visualization"

        self.visualization_publisher = rospy.Publisher(
            vis_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = CulvertAI(
            weights=weights, device=device
        )

        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )
        rospy.loginfo("CulvertAI initialization complete. Ready to start inference")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape

        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)
        
        # inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections_ = createOutputImage(detections[0])


        output = cv2.addWeighted(np_img_resized, 1, detections_, 1,0)
        
        # FPS AND CONFIDENCE SCORE
        os.system('clear')
        timediff = datetime.now() - self.time
        self.time = datetime.now()
        fps = round(1.0 / timediff.total_seconds(),2)
        print("FPS: " + str(fps))
        

        # # # visualizing if required
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
            self.visualization_publisher.publish(vis_msg)
        


if __name__ == "__main__":

    rospy.init_node("culvertAI_node")

    ns = rospy.get_name() + "/"

    weights_path = rospy.get_param(ns + "weights_path")
    img_topic = rospy.get_param(ns + "img_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    img_size = rospy.get_param(ns + "img_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError(f"Weights not found ({weights_path}).")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")


    publisher = CulvertAIPublisher(
        img_topic=img_topic,
        pub_topic=out_topic,
        weights=weights_path,
        device=device,
        visualize=visualize,
        img_size=(img_size, img_size),
        queue_size=queue_size,
    )

    rospy.spin()
