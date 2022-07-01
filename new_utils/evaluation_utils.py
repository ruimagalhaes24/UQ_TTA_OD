from collections import defaultdict
import os
from matplotlib.pyplot import box
import torch
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
def get_per_frame_preprocessed_instances():
    preprocess_gt_instances()
    preprocess_prediction_intances()
    return preprocessed_gt_instances, preprocessed_predicted_instances
"""
#Function that reads json file with labels from bdd and preprocesses them
#To obtain relevant information for metrics calculation
#Basically: For each image,  ground truth bboxes variables and associated classes.
def get_preprocess_ground_truth_instances(path_to_dataset_labels):
    try:
        #Load previously processed ground truth instances
        preprocessed_gt_instances = torch.load(
            os.path.join(path_to_dataset_labels,
            "preprocessed_gt_instances.pth"), 
            map_location=device)
        return preprocessed_gt_instances
    except FileNotFoundError:
        #If file does not exist yet, preprocess gt instances and save it.

        #Load json file with labels
        gt_info = json.load(
            open(os.path.join(path_to_dataset_labels,"val_coco_format.json"),"r")
            )
        #Get annotations from json file (list with all the ground truth bounding boxes)
        #Each Bbox has 4 variables for its location, the class, and the image associated with   
        gt_instances = gt_info['annotations']

        gt_boxes, gt_cat_idxs = defaultdict(torch.Tensor), defaultdict(torch.Tensor)
        for gt_instance in gt_instances:
            box_inds = gt_instance['bbox']
            #He does this transformation from [x,y,w,h] to [x1,y1,x2,y2] to both gt and predictions
            #Must be relevant/easier for metrics calculations
            box_inds = np.array([box_inds[0],
                                 box_inds[1],
                                 box_inds[0] + box_inds[2],
                                 box_inds[1] + box_inds[3]])
            #Append the new bbox instance to the list
            gt_boxes[gt_instance['image_id']] = torch.cat((gt_boxes[gt_instance['image_id']].cuda() , torch.as_tensor([box_inds],dtype=torch.float32).to(device)))

            gt_cat_idxs[gt_instance['image_id']] = torch.cat((gt_cat_idxs[gt_instance['image_id']].cuda() , torch.as_tensor([[gt_instance['category_id']]],dtype=torch.float32).to(device)))

        preprocessed_gt_instances = dict({'gt_boxes': gt_boxes,
                                          'gt_cat_idxs': gt_cat_idxs})

        torch.save(preprocessed_gt_instances,
        os.path.join(path_to_dataset_labels,
            "preprocessed_gt_instances.pth"),
        )
        return preprocessed_gt_instances