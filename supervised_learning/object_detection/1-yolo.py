#!/usr/bin/env python3
"""
    Initialize Yolo
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
        Class Yolo uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
            Class constructor of Yolo class

            :param model_path: path where Darknet Keras model is stored
            :param classes_path:path where list of class names,
                in order of index
            :param class_t: float, box score threshold
                for initial filtering step
            :param nms_t: float, IOU threshold for non-max suppression
            :param anchors: ndarray, shape(outputs, anchor_boxes, 2)
                all anchor boxes
                outputs: number of outputs (prediction) made by Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2: [anchor_box_width, anchor_box_height]

        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.class_names.append(line)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
            Function to process outputs

        :param outputs: list of ndarray, predictions from a single image
                each output,
                shape(grid_height, grid_width, anchor_boxes, 4+1+classes)
                grid_height, grid_width: height and width of grid
                 used for the output
                anchor_boxes: number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => classes probabilities for all classes
        :param image_size: ndarray,
               image's original size [image_height, image_width]

        :return: tuple (boxes, box_confidences, box_class_probs):
                boxes: list of ndarrays,
                       shape(grid_height, grid_width, anchor_boxes, 4)
                        processed boundary boxes for each output
                        4 => (x1,y1, x2, y2)
                boxe_confidences: list ndarray,
                    shape(grid_height, grid_width, anchor_boxes, 1)
                    boxe confidences for each output
                box_class_probs: list ndarray,
                    shape(grid_height, grid_width, anchor_boxes, classes)
                    box's class probabilities for each output
        """
        # extract image size
        image_height, image_height = image_size

        boxes = []
        box_confidences = []
        box_class_probs = []

        # process for each output
        for idx, output in enumerate(outputs):

            # extract height, width, number of anchor box for current output
            grid_height, grid_width, nbr_anchor, _ = output.shape

            # extract coordinate of output NN
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            # grid coordinate
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))

            # Repeat grid coordinate for each anchor box
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            # extract anchor_box_width, anchor_box_height
            p_w = self.anchors[idx, :, 0]
            p_h = self.anchors[idx, :, 1]

            # size image
            image_height, image_width = image_size

            # sigmoid : grid scale (value between 0 and 1)
            # + c_x or c_y : coordinate of cells in the grid
            b_x = ((1.0 / (1.0 + np.exp(-t_x))) + grid_x) / grid_width
            b_y = ((1.0 / (1.0 + np.exp(-t_y))) + grid_y) / grid_height
            # exp for predicted height and width
            b_w = p_w * np.exp(t_w)
            b_w /= self.model.input.shape[1].value
            b_h = p_h * np.exp(t_h)
            b_h /= self.model.input.shape[2].value

            # conv in pixel : absolute coordinate
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_w / 2 + b_x) * image_width
            y2 = (b_h / 2 + b_y) * image_height

            # Update box array with box coordinates and dimensions
            box = np.zeros((grid_height, grid_width, nbr_anchor, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidences = output[:, :, :, 4:5]
            sigmoid_confidence = 1 / (1 + np.exp(-confidences))
            class_probs = output[:, :, :, 5:]
            sigmoid_class_probs = 1 / (1 + np.exp(-class_probs))

            box_confidences.append(sigmoid_confidence)
            box_class_probs.append(sigmoid_class_probs)

        return boxes, box_confidences, box_class_probs
