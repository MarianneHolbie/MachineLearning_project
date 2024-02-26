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
        image_height, image_width = image_size

        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:

            grid_height, grid_width, nbr_anchor, _ = output.shape
            for row in range(grid_height):
                for col in range(grid_width):
                    for a in range(nbr_anchor):
                        t_x, t_y, t_w, t_h = output[row, col, a, :4]

                        anchor_width = self.anchors[a][0]
                        anchor_height = self.anchors[a][1]

                        # absolute coordinate of box
                        x1 = (col + tf.sigmoid(t_x)) / grid_width
                        y1 = (row + tf.sigmoid(t_y)) / grid_height
                        w = anchor_width * np.exp(t_w) / image_width
                        h = anchor_height * np.exp(t_h) / image_height

                        # conv in pixel
                        x1 = (x1 - w / 2) * image_width
                        y1 = (y1 - h / 2) * image_height
                        x2 = (x1 + w) * image_width
                        y2 = (y1 + h) * image_height

                        boxes.append([x1, y1, x2, y2])
                        box_confidences.append(output[row, col, a, 4])
                        box_class_probs.append(output[row, col, a, 5:])

        return (boxes, box_confidences, box_class_probs)
