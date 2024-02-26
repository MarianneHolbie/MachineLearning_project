#!/usr/bin/env python3
"""
    Initialize Yolo
"""
import tensorflow.keras as K


def load_model(model_path):
    """
        function to load a model
        :param model_path: path where model can be found

        :return: model
    """
    return K.models.load_model(model_path)


def extract_class(classes_path):
    """
        function to extract list of class from text file

        :param classes_path: path where text file containing classes

        :return: list of classes
    """
    classes_name = []
    with open(classes_path, 'r') as f:
        for line in f:
            line = line.strip()
            classes_name.append(line)

    return classes_name


class Yolo:
    """
        Class Yolo
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
            Initialize Class Yolo uses Yolo v3 algorithm

            :param self:
            :param model_path: path where Darknet Keras model is stored
            :param classes_path:path where list of class names, in order of index
            :param class_t: float, box score threshold for initial filtering step
            :param nms_t: float, IOU threshold for non-max suppression
            :param anchors: ndarray, shape(outputs, anchor_boxes, 2) all anchor boxes
                outputs: number of outputs (prediction) made by Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2: [anchor_box_width, anchor_box_height]

            :return:
        """
        self.model = load_model(model_path)
        self.class_names = extract_class(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
