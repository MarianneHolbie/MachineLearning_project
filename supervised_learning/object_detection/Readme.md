# Object detection

The goal of this project is to detect objects in images and videos. We will use the YOLO algorithm, which is a state-of-the-art, real-time object detection system.

## Tasks
| Task                                | Description                                                                                                                    |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [0. Initialize Yolo](0-yolo.py)     | Write a class `Yolo` that uses the Yolo v3 algorithm to perform object detection.                                              |
| [1. Process Outputs](1-yolo.py)     | Update the class `Yolo` by adding the public method `def process_outputs(self, outputs, image_size):`.                         |
| [2. Filter Boxes](2-yolo.py)        | Update the class `Yolo` by adding the public method `def filter_boxes(self, boxes, box_confidences, box_class_probs):`.        |
| [3. Non-max Suppression](3-yolo.py) | Update the class `Yolo` by adding the public method `def non_max_suppression(self, filtered_boxes, box_classes, box_scores):`. |
| [4. Load Images](4-yolo.py)         | Update the class `Yolo` by adding the public method `def load_images(folder_path):`.                                           |
| [5. Preprocess Images](5-yolo.py)   | Update the class `Yolo` by adding the public method `def preprocess_images(self, images):`.                                    |
| [6. Show Boxes](6-yolo.py)          | Update the class `Yolo` by adding the public method `def show_boxes(self, image, boxes, box_classes, box_scores, file_name):`. |
| [7. Predict](7-yolo.py)             | Update the class `Yolo` by adding the public method `def predict(self, folder_path):`.                                         |