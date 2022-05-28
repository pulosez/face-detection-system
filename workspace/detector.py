import os
import cv2
import numpy as np
import tensorflow as tf
from cfg.global_cfg import MODEL_PATH, THRESHOLD, logger


class Detector:
    def __init__(self):
        self.model = None
        self.model_name = 'face_detection_model'

    def load_model(self):
        """
        function to load face detection model
        """
        logger.info('loading model ' + self.model_name + '...')
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(MODEL_PATH, self.model_name, 'saved_model'))
        logger.info(f'Model {self.model_name} is loaded successfully')

    def create_bounding_box(self, image: cv2.imread, threshold: float = THRESHOLD) -> np.ndarray:
        """
        function to create bounding box for analyzing image where face detection system was found faces
        :param image: image for analyze by face-detection-system
        :param threshold: minimum value of confidence for detected objects
            which will be suitable to call this object true
        :return: np.ndarray object with drawn bounding box
        """
        detections = self.model(tf.convert_to_tensor(
            cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), dtype=tf.uint8)[tf.newaxis, ...])
        bboxes = detections['detection_boxes'][0].numpy()
        class_scores = detections['detection_scores'][0].numpy()
        bbox_ids = tf.image.non_max_suppression(
            bboxes, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
        if len(bbox_ids) != 0:
            for i in bbox_ids:
                display_text = '{}: {}%'.format('FACE', round(100 * class_scores[i]))
                self.build_rectangle(image=image, bbox=bboxes[i], display_text=display_text)
        return image

    @staticmethod
    def build_rectangle(image, bbox, display_text):
        """
        function to draw rectangle and display text and put it on the image
        :param image: image for detecting face
        :param bbox: coordinates for detection rectangle
        :param display_text: text with confidence
        """
        height, width, color = image.shape
        y_min, x_min, y_max, x_max = tuple(bbox.tolist())
        x_min, x_max, y_min, y_max = int(x_min * width), int(x_max * width), \
                                     int(y_min * height), int(y_max * height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=1)
        cv2.putText(image, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    def predict_image(self, image_path: str, threshold: float = 0.5) -> np.ndarray:
        """
        function to run face prediction process
        :param image_path: path to the image file for analyzing
        :param threshold: minimum value of confidence for detected objects
            which will be suitable to call this object true
        :return: np.ndarray object with drawn bounding box
        """
        image = cv2.imread(filename=image_path)
        bbox_image = self.create_bounding_box(image=image, threshold=threshold)
        return bbox_image
