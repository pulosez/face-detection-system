import unittest
from workspace.detector import *
from cfg.global_cfg import TEST_INPUT_DATA_PATH, THRESHOLD


class TestDetectorMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = Detector()
        self.detector.load_model()

    def test_model_name(self):
        self.assertEqual(self.detector.model_name, 'face_detection_model')

    def test_load_model(self):
        self.assertTrue(self.detector.model is not None)

    def test_predict_image(self):
        path = str(TEST_INPUT_DATA_PATH) + '/Condoleezza_Rice_0008.jpg'
        prediction = self.detector.predict_image(image_path=path, threshold=THRESHOLD)
        self.assertTrue(prediction is not None)

    def test_predict_confidence(self):
        image = cv2.imread(filename=str(TEST_INPUT_DATA_PATH) + '/Condoleezza_Rice_0008.jpg')
        prediction, confidence = self.detector.create_bounding_box(image=image, threshold=THRESHOLD)
        self.assertEqual(len(confidence), 1)

    def test_count_faces(self):
        image = cv2.imread(filename=str(TEST_INPUT_DATA_PATH) + '/Condoleezza_Rice_0008.jpg')
        prediction, confidence = self.detector.create_bounding_box(image=image, threshold=THRESHOLD)
        self.assertEqual(confidence[0], 100)


if __name__ == '__main__':
    unittest.main()
