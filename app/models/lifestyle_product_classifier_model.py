import os
import numpy as np
import onnxruntime
import cv2

MODELS_DIRECTORY = 'models'
APP_DIRECTORY = 'app'

class ImageClassifierModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        model_path = os.path.join(APP_DIRECTORY, MODELS_DIRECTORY, f'{self.model_name}.onnx')
        return onnxruntime.InferenceSession(model_path)

    def predict(self, input_image):
        preprocessed_image = self._preprocess(input_image)
        output = self.model.run(None, {"conv2d_input": preprocessed_image})
        predicted_class = np.round(output)[0][0]
        return False if predicted_class == 1 else True


    @staticmethod
    def _preprocess(image_bytes, target_size=(224, 224)):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        image = np.expand_dims(image, axis=0)
        return image

        