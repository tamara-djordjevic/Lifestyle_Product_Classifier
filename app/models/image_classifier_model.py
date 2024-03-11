import os
import numpy as np
import onnxruntime
from torchvision import transforms

MODELS_DIRECTORY = 'models'
APP_DIRECTORY = 'app'

LABEL_ENCODER_MAP = {
    0: 'apparel',
    1: 'beauty_and_healthcare',
    2: 'electronics_and_tools',
    3: 'furnishings',
    4: 'jewelry'
}


class ImageClassifierModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        model_path = os.path.join(APP_DIRECTORY, MODELS_DIRECTORY, f'{self.model_name}.onnx')
        return onnxruntime.InferenceSession(model_path)

    def predict(self, input_image) -> str:
        preprocessed_image = self._preprocess(input_image)
        preprocessed_image_array = preprocessed_image
        input_data = preprocessed_image_array.unsqueeze(0).numpy()

        outputs = self.model.run(None, {"input.1": input_data})

        predicted_class_index = np.argmax(outputs[0]).item()

        return LABEL_ENCODER_MAP[predicted_class_index]
    

    @staticmethod
    def _preprocess(image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return preprocess(image)