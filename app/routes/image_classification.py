import io
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from app.models.lifestyle_product_classifier_model import ImageClassifierModel


router = APIRouter()

model = ImageClassifierModel('lifestyle_classifier')


class ImageClassifierResponse(BaseModel):
    isLifestyle: bool = Field(..., alias='isLifestyle')


@router.post('/predict')
async def predict(background_tasks: BackgroundTasks, image_file: UploadFile = File(...)):
    image_bytes = await image_file.read()
    pillow_image = parse_image(image_bytes)

    image_category_prediction = model.predict(pillow_image)

    background_tasks.add_task(image_file.file.close)

    return ImageClassifierResponse(isLifestyle=image_category_prediction)


def parse_image(image_bytes: bytes):
    pillow_image = Image.open(io.BytesIO(image_bytes))

    if (pillow_image.format is not None) and (pillow_image.format.lower() == 'png'):
        pillow_image = pillow_image.convert('RGBA')

    if pillow_image.mode == 'CMYK':
        pillow_image = pillow_image.convert('RGB')

    if pillow_image.mode in ('L', 'I;16'):
        pillow_image = pillow_image.convert('RGB')

    # Save the processed image to a BytesIO object
    output_buffer = io.BytesIO()
    pillow_image.save(output_buffer, format='JPEG')
    image_bytes_output = output_buffer.getvalue()

    return image_bytes_output