import io
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from app.models.image_classifier_model import ImageClassifierModel


router = APIRouter()

model = ImageClassifierModel('multiclass_mobile')


class ImageClassifierResponse(BaseModel):
    image_category: str = Field(..., alias='imageCategory')


@router.post('/predict')
async def predict(background_tasks: BackgroundTasks, image_file: UploadFile = File(...)):
    image_bytes = await image_file.read()
    pillow_image = parse_image(image_bytes)

    image_category_prediction = model.predict(pillow_image)

    background_tasks.add_task(image_file.file.close)

    return ImageClassifierResponse(imageCategory=image_category_prediction)


def parse_image(image_bytes: bytes):
    pillow_image = Image.open(io.BytesIO(image_bytes))

    if (pillow_image.format is not None) and (pillow_image.format.lower() == 'png'):
        pillow_image = pillow_image.convert('RGBA')

    if pillow_image.mode == 'CMYK':
        pillow_image = pillow_image.convert('RGB')

    if pillow_image.mode in ('L', 'I;16'):
        pillow_image = pillow_image.convert('RGB')

    return pillow_image