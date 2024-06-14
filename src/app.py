import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from PIL import Image
from src.selfie import SelfieProcessor, ProcessingType, BackgroundColor
from src.config import get_settings

SETTTINS = get_settings()

app = FastAPI(
    title=SETTTINS.api_name,
    version=SETTTINS.revision,
)

def get_selfie_processor():
    print("loading model...")
    return SelfieProcessor()

def get_image_array(file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    # convertir a una imagen de Pillow
    img_obj = Image.open(img_stream)
    # crear array de numpy
    img_array = np.array(img_obj)

    return img_array

@app.post("/selfies")
def process_selfie(
    task: ProcessingType,
    bg_red: int | None = None,
    bg_green: int | None = None,
    bg_blue: int | None = None,
    threshold: float = 0.5,
    file: UploadFile = File(...),
    processor: SelfieProcessor = Depends(get_selfie_processor)
) -> Response:
    image_array = get_image_array(file)
    match task:
        case ProcessingType.remove_background:
            processed_image = processor.remove_background(image_array, (bg_red, bg_green, bg_blue))
        case ProcessingType.blur_background:
            processed_image = processor.blur_background(image_array)

    img_pil = Image.fromarray(processed_image)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)

    return Response(content=image_stream.read(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)