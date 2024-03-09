from typing import Annotated
from cv2 import imdecode, IMREAD_COLOR, resize
from fastapi import Body, FastAPI, UploadFile
import uvicorn
from load_models import load
from io import BytesIO
import numpy as np

models = load()

IMG_SIZE = 256
app = FastAPI()


@app.get("/")
async def root():
    return "SwiftScan server is running..."


@app.post("/predict")
async def predict_disease(id: Annotated[int, Body()], img_binary: UploadFile):
    stream = BytesIO(await img_binary.read())

    img = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = imdecode(img, IMREAD_COLOR)

    img = resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    pred = models[id]["model"].predict(img)
    return models[id]["classes"][np.argmax(pred, axis=1)[0]]


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", reload=True)
