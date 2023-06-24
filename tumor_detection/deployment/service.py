from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values
from tqdm import tqdm
from PIL import Image
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)

import numpy as np
import json
import cv2
import shutil
import glob
import sys
import os

BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

IMG_RESOLUTION = (256, 256)
WIDTH = 256
HEIGHT = 265

INIT_LR = 3e-4
DECAY = 1e-6
BS = 32

MODEL_PATH = BASE_DIR + "/models/"
MODEL_NAME = "vgg16_model"

CLASSES = [
    "No Existe Tumor",
    "Tumor Detectado",
    "No Coincide",
]

IMG = "test_image.tif"


def read_image(image):
    file_location = f"images/{image.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(image.file, file_object)
    return str(file_location)

#def read_image(image):
#    os.makedirs('images', exist_ok=True)
#    file_location = f"images/{image.filename}"
#    with open(file_location, "wb+") as file_object:
#        shutil.copyfileobj(image.file, file_object)
#    return str(file_location)


def image_contrast(image: Image.Image):
    test_image = cv2.imread(IMG)
    contrast_image = cv2.imread(image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, IMG_RESOLUTION)
    contrast_image = cv2.resize(contrast_image, IMG_RESOLUTION)
    return ssim(test_image, contrast_image)

def pre_process(image: Image.Image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = cv2.resize(image, (IMG_RESOLUTION[0], IMG_RESOLUTION[1]))  # Note the shape here is (width, height)
    image = np.expand_dims(image, axis=0)  # This will reshape your image to (1, 256, 256, 3), if you want to feed one image at a time
    return image

def loading_model():
    model = load_model(MODEL_PATH + MODEL_NAME + ".h5")
    return model

def predicted_class(prediction, s: float):
    if s >= 0.1:
        model_prediction = CLASSES[prediction[0]]
    if s < 0.1:
        model_prediction = CLASSES[prediction]
    return model_prediction

def predicted_dict(prediction, model_prediction, percent_pred_list):
    if prediction == 0:
        r = {
            "label": model_prediction,
            "score": "{0:.2f}".format(percent_pred_list[0, 0] * 100),
        }
    elif prediction == 1:
        r = {
            "label": model_prediction,
            "score": "{0:.2f}".format(percent_pred_list[0, 1] * 100),
        }
    else:
        r = {"label": model_prediction}
    return r

credentials = dotenv_values(".env")

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prediction/deployment/")
async def predict_image(imagen: UploadFile = File(...)):

    data = {"success": False}
    # Read image
    image = read_image(imagen)

    # Image filter 
    s = image_contrast(image)

    data["predictions"] = [] 
    if s >= 0.1:
        # Data preprocessing
        image_file = pre_process(image)
        # Loading model
        opt = Adam(lr=INIT_LR, decay=DECAY)
        model = loading_model()
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Making prediction
        prediction = model.predict(image_file, batch_size=BS)
        # Storing prediction percentages array
        percent_pred_list = prediction
        # Transforming percentages into numeric labels
        prediction = np.argmax(prediction, axis=1)
    elif s < 0.1:
        # For not classified images
        prediction = 2
        percent_pred_list = 0

    # Obtaining prediction class
    model_prediction = predicted_class(prediction, s)
    
    # Generating a dictionary
    r = predicted_dict(prediction, model_prediction, percent_pred_list)
    data["predictions"].append(r)
    data["success"] = True

    # Generating a json encoder for the platform
    json_data = jsonable_encoder(data)

    return json_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

