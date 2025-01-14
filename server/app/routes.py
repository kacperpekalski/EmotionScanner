import os
import shutil
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


router = APIRouter()

upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(upload_dir, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model_v3.h5")
model = InceptionV3(weights=None)
model = tf.keras.models.load_model(MODEL_PATH)

classes = ['neutral', 'happy', 'sad']


def preprocess_image(image_path, target_size=(299, 299), color_mode='grayscale'):
    """
    Przetwarza obraz wejściowy, aby był gotowy do predykcji przez model.

    Args:
        image_path (str): Ścieżka do obrazu.
        target_size (tuple): Rozmiar obrazu wymagany przez model (np. 299x299 dla InceptionV3).
        color_mode (str): Tryb koloru ('grayscale' dla modelu z obrazami w odcieniach szarości).

    Returns:
        np.ndarray: Tensor obrazu gotowy do predykcji.
    """
    try:
        # Załaduj obraz i zmień jego rozmiar
        image = load_img(image_path, target_size=target_size, color_mode=color_mode)

        # Konwersja obrazu na tablicę NumPy
        image_array = img_to_array(image)

        # Dodaj dodatkowy wymiar (batch dimension)
        image_array = np.expand_dims(image_array, axis=0)

        # Jeśli obraz jest w odcieniach szarości, nie używamy preprocess_input
        image_array = image_array / 255.0  # Skalowanie pikseli do zakresu [0, 1]

        return image_array
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg"]:
        return JSONResponse(content={"error": "Plik nie jest w formacie JPG."}, status_code=400)

    try:
        image_path = os.path.join(upload_dir, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image_array = preprocess_image(image_path)

        if image_array is None:
            return JSONResponse(content={"error": "Nie udało się przetworzyć obrazu."}, status_code=400)
        
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        emotion = classes[predicted_class]
        
        return {"predicted_class": predicted_class, "emotion": emotion}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


