import os
import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


router = APIRouter()

base_dir = os.path.dirname(__file__)
upload_dir = os.path.join(base_dir, "uploads")
os.makedirs(upload_dir, exist_ok=True)

MODEL_PATH = os.path.join(base_dir, "..", "trained_model_v3.h5")
model = InceptionV3(weights=None)
model = tf.keras.models.load_model(MODEL_PATH)

classes = ['happy', 'neutral', 'sad']


def preprocess_image(image_path, target_size=(299, 299)):
    """
    Przetwarza obraz wejściowy, aby był gotowy do predykcji przez model.

    Args:
        image_path (str): Ścieżka do obrazu.
        target_size (tuple): Rozmiar obrazu wymagany przez model (np. 299x299 dla InceptionV3).

    Returns:
        np.ndarray: Tensor obrazu gotowy do predykcji.
    """
    try:
        # Załaduj obraz i zmień jego rozmiar, konwertując do RGB
        image = load_img(image_path, target_size=target_size, color_mode='rgb')

        # Konwersja obrazu na tablicę NumPy
        image_array = img_to_array(image)

        # Przeskalowanie pikseli do zakresu [0, 1]
        image_array = image_array / 255.0

        # Dodaj dodatkowy wymiar (batch dimension)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Plik nie jest w formacie JPG lub PNG."}, status_code=400)

    try:
        # Zapisz przesłany plik na dysku
        image_path = os.path.join(upload_dir, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Przetwarzanie obrazu
        image_array = preprocess_image(image_path)

        if image_array is None:
            return JSONResponse(content={"error": "Nie udało się przetworzyć obrazu."}, status_code=400)
        
        # Przewidywanie emocji za pomocą modelu
        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction))  # Rzutowanie na int
        emotion = classes[predicted_class]
        
        return {"predicted_class": predicted_class, "emotion": emotion}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
