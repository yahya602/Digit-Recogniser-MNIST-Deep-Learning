import io
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

app = FastAPI(title="MNIST Digit Recognition API")

# Trained Model Load Karein
MODEL = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Standard MNIST Preprocessing without distorting crops."""
    # 1. Image bytes ko open karke Grayscale ('L') mode mein badlein
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 2. Auto Invert Check: Agar background White hai to Black banayein
    img_array_temp = np.array(image)
    corner_avg = (
        img_array_temp[0, 0]
        + img_array_temp[0, -1]
        + img_array_temp[-1, 0]
        + img_array_temp[-1, -1]
    ) / 4.0

    if corner_avg > 127:
        image = ImageOps.invert(image)

    # 3. Direct Smooth Resize to 28x28 (Anti-aliasing / Bilinear filters ke sath)
    # Isse line thickness aur loops distort nahi hoti
    image = image.resize((28, 28), Image.Resampling.BILINEAR)

    # 4. Normalize pixels (0.0 to 1.0)
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 5. Flatten to (1, 784) according to ANN Input requirement
    if len(MODEL.input_shape) == 2 and MODEL.input_shape[1] == 784:
        return img_array.reshape(1, 784)
    else:
        return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"status": "API is Live"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)

        predictions = MODEL.predict(processed_img)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        return {
            "filename": file.filename,
            "predicted_digit": predicted_class,
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))