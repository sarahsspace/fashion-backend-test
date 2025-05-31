from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

def extract_features(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array)

@app.post("/match")
async def match_images(
    pinterest_image: UploadFile = File(...),
    wardrobe_images: List[UploadFile] = File(...),
    occasion: str = Form(...)
):
    try:
        pinterest_feat = extract_features(await pinterest_image.read())
        matches = []

        for w_img in wardrobe_images:
            feat = extract_features(await w_img.read())
            sim = cosine_similarity(pinterest_feat, feat)[0][0]
            matches.append({
                "filename": w_img.filename,
                "similarity": float(sim)
            })

        return JSONResponse(content={
            "occasion": occasion,
            "matches": matches
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
