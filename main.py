from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import io
import os

app = FastAPI()

MODEL_NAME = "google/siglip2-base-patch16-224"
device = "cpu"

processor = None
model = None


def load_model():
    global processor, model

    if processor is None or model is None:
        print("Loading model...")
        hf_token = os.getenv("HF_TOKEN")

        processor = AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
        model.to(device)
        model.eval()
        print("Model loaded ✅")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    try:
        load_model()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            features = pooled / pooled.norm(dim=-1, keepdim=True)

        vector = features[0].cpu().tolist()

        return {
            "embedding": vector,
            "dim": len(vector)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
