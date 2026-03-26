from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import io

app = FastAPI()

MODEL_NAME = "google/siglip2-base-patch16-224"
device = "cpu"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("Model loaded ✅")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        vector = features.cpu().tolist()[0]

        return {
            "embedding": vector,
            "dim": len(vector)
        }

    except Exception as e:
        return {"error": str(e)}