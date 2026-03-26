from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import io

app = FastAPI()

MODEL_NAME = "google/siglip2-base-patch16-224"
device = "cpu"

processor = None
model = None


def load_model():
    global processor, model

    if processor is None or model is None:
        print("Loading model...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
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
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                features = outputs.image_embeds
            else:
                features = model.get_image_features(**inputs)

            features = features / features.norm(dim=-1, keepdim=True)

        vector = features[0].cpu().tolist()

        return {
            "embedding": vector,
            "dim": len(vector)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
