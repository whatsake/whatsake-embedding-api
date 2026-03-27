from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
import io
import os
import threading
from collections import defaultdict

app = FastAPI()

MODEL_NAME = "google/siglip2-base-patch16-224"
DEVICE = "cpu"

processor = None
model = None
qdrant = None
model_lock = threading.Lock()


def load_model():
    global processor, model

    if processor is None or model is None:
        with model_lock:
            if processor is None or model is None:
                print("Loading model...")
                hf_token = os.getenv("HF_TOKEN")

                processor = AutoProcessor.from_pretrained(
                    MODEL_NAME,
                    token=hf_token,
                    use_fast=False
                )
                model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
                model.to(DEVICE)
                model.eval()
                print("Model loaded ✅")


def get_qdrant():
    global qdrant

    if qdrant is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL or QDRANT_API_KEY is missing")

        qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            check_compatibility=False,
        )

    return qdrant


@torch.no_grad()
def embed_image(image: Image.Image):
    load_model()

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    features = pooled / pooled.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def aggregate_results(results, final_limit=5):
    grouped = defaultdict(list)

    for r in results:
        payload = r.payload or {}
        sake_id = payload.get("sake_id")
        gcs_path = payload.get("gcs_path")
        score = r.score

        if not sake_id:
            continue

        grouped[sake_id].append({
            "score": score,
            "gcs_path": gcs_path
        })

    ranked = []

    for sake_id, matches in grouped.items():
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)

        scores = [m["score"] for m in matches]
        best_score = scores[0]
        top3_scores = scores[:3]
        mean_top3_score = sum(top3_scores) / len(top3_scores)
        count_in_results = len(scores)

        final_score = (
            best_score * 0.7 +
            mean_top3_score * 0.3 +
            min(count_in_results, 3) * 0.01
        )

        ranked.append({
            "sake_id": sake_id,
            "match_score": final_score,
            "best_score": best_score,
            "mean_top3_score": mean_top3_score,
            "count_in_results": count_in_results,
            "best_match_image": matches[0]["gcs_path"]
        })

    ranked = sorted(ranked, key=lambda x: x["match_score"], reverse=True)
    return ranked[:final_limit], grouped


def search_until_enough_unique(vector, target_unique=5, initial_limit=20, max_limit=200, multiplier=2):
    client = get_qdrant()
    collection_name = os.getenv("QDRANT_COLLECTION", "sake-label-vectors")

    limit = initial_limit
    last_results = None
    last_ranked = None
    last_unique_count = 0

    while limit <= max_limit:
        print(f"Searching Qdrant with limit={limit}...")

        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True
        )

        results = response.points
        ranked, grouped = aggregate_results(results, final_limit=target_unique)
        unique_count = len(grouped)

        print(f"Found {unique_count} unique sake_ids in top {limit} raw matches.")

        last_results = results
        last_ranked = ranked
        last_unique_count = unique_count

        if unique_count >= target_unique:
            return results, ranked, limit, unique_count

        limit *= multiplier

    return last_results, last_ranked, limit // multiplier, last_unique_count


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = embed_image(image)

        raw_results, final_results, used_limit, unique_count = search_until_enough_unique(
            vector=vector,
            target_unique=5,
            initial_limit=20,
            max_limit=200,
            multiplier=2
        )

        raw_matches = []
        for r in raw_results:
            payload = r.payload or {}
            raw_matches.append({
                "score": r.score,
                "sake_id": payload.get("sake_id"),
                "gcs_path": payload.get("gcs_path")
            })

        return {
            "success": True,
            "embedding_dim": len(vector),
            "search_limit_used": used_limit,
            "unique_sake_count": unique_count,
            "top_5_unique_sakes": final_results,
            "raw_matches": raw_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
