import os
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def model_fn(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}

def input_fn(request_body, request_content_type="application/json"):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        inputs = data.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        return inputs
    raise ValueError(f"Unsupported content type: {request_content_type}. Expected application/json")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    encoded = tokenizer(
        input_data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    id2label = {0: "negative", 1: "positive"}
    results = []
    probs_cpu = probabilities.cpu().tolist()

    for idx, probs in enumerate(probs_cpu):
        predicted_class_id = int(torch.argmax(probabilities[idx]).item())
        results.append({
            "text": input_data[idx],
            "label": id2label.get(predicted_class_id, "unknown"),
            "score": probs[predicted_class_id],
            "probabilities": {
                "negative": probs[0],
                "positive": probs[1]
            }
        })

    return results

def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
