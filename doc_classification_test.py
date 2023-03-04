from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch
from PIL import Image
from pathlib import Path
from typing import List
import torch.nn.functional as F


def scale_bounding_box(box: List[int], width_scale: float, height_scale: float) -> List[int]:
    return [
        int(box[0] * width_scale)%1000,
        int(box[1] * height_scale)%1000,
        int(box[2] * width_scale)%1000,
        int(box[3] * height_scale)%1000
    ]

def predict_document_image(
        image_path,
        model: LayoutLMv3ForSequenceClassification,
        processor: LayoutLMv3Processor,
        ocr_result
    ):
    if isinstance(image_path,str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(image_path).convert("RGB")
    width, height = image.size
        
    width_scale = (1000/width)%1000
    height_scale = (1000/height)%1000
    
    words = []
    boxes = []
    for row in ocr_result:
        boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
        words.append(row["word"])
        

    encoding = processor(
        image,
        words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        output = model(
            input_ids=encoding["input_ids"].to("cpu"),
            attention_mask=encoding["attention_mask"].to("cpu"),
            bbox=encoding["bbox"].to("cpu"),
            pixel_values=encoding["pixel_values"].to("cpu")
        )

    predicted_class = output.logits.argmax()
    probs = F.softmax(output.logits, dim=1)
    confidence_score, predicted_label = torch.max(probs, dim=1)
    print("I", predicted_label)
    return { "class": model.config.id2label[predicted_class.item()], "score": confidence_score.item()*100}