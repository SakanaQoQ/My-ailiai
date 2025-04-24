# qwen_vl.py
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# 自動選擇設備
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 加載處理器和模型（首次會自動下載，需科學上網）
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()

def qwen_vl_infer(text, image_path=None, max_new_tokens=256):
    """
    text: str，輸入文本
    image_path: str or None，圖片路徑（可選）
    return: str，模型生成結果
    """
    images = [Image.open(image_path)] if image_path else None
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result