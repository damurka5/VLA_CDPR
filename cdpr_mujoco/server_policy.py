from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, torch, numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import base64, io

MODEL_ID = "openvla/openvla-7b"  # or openvla-v01-7b
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# Torch 2.x env → you can use bfloat16 or float16; bfloat16 preferred on A100/H100
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(DEVICE)

class ActReq(BaseModel):
    instruction: str
    image_b64: str  # base64-encoded PNG/JPEG bytes

def decode_img(b64):
    arr = base64.b64decode(b64)
    im = Image.open(io.BytesIO(arr)).convert("RGB")
    return im

@app.post("/act")
def act(req: ActReq):
    img = decode_img(req.image_b64)
    prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode():
        a = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    elif isinstance(a, (list, tuple, np.ndarray)):
        a = np.asarray(a, dtype=np.float32)
    else:
        a = np.array([0,0,1,0,0,0,1], dtype=np.float32)
    # Return first 3 for now; extend to 7D once you attach gripper+yaw fully
    out = a[:3].tolist()
    return {"action": out}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7071)
