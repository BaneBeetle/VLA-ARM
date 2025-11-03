import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# ── 1️⃣  Verify GPU ────────────────────────────────
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  Running on CPU (much slower)")

# ── 2️⃣  Load the processor + model ────────────────
print("\nLoading OpenVLA model...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,          # use GPU bfloat16 if available
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ Model loaded!")

# ── 3️⃣  Grab a test image ─────────────────────────

image_path = r"C:\Users\lolly\OneDrive\Desktop\Projects\VLA-ARM\images\test_scene.jpg"
image = Image.open(image_path).convert("RGB")

prompt = "In: Pick up the red cube. Move in small increments.\nOut:"

# ── 4️⃣  Run a forward pass ────────────────────────
inputs = processor(prompt, image).to("cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print("raw action shape:", getattr(action, "shape", None))
print("raw action (first 8):", action[:8])
