# ============================================================
#  pc_control.py  (Windows, openvla env)
#  OpenVLA control with voice input
# ============================================================

import os, socket, struct, time, sys, importlib.util
import cv2, torch, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import speech_recognition as sr 

# 🧩 environment prep
os.environ["TRANSFORMERS_NO_TF"] = "1"

def prewarm_openvla_imports():
    """Force-load NumPy before Hugging Face dynamically imports OpenVLA modules."""
    try:
        import transformers.dynamic_module_utils as dmu
        for name, mod in list(sys.modules.items()):
            if "transformers_modules" in name and "openvla" in name:
                importlib.reload(mod)
        print("✅ Preloaded OpenVLA dynamic modules with NumPy in scope")
    except Exception as e:
        print("Skip dynamic reload:", e)

# === CONFIG ==================================================
PI_IP, PI_PORT = "192.168.0.147", 5005
CAM_ID = 0
UNNORM_KEY = "bridge_orig"
FPS_LIMIT = 10
PROMPT = ""  # Will be updated from voice input
# =============================================================

# === CALIBRATIONS ============================================
NEUTRAL   = np.array([-90, -90, -90,  0], dtype=np.float32)
JOINT_DIR = np.array([  1,   1,   1,   1], dtype=np.float32)
RANGE_DEG = np.array([ 20,  20,  20,  20], dtype=np.float32)
LIMITS    = np.array([[-90, 90], [-90, 90], [-90, 90], [-90, 90]], dtype=np.float32)
# =============================================================

def listen_for_command():
    """Listen through mic and return recognized speech."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nSay your command...")
        audio = r.listen(source, phrase_time_limit=4)
    try:
        text = r.recognize_google(audio)
        print(f"You said: '{text}'")
        return text
    except sr.UnknownValueError:
        print("Didn't catch that. Try again.")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return ""

def action_to_absolute(action_vec):
    """Map model action [-1,1] -> absolute target angles (deg)."""
    a = np.array(action_vec, dtype=np.float32).reshape(-1)
    a4 = a[:4] if a.size >= 4 else np.pad(a, (0, max(0, 4 - a.size)))
    a4 = np.clip(a4, -1.0, 1.0)
    targets = NEUTRAL + (JOINT_DIR * a4 * RANGE_DEG)
    for i in range(4):
        targets[i] = float(np.clip(targets[i], LIMITS[i, 0], LIMITS[i, 1]))
    return targets.astype(np.float32)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("CUDA:", torch.cuda.is_available())

    prewarm_openvla_imports()

    print("Loading OpenVLA model...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    print("✅ Model loaded")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(CAM_ID)
    assert cap.isOpened(), "Camera not found!"

    last_t = 0.0
    running = True
    print("Controls: [SPACE]=pause/resume, [Q]=quit, [V]=voice input")

    try:
        global PROMPT
        PROMPT = "In: Pick up the red cube.\nOut:"  # default

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            cv2.putText(vis, f"RUNNING: {running}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if running else (0, 0, 255), 2)
            cv2.putText(vis, f"PROMPT: {PROMPT[:50]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            now = time.time()
            if running and (now - last_t) >= (1.0 / FPS_LIMIT):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = Image.fromarray(rgb)

                inputs = processor(PROMPT, rgb).to(
                    device, dtype=(torch.bfloat16 if "cuda" in device else torch.float32)
                )

                with torch.no_grad():
                    action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

                targets = action_to_absolute(action)
                payload = struct.pack("!4f", *targets)
                sock.sendto(payload, (PI_IP, PI_PORT))

                last_t = now

            cv2.imshow("OpenVLA Control", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                running = not running
            elif k == ord('v'):
                new_prompt = listen_for_command()
                if new_prompt:
                    PROMPT = f"In: {new_prompt}\nOut:"
            elif k == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("Closed.")

if __name__ == "__main__":
    main()
