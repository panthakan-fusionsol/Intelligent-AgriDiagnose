import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw
from PIL import ImageOps
import requests
import json
from streamlit_cropper import st_cropper
import io
import numpy as np
from datetime import datetime
import os
import cv2
import torch.nn as nn
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw
from PIL import ImageOps
import requests
import json
import io
import numpy as np
from datetime import datetime
import os
import cv2
import torch.nn as nn
from languages import LANGUAGES
import dotenv
from gpt_prompt import GPT_PROMPTS
from openai import OpenAI
dotenv.load_dotenv(override=True);

# ------------------ Safe secret/env retrieval ------------------
def _get_config(name: str, default: str = None):
    """Return config from st.secrets[name] if available, else environment, else default.
    Avoids StreamlitSecretNotFoundError when no secrets file exists."""
    try:
        # Access may raise if no secrets file; catch broadly.
        if name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        pass
    return os.getenv(name, default)

# ------------------ Custom Vision Config ------------------
CUSTOM_VISION_ENDPOINT = "https://southeastasia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/0c96d1d3-e022-47b0-a028-177007d20bdf/detect/iterations/Iteration2/image"
CUSTOM_VISION_PREDICTION_KEY = os.getenv("CUSTOM_VISION_PREDICTION_KEY")

# ------------------ Grad-CAM Implementation ------------------
def compute_gradcam(model: nn.Module, image_tensor: torch.Tensor, layer_name: str, target_class: int = None):
    """
    Compute Grad-CAM for a single image
    image_tensor: [1,C,H,W] on device
    return: numpy heatmap shape [H', W'] normalized to [0,1]
    """
    # Find the target layer
    layer = None
    for named, module in model.named_modules():
        if named == layer_name:
            layer = module
            break
    
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    feats = None
    grads = None

    def forward_hook(_module, _input, output):
        nonlocal feats
        feats = output

    def backward_hook(_module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    # Register hooks
    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    model.eval()
    
    # Forward pass
    preds = model(image_tensor)
    
    # Use target class or predicted class
    if target_class is None:
        target_class = preds.softmax(dim=-1).argmax(dim=-1).item()
    
    # Backward pass
    model.zero_grad()
    class_score = preds[0, target_class]
    class_score.backward()

    # Remove hooks
    h1.remove()
    h2.remove()

    # Compute Grad-CAM
    weights = grads.mean(dim=(-2, -1))  # Global average pooling
    cam = (weights.unsqueeze(-1).unsqueeze(-1) * feats).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam.detach().cpu().numpy()

def overlay_gradcam(original_image: Image.Image, cropped_image: Image.Image, 
                   heatmap: np.ndarray, alpha: float = 0.4, 
                   crop_coords: dict = None) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original image
    """
    # Convert PIL to numpy
    orig_np = np.array(original_image)
    
    # Resize heatmap to match cropped region size
    if crop_coords:
        crop_h = crop_coords['height']
        crop_w = crop_coords['width']
        heatmap_resized = cv2.resize(heatmap, (crop_w, crop_h))
        
        # Create full-size heatmap initialized with zeros
        full_heatmap = np.zeros((orig_np.shape[0], orig_np.shape[1]), dtype=np.float32)
        
        # Place the heatmap in the correct position
        y1, y2 = crop_coords['top'], crop_coords['bottom']
        x1, x2 = crop_coords['left'], crop_coords['right']
        full_heatmap[y1:y2, x1:x2] = heatmap_resized
    else:
        # If no crop coords, resize to full image
        full_heatmap = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    
    # Convert heatmap to color
    heatmap_u8 = np.uint8(255 * full_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    blended = cv2.addWeighted(orig_np, 1 - alpha, heatmap_color, alpha, 0)
    
    return Image.fromarray(blended)

def default_layer_name(model_type: str) -> str:
    """Return default layer name for Grad-CAM based on model architecture"""
    if model_type.lower() == "resnet18":
        return "layer4.1.conv2"
    elif model_type.lower() == "resnet50":
        return "layer4.2.conv3"
    else:
        return "layer4.1.conv2"  # Default to ResNet18

# ------------------ Custom Vision Helper ------------------
def custom_vision_detect(image: Image.Image):
    """Send image to Custom Vision detection endpoint and return top prediction bbox in pixel coordinates."""
    if not CUSTOM_VISION_ENDPOINT or not CUSTOM_VISION_PREDICTION_KEY:
        return None, "ยังไม่ได้ตั้งค่า CUSTOM_VISION_ENDPOINT หรือ CUSTOM_VISION_PREDICTION_KEY"
    try:
        # Enforce Custom Vision 4MB limit
        MAX_BYTES = 3_900_000
        work_img = image.copy()
        
        def encode(img, quality=90):
            buf = io.BytesIO()
            img.save(buf, format='png', quality=quality, optimize=True)
            return buf.getvalue()
        
        img_bytes = encode(work_img, quality=90)
        
        # Compress if needed
        if len(img_bytes) > MAX_BYTES:
            for q in [80, 70, 60, 50, 40, 30]:
                img_bytes = encode(work_img, quality=q)
                if len(img_bytes) <= MAX_BYTES:
                    break
        
        if len(img_bytes) > MAX_BYTES:
            max_side = max(work_img.size)
            target_side = 1800
            while len(img_bytes) > MAX_BYTES and target_side >= 600:
                if max_side > target_side:
                    scale = target_side / max_side
                    new_size = (int(work_img.width * scale), int(work_img.height * scale))
                    work_img = work_img.resize(new_size, Image.LANCZOS)
                    img_bytes = encode(work_img, quality=70)
                    if len(img_bytes) > MAX_BYTES:
                        img_bytes = encode(work_img, quality=50)
                target_side -= 250
        
        if len(img_bytes) > MAX_BYTES:
            return None, f"ภาพมีขนาดใหญ่เกินไปหลังบีบอัด ({len(img_bytes)/1024:.0f}KB > {MAX_BYTES/1024:.0f}KB) ลองอัปโหลดภาพที่เล็กลง"
        
        headers = {
            "Prediction-Key": CUSTOM_VISION_PREDICTION_KEY,
            "Content-Type": "application/octet-stream"
        }
        
        resp = requests.post(CUSTOM_VISION_ENDPOINT, headers=headers, data=img_bytes, timeout=30)
        if resp.status_code != 200:
            return None, f"Custom Vision error {resp.status_code}: {resp.text[:200]}"
        
        data = resp.json()
        preds = data.get("predictions") or data.get("Predictions") or []
        if not preds:
            return None, "ไม่มีผลลัพธ์ (predictions ว่าง)"
        
        top_pred = max(preds, key=lambda p: p.get("probability", p.get("confidence", 0)))
        prob = top_pred.get("probability", top_pred.get("confidence", 0))
        tagName = top_pred.get("tagName") or top_pred.get("TagName") or "unknown"
        bbox = top_pred.get("boundingBox") or top_pred.get("BoundingBox") or {}
        
        if not bbox or all(k not in bbox for k in ("left","top","width","height")):
            return None, "ผลลัพธ์ไม่มี boundingBox (อาจใช้ endpoint ประเภท classify แทน detect)"
        
        # Convert normalized bbox to pixel coordinates
        W, H = image.size
        n_left = bbox.get('left', 0)
        n_top = bbox.get('top', 0)
        n_width = bbox.get('width', 0)
        n_height = bbox.get('height', 0)
        
        left = int(n_left * W)
        top = int(n_top * H)
        right = int((n_left + n_width) * W)
        # BUGFIX: use image height for bottom coordinate (was using W)
        bottom = int((n_top + n_height) * H)
        
        if n_width <= 0 or n_height <= 0:
            return None, f"boundingBox width/height เป็นศูนย์ (n_width={n_width}, n_height={n_height})"
        
        # Clamp coordinates
        left = max(0, left)
        top = max(0, top)
        right = min(W, right)
        bottom = min(H, bottom)
        
        if right <= left or bottom <= top:
            return None, "กรอบไม่ถูกต้องจาก Custom Vision (skip)"
        
        result = {
            "tagName": tagName,
            "probability": prob,
            "bbox": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
                "normalized": {
                    "left": n_left,
                    "top": n_top,
                    "width": n_width,
                    "height": n_height
                }
            },
            "raw": data
        }
        return result, None
    except Exception as e:
        return None, str(e)

# ------------------ Drawing Helper ------------------
def _draw_bbox_on_image(image: Image.Image, bbox: dict, color=(0, 255, 0), width: int = 4):
    """Return a copy of image with rectangle bbox drawn."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']], outline=color, width=width)
    return img_copy

def _clamp_and_sort_bbox(l, t, r, b, W, H):
    """Clamp bbox to image bounds and ensure proper ordering."""
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    l = max(0, min(l, W - 1)); r = max(0, min(r, W))
    t = max(0, min(t, H - 1)); b = max(0, min(b, H))
    return l, t, r, b

# ------------------ Multi Detection Helper ------------------
def custom_vision_detect_multi(image: Image.Image, min_prob: float = 0.5):
    """Return list of detection results (each like single detect result) filtered by min_prob (normalized 0-1)."""
    if not CUSTOM_VISION_ENDPOINT or not CUSTOM_VISION_PREDICTION_KEY:
        return [], f"ยังไม่ได้ตั้งค่า CUSTOM_VISION_ENDPOINT หรือ CUSTOM_VISION_PREDICTION_KEY"
    try:
        MAX_BYTES = 3_900_000
        work_img = image.copy()

        def encode(img, quality=90):
            buf = io.BytesIO()
            img.save(buf, format='png', quality=quality, optimize=True)
            return buf.getvalue()

        img_bytes = encode(work_img, quality=90)
        if len(img_bytes) > MAX_BYTES:
            for q in [80,70,60,50,40,30]:
                img_bytes = encode(work_img, quality=q)
                if len(img_bytes) <= MAX_BYTES:
                    break
        if len(img_bytes) > MAX_BYTES:
            max_side = max(work_img.size)
            target_side = 1800
            while len(img_bytes) > MAX_BYTES and target_side >= 600:
                if max_side > target_side:
                    scale = target_side / max_side
                    new_size = (int(work_img.width * scale), int(work_img.height * scale))
                    work_img = work_img.resize(new_size, Image.LANCZOS)
                    img_bytes = encode(work_img, quality=70)
                    if len(img_bytes) > MAX_BYTES:
                        img_bytes = encode(work_img, quality=50)
                target_side -= 250
        if len(img_bytes) > MAX_BYTES:
            return [], f"ภาพมีขนาดใหญ่เกินไปหลังบีบอัด ({len(img_bytes)/1024:.0f}KB)"

        headers = {"Prediction-Key": CUSTOM_VISION_PREDICTION_KEY, "Content-Type": "application/octet-stream"}
        resp = requests.post(CUSTOM_VISION_ENDPOINT, headers=headers, data=img_bytes, timeout=30)
        if resp.status_code != 200:
            return [], f"Custom Vision error {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        preds = data.get("predictions") or data.get("Predictions") or []
        if not preds:
            return [], "ไม่มีผลลัพธ์ (predictions ว่าง)"
        W, H = image.size
        results = []
        for p in preds:
            prob = p.get("probability", p.get("confidence", 0.0))
            if prob < min_prob:
                continue
            bbox = p.get("boundingBox") or p.get("BoundingBox") or {}
            if not bbox:
                continue
            n_left = bbox.get('left', 0)
            n_top = bbox.get('top', 0)
            n_width = bbox.get('width', 0)
            n_height = bbox.get('height', 0)
            if n_width <= 0 or n_height <= 0:
                continue
            left = int(n_left * W)
            top = int(n_top * H)
            right = int((n_left + n_width) * W)
            bottom = int((n_top + n_height) * H)
            left = max(0, left); top = max(0, top)
            right = min(W, right); bottom = min(H, bottom)
            if right <= left or bottom <= top:
                continue
            results.append({
                "tagName": p.get("tagName") or p.get("TagName") or "unknown",
                "probability": prob,
                "bbox": {
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "width": right-left, "height": bottom-top,
                    "normalized": {"left": n_left, "top": n_top, "width": n_width, "height": n_height}
                },
                "raw": p
            })
        # Sort desc by probability
        results.sort(key=lambda r: r['probability'], reverse=True)
        return results, None
    except Exception as e:
        return [], str(e)

# ------------------ Load Model ------------------
# @st.cache_resource
# def load_model(model_path, device):
#     checkpoint = torch.load(model_path, map_location=device)
#     num_classes = len(checkpoint['class2id'])
#     class_names = list(checkpoint['class2id'].keys())
    
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
#     return model, class_names

# Add this class mapping after loading the model
@st.cache_resource
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(checkpoint['class2id'])
    
    # Original class names from checkpoint
    original_class_names = list(checkpoint['class2id'].keys())
    
    # Create mapping from original to new names
    class_name_mapping = {
        'abnormality': "Herbicide Injury ",
        'blight': "Northern Corn Leaf Blight",
        'brownspot': "Brown Spot", 
        'curl': "Twisted Whorl",
        'healthy': "Healthy",
        'mildew': "Downy Mildew",
        'rust': "Rust",
        'smut': "Smut",
        'spot': "Leaf Sheath and Leaf Spot",
        'virus': "Virus"
    }
    
    # Map to new class names while preserving order
    class_names = []
    for original_name in original_class_names:
        new_name = class_name_mapping.get(original_name, original_name)
        class_names.append(new_name)
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, class_names

# ------------------ Transform ------------------
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ------------------ Prediction ------------------
def predict_image(model, image, transform, class_names, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
    return predicted_class, confidence_score

def predict_with_gradcam(model, image, transform, class_names, device, layer_name="layer4.1.conv2"):
    """Predict and compute Grad-CAM simultaneously"""
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        predicted_idx = predicted.item()
    
    # Grad-CAM
    heatmap = compute_gradcam(model, image_tensor, layer_name, target_class=predicted_idx)
    
    return predicted_class, confidence_score, heatmap

openai_client = OpenAI(
    base_url="https://chat-gpt-corn.openai.azure.com/openai/v1/",
    api_key=os.getenv('chat-gpt-api-key')
)
deployment_name = "gpt-4.1-mini"
# ------------------ GPT Call ------------------
def call_gpt(predicted_class, confidence, selected_language="ไทย"):
    # Get prompts for the selected language, fallback to Thai if not found
    prompts = GPT_PROMPTS.get(selected_language, GPT_PROMPTS["ไทย"])
    
    # Format the user prompt with the predicted class and confidence
    user_prompt = prompts["user_template"].format(
        predicted_class=predicted_class,
        confidence=confidence
    )
    
    if selected_language == "ไทย":
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompts["system"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    else:
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": user_prompt}
        ]
    
    try:
        completion = openai_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.0,
            max_tokens=300
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error calling GPT: {str(e)}"

# ------------------ Streamlit UI ------------------
# Add mobile-friendly CSS
st.markdown("""
<style>
    body {
        overflow-x: hidden !important;
    }

    div[data-testid="stSelectbox"] {
        max-width: 220px;
        margin-left: auto;
    }

    div[data-testid="stSelectbox"] > div {
        width: 100% !important;
    }

    .stButton > button {
        min-height: 44px;
        font-size: 16px;
    }

    div[data-testid="stNumberInput"] input {
        max-width: 80px;
    }

    canvas {
        touch-action: manipulation !important;
        -webkit-touch-callout: none;
        -webkit-user-select: none;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
    }

    .stCanvas canvas {
        max-width: 100% !important;
        height: auto !important;
        touch-action: manipulation !important;
    }

    div[data-testid="stCropper"] {
        width: 100%;
        max-width: 720px;
        margin: 0 auto;
    }

    div[data-testid="stCropper"] > div {
        width: 100% !important;
        max-width: 100vw !important;
    }

    div[data-testid="stCropper"] > div > div {
        width: 100% !important;
        max-width: 100% !important;
    }

    div[data-testid="stCropper"] canvas {
        display: block;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
    }

    .stImage {
        display: flex;
        justify-content: center;
    }

    .stImage > img,
    .stImage > div img {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }

    @media (max-width: 768px) {
        div[data-testid="block-container"] {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.5rem !important;
        }

        section.main > div:first-child {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .stCanvas > div {
            max-width: 100% !important;
            overflow-x: auto;
        }

        div[data-testid="stSelectbox"] {
            max-width: 160px !important;
            width: 100% !important;
            margin-right: 0 !important;
        }

        div[data-testid="stCropper"] {
            max-width: 100% !important;
        }

        div[data-testid="stCropper"] > div {
            width: 100% !important;
        }

        div[data-testid="column"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }

        .stSlider {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .stButton > button {
            width: 100%;
        }

        .stCanvas {
            overflow: visible !important;
        }

        .stCanvas .toolbar {
            background: rgba(255,255,255,0.9) !important;
            border: 1px solid #ccc !important;
            border-radius: 5px !important;
        }

        .element-container {
            touch-action: manipulation !important;
        }

        .stImage > img, .stImage > div img {
            width: 100% !important;
            height: auto !important;
        }
    }

    @media (max-width: 480px) {
        h1, .stMarkdown h1 {
            font-size: 1.6rem !important;
        }

        h2, .stMarkdown h2 {
            font-size: 1.3rem !important;
        }

        div[data-testid="stSelectbox"] {
            max-width: 140px !important;
        }

        .stButton > button {
            font-size: 15px;
            padding: 0.6rem 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Initialize language selection
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

# Language dropdown in top right
col_title, col_lang = st.columns([4, 1])
with col_title:
    pass
with col_lang:
    language = st.selectbox(
        "Languages",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
        key="language_selector"
    )
    if language != st.session_state.selected_language:
        st.session_state.selected_language = language
        # Clear all GPT response states when changing language
        keys_to_clear = []
        for key in st.session_state.keys():
            if key.startswith('gpt_response_leaf_') or key == 'gpt_response_manual' or key == 'manual_gpt_response':
                keys_to_clear.append(key)
        for key in keys_to_clear:
            del st.session_state[key]
        # Reset GPT-related flags
        if 'trigger_gpt' in st.session_state:
            st.session_state.trigger_gpt = False
        if 'gpt_called_for_current_result' in st.session_state:
            st.session_state.gpt_called_for_current_result = False
        if 'is_analyzing' in st.session_state:
            st.session_state.is_analyzing = False
        # No st.rerun() here - let natural flow handle the language change

# Get current language texts
lang = LANGUAGES[st.session_state.selected_language]

st.title(lang["title"])
st.write(lang["subtitle"])

# Function to load image from URL
@st.cache_data
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image, None
    except requests.exceptions.RequestException as e:
        return None, f"เกิดข้อผิดพลาดในการดาวน์โหลดภาพ: {str(e)}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}"

# Initialize session state
if 'url_input' not in st.session_state:
    st.session_state.url_input = ""
if 'clear_uploader' not in st.session_state:
    st.session_state.clear_uploader = False
if 'show_gradcam' not in st.session_state:
    # Default OFF until user toggles
    st.session_state.show_gradcam = False
# Initialize slider values to preserve on language change
if 'custom_conf_value' not in st.session_state:
    st.session_state.custom_conf_value = 95.00
if 'model_conf_value' not in st.session_state:
    st.session_state.model_conf_value = 97.5
if 'custom_conf_slider' not in st.session_state:
    st.session_state.custom_conf_slider = float(st.session_state.custom_conf_value)
if 'custom_conf_number' not in st.session_state:
    st.session_state.custom_conf_number = float(st.session_state.custom_conf_value)
if 'model_conf_slider' not in st.session_state:
    st.session_state.model_conf_slider = float(st.session_state.model_conf_value)
if 'model_conf_number' not in st.session_state:
    st.session_state.model_conf_number = float(st.session_state.model_conf_value)
# Initialize mode to preserve on language change
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = 'auto'
# Initialize persistent image data to preserve across language changes
if 'persistent_image_data' not in st.session_state:
    st.session_state.persistent_image_data = None
if 'persistent_file_name' not in st.session_state:
    st.session_state.persistent_file_name = None
if 'persistent_image_source' not in st.session_state:
    st.session_state.persistent_image_source = None

# Initialize manual crop state defaults
if 'last_cropped_result' not in st.session_state:
    st.session_state.last_cropped_result = None
if 'last_rect_coords' not in st.session_state:
    st.session_state.last_rect_coords = None
if 'crop_done' not in st.session_state:
    st.session_state.crop_done = False

# Initialize variables
uploaded_file = None
image = None
current_file_name = None
image_source = None

# Check for URL parameter first
query_params = st.query_params
param_url = query_params.get("images", None)

# Create input sections
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader(lang["select_input"])
    st.markdown(lang["input_note"])

def _update_custom_from_slider():
    val = float(st.session_state.custom_conf_slider)
    val = min(99.0, max(50.0, val))
    st.session_state.custom_conf_value = val
    st.session_state.custom_conf_slider = val
    st.session_state.custom_conf_number = val

def _update_custom_from_number():
    val = float(st.session_state.custom_conf_number)
    val = min(99.0, max(50.0, val))
    st.session_state.custom_conf_value = val
    st.session_state.custom_conf_slider = val
    st.session_state.custom_conf_number = val

def _update_model_from_slider():
    val = float(st.session_state.model_conf_slider)
    val = min(99.0, max(0.0, val))
    st.session_state.model_conf_value = val
    st.session_state.model_conf_slider = val
    st.session_state.model_conf_number = val

def _update_model_from_number():
    val = float(st.session_state.model_conf_number)
    val = min(99.0, max(0.0, val))
    st.session_state.model_conf_value = val
    st.session_state.model_conf_slider = val
    st.session_state.model_conf_number = val

with col2:
    leaf_label_col, leaf_input_col = st.columns([0.5, 0.3])
    with leaf_label_col:
        st.markdown(f"**{lang['leaf_confidence']}**")
    with leaf_input_col:
        st.number_input(
            lang["leaf_confidence"],
            min_value=50.0,
            max_value=99.0,
            step=0.1,
            format="%.1f",
            key="custom_conf_number",
            on_change=_update_custom_from_number,
            label_visibility="collapsed"
        )
    st.slider(
        lang["leaf_confidence"],
        50.00,
        99.00,
        step=0.1,
        key="custom_conf_slider",
        on_change=_update_custom_from_slider,
        label_visibility="collapsed"
    )

    disease_label_col, disease_input_col = st.columns([0.5, 0.3])
    with disease_label_col:
        st.markdown(f"**{lang['disease_confidence']}**")
    with disease_input_col:
        st.number_input(
            lang["disease_confidence"],
            min_value=0.0,
            max_value=99.0,
            step=0.1,
            format="%.1f",
            key="model_conf_number",
            on_change=_update_model_from_number,
            label_visibility="collapsed"
        )
    st.slider(
        lang["disease_confidence"],
        0.00,
        99.00,
        step=0.1,
        key="model_conf_slider",
        on_change=_update_model_from_slider,
        label_visibility="collapsed"
    )
    custom_conf = float(st.session_state.custom_conf_value)
    model_conf = float(st.session_state.model_conf_value)

# Track slider changes to trigger re-detect without losing current image
if 'prev_custom_conf' not in st.session_state:
    st.session_state.prev_custom_conf = custom_conf
if 'prev_model_conf' not in st.session_state:
    st.session_state.prev_model_conf = model_conf
custom_conf_changed = custom_conf != st.session_state.prev_custom_conf
model_conf_changed = model_conf != st.session_state.prev_model_conf

# Clear GPT responses when slider values change
if custom_conf_changed or model_conf_changed:
    keys_to_clear = []
    for key in st.session_state.keys():
        if key.startswith('gpt_response_leaf_') or key == 'gpt_response_manual' or key == 'manual_gpt_response':
            keys_to_clear.append(key)
    for key in keys_to_clear:
        del st.session_state[key]
    # Reset GPT-related flags
    if 'gpt_called_for_current_result' in st.session_state:
        st.session_state.gpt_called_for_current_result = False

# (Grad-CAM toggle moved to bottom mode button sections)
st.session_state.prev_custom_conf = custom_conf
st.session_state.prev_model_conf = model_conf


# Create columns for better layout
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(f"**{lang['url_method']}**")
    
    url_key = "url_input_field"
    image_url = st.text_input(
        lang["url_label"],
        value=st.session_state.url_input,
        placeholder=lang["url_placeholder"],
        help=lang["url_help"],
        key=url_key
    )
    
    if image_url != st.session_state.url_input:
        st.session_state.url_input = image_url
        if image_url.strip():
            st.session_state.clear_uploader = True
            # Clear persistent data when user enters new URL
            st.session_state.persistent_image_data = None
            st.session_state.persistent_file_name = None
            st.session_state.persistent_image_source = None

with col2:
    st.markdown(f"**{lang['upload_method']}**")

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if st.session_state.clear_uploader:
        st.session_state.uploader_key += 1
        st.session_state.clear_uploader = False

    uploader_key = f"file_uploader_{st.session_state.uploader_key}"
    uploaded_file = st.file_uploader(
        lang["select_image"], 
        type=["jpg", "jpeg", "png"],
        help=lang["upload_help"],
        key=uploader_key
    )
    
    if uploaded_file is not None:
        if st.session_state.url_input:
            st.session_state.url_input = ""
        # Clear persistent data when user uploads new file
        st.session_state.persistent_image_data = None
        st.session_state.persistent_file_name = None
        st.session_state.persistent_image_source = None

# Handle URL parameter
if param_url and not st.session_state.url_input and uploaded_file is None:
    st.session_state.url_input = param_url

# Process the active input source
active_source = None

# Priority 1: Check if there's a valid URL input
if st.session_state.url_input and st.session_state.url_input.strip():
    image_url = st.session_state.url_input.strip()
    with st.spinner(lang["loading_url"]):
        image, error = load_image_from_url(image_url)
    
    if error:
        st.error(lang["url_load_failed"])
        st.caption(f"Error: {error}")
        active_source = None
    elif image is not None:
        st.success(lang["url_load_success"])
        current_file_name = f"url_{hash(image_url)}"
        image_source = "url"
        active_source = "url"
        
        # Store persistent image data
        st.session_state.persistent_image_data = image
        st.session_state.persistent_file_name = current_file_name
        st.session_state.persistent_image_source = image_source
        
        if param_url:
            st.info(f"Loaded image from URL parameter: {image_url}")

# Priority 2: Check uploaded file
elif uploaded_file is not None:
    if uploaded_file.type == "image/png":
        img = Image.open(uploaded_file).convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img = Image.alpha_composite(background.convert("RGBA"), img)
        image = rgb_img.convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")
    
    current_file_name = uploaded_file.name
    image_source = "upload"
    active_source = "upload"
    
    # Store persistent image data
    st.session_state.persistent_image_data = image
    st.session_state.persistent_file_name = current_file_name
    st.session_state.persistent_image_source = image_source
    
    st.success(f"{lang['upload_success']} {uploaded_file.name}")

# Priority 3: Restore from persistent storage if no new input but data exists
elif (st.session_state.persistent_image_data is not None and 
      st.session_state.persistent_file_name is not None):
    image = st.session_state.persistent_image_data
    current_file_name = st.session_state.persistent_file_name
    image_source = st.session_state.persistent_image_source
    active_source = image_source
    
    # Show restore message
    if image_source == "url":
        st.success(f"{lang['upload_success']} {uploaded_file.name}")
    else:
        st.success(f"{lang['image_restored']}: {current_file_name}")

# Check for new file upload or URL change and reset everything if needed
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# Reset all states when a new file is uploaded or URL changes
if current_file_name is not None:
    if st.session_state.last_uploaded_file_name != current_file_name:
        # Reset all drawing-related states
        st.session_state.crop_done = False
        st.session_state.last_rect_coords = None
        st.session_state.last_cropped_result = None
        st.session_state.transform_mode = False
        st.session_state.is_analyzing = False
        st.session_state.last_uploaded_file_name = current_file_name
        st.session_state.gpt_called_for_current_result = False
        st.session_state.just_cleared = False
        if 'manual_gpt_response' in st.session_state:
            del st.session_state.manual_gpt_response
        if "too_many_boxes_warning" in st.session_state:
            del st.session_state.too_many_boxes_warning
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = 0
        st.session_state.canvas_key += 1
        # Clear auto-detect related states
        for k in ['auto_detect_attempted','auto_detect_success','auto_detect_bbox','auto_detect_meta','bbox']:
            if k in st.session_state:
                del st.session_state[k]
        # Clear all GPT response states when changing image
        keys_to_clear = []
        for key in st.session_state.keys():
            if key.startswith('gpt_response_leaf_') or key == 'gpt_response_manual' or key == 'manual_gpt_response':
                keys_to_clear.append(key)
        for key in keys_to_clear:
            del st.session_state[key]
        # Clear per-box Grad-CAM states
        if 'per_box_gradcam_enabled' in st.session_state:
            st.session_state.per_box_gradcam_enabled = {}
        if 'per_box_gradcam_heatmaps' in st.session_state:
            st.session_state.per_box_gradcam_heatmaps = {}
        if 'manual_cropper_key' in st.session_state:
            st.session_state.manual_cropper_key += 1
        else:
            st.session_state.manual_cropper_key = 0
        st.session_state.mode = 'auto'
        st.session_state.current_mode = 'auto'

# Handle case where there's no current image but persistent data exists
elif (current_file_name is None and 
      st.session_state.persistent_image_data is not None and 
      st.session_state.persistent_file_name is not None):
    # Restore states for display but don't trigger reset
    current_file_name = st.session_state.persistent_file_name
    image_source = st.session_state.persistent_image_source

if image is not None:
    # Load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/mnt/c/fusion/corn_project/model_training/resnet18_448_cosine_croponly_10_jitter/checkpoints/best.pth"
    model, class_names = load_model(model_path, device)
    transform = get_test_transform()
    
    # User agent for mobile detection
    user_agent = st.context.headers.get("User-Agent", "").lower() if hasattr(st, 'context') and hasattr(st.context, 'headers') else ""
    is_mobile = any(x in user_agent for x in ['mobile', 'android', 'iphone', 'ipad']) if user_agent else False
    
    # Ensure analysis flags exist
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    if 'gpt_called_for_current_result' not in st.session_state:
        st.session_state.gpt_called_for_current_result = False
    if 'trigger_gpt' not in st.session_state:
        st.session_state.trigger_gpt = False

    layer_name = default_layer_name("resnet18")

    # Lazy Grad-CAM computation now handled when user toggles at bottom buttons
    # ------------------ One-time Auto Detection & Auto Crop ------------------
    if 'auto_detect_attempted' not in st.session_state:
        st.session_state.auto_detect_attempted = False
    if 'auto_detect_success' not in st.session_state:
        st.session_state.auto_detect_success = False
    if 'show_manual_cropper' not in st.session_state:
        st.session_state.show_manual_cropper = False

    # Helper function to perform auto-detect
    def _run_auto_detect():
        detection_threshold = custom_conf / 100.0
        classification_threshold = model_conf / 100.0
        with st.spinner(lang["AUTO_MULTI_DETECT"]):
            det_list, det_err = custom_vision_detect_multi(image, min_prob=detection_threshold)
            st.session_state.auto_detect_attempted = True
            st.session_state.auto_detect_last_error = det_err
            if det_err:
                st.session_state.auto_detect_success = False
                st.warning(f"⚠️ Auto Detect ล้มเหลว: {det_err}")
                return
            if not det_list:
                st.session_state.auto_detect_success = False
                st.warning(lang["no_threshold_boxes"])
                return
            # Classify each detection once
            results = []
            for idx, det in enumerate(det_list):
                bbox = det['bbox']
                cropped = image.crop((bbox['left'], bbox['top'], bbox['right'], bbox['bottom']))
                if st.session_state.show_gradcam:
                    pred_class, conf, heatmap = predict_with_gradcam(
                        model, cropped, transform, class_names, device, layer_name
                    )
                else:
                    pred_class, conf = predict_image(model, cropped, transform, class_names, device)
                    heatmap = None
                results.append({
                    'detection_probability': det['probability'],
                    'detector_tag': det['tagName'],
                    'bbox': bbox,
                    'predicted_class': pred_class,
                    'classification_confidence': conf,
                    'image': cropped,
                    'heatmap': heatmap
                })
            # กรองผลลัพธ์: เก็บเฉพาะ 1 ใบต่อ 1 โรค (เลือกใบที่มี confidence สูงสุด)
            disease_groups = {}
            for r in results:
                disease = r['predicted_class']
                if disease not in disease_groups:
                    disease_groups[disease] = r
                else:
                    # เลือกใบที่มี classification confidence สูงกว่า
                    if r['classification_confidence'] > disease_groups[disease]['classification_confidence']:
                        disease_groups[disease] = r
            
            # แปลงกลับเป็น list และเรียงตาม confidence
            filtered_results = list(disease_groups.values())
            filtered_results.sort(key=lambda x: x['classification_confidence'], reverse=True)

            passing_results = [r for r in filtered_results if r['classification_confidence'] >= classification_threshold]

            if not passing_results:
                st.session_state.auto_detect_results = []
                st.session_state.auto_detect_success = False
                st.session_state.last_cropped_result = None
                st.session_state.last_gradcam_heatmap = None
                st.session_state.crop_done = False
                st.session_state.last_rect_coords = None
                for key in ('auto_detect_bbox', 'auto_detect_meta', 'auto_detect_chosen_index', 'bbox'):
                    if key in st.session_state:
                        del st.session_state[key]
                st.warning(lang["no_threshold_boxes"])
                return

            st.session_state.auto_detect_results = passing_results
            chosen = passing_results[0]
            st.session_state.auto_detect_bbox = chosen['bbox']
            st.session_state.auto_detect_meta = {
                'probability': chosen['detection_probability'],
                'tag': chosen['detector_tag'],
                'raw': None
            }
            st.session_state.auto_detect_success = True
            st.session_state.last_cropped_result = {
                'image': chosen['image'],
                'predicted_class': chosen['predicted_class'],
                'confidence': chosen['classification_confidence'],
                'crop_coords': {
                    'left': chosen['bbox']['left'], 'top': chosen['bbox']['top'], 'right': chosen['bbox']['right'], 'bottom': chosen['bbox']['bottom'],
                    'width': chosen['bbox']['width'], 'height': chosen['bbox']['height'], 'auto_detected': True,
                    'detector_probability': chosen['detection_probability'], 'detector_tag': chosen['detector_tag']
                }
            }
            if st.session_state.show_gradcam and chosen.get('heatmap') is not None:
                st.session_state.last_gradcam_heatmap = chosen['heatmap']
            else:
                st.session_state.last_gradcam_heatmap = None

    # Always run auto-detect once per new image
    # Re-run if first time OR detection threshold changed
    if (custom_conf_changed or model_conf_changed) and image is not None:
        # Reset only detection-related states, keep image & manual states
        for k in ['auto_detect_results','auto_detect_bbox','auto_detect_meta','auto_detect_last_error']:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.auto_detect_attempted = False
        # Optional: debug indicator
        st.session_state.last_threshold_refresh = {
            'custom_conf': custom_conf,
            'model_conf': model_conf
        }
    if st.session_state.get('mode', 'auto') == 'auto' and not st.session_state.auto_detect_attempted:
        _run_auto_detect()

    # ------------------ Mode Buttons (Top) ------------------
    st.subheader(lang["mode_select"])
    if 'mode' not in st.session_state:
        st.session_state.mode = st.session_state.current_mode
    if 'crop_area' not in st.session_state:
        st.session_state.crop_area = None
    if 'bbox' not in st.session_state and st.session_state.get('auto_detect_bbox'):
        st.session_state.bbox = st.session_state.auto_detect_bbox
    if 'manual_cropper_key' not in st.session_state:
        st.session_state.manual_cropper_key = 0

    auto_box = st.session_state.get('bbox') or st.session_state.get('auto_detect_bbox')

    def _clear_all_gpt_responses():
        keys_to_clear = [
            key for key in st.session_state.keys()
            if key.startswith('gpt_response_leaf_') or key in ('gpt_response_manual', 'manual_gpt_response')
        ]
        for key in keys_to_clear:
            del st.session_state[key]

    def _switch_mode(new_mode: str):
        current = st.session_state.get('mode', st.session_state.get('current_mode', 'auto'))
        if current == new_mode:
            return
        _clear_all_gpt_responses()
        st.session_state.gpt_called_for_current_result = False
        st.session_state.trigger_gpt = False
        if new_mode == 'crop':
            st.session_state.manual_cropper_key = st.session_state.get('manual_cropper_key', 0) + 1
            st.session_state.last_gradcam_heatmap = None
            st.session_state.auto_detect_attempted = True
        elif new_mode == 'auto':
            st.session_state.auto_detect_attempted = False
        st.session_state.mode = new_mode
        st.session_state.current_mode = new_mode

    def _clear_manual_crop_state():
        st.session_state.manual_cropper_key = st.session_state.get('manual_cropper_key', 0) + 1
        st.session_state.last_cropped_result = None
        st.session_state.last_rect_coords = None
        st.session_state.crop_done = False
        st.session_state.gpt_called_for_current_result = False
        st.session_state.last_gradcam_heatmap = None
        for key in ('manual_gpt_response', 'gpt_response_manual'):
            if key in st.session_state:
                del st.session_state[key]

    def _render_mode_controls():
        current = st.session_state.get('mode', 'auto')
        ctrl_cols = st.columns([1, 1, 1])
        with ctrl_cols[0]:
            st.button(
                lang["auto_crop"],
                key="mode_switch_auto_bottom",
                disabled=(current == 'auto'),
                on_click=_switch_mode,
                args=('auto',)
            )
        with ctrl_cols[1]:
            st.button(
                lang["manual_crop"],
                key="mode_switch_manual_bottom",
                disabled=(current == 'crop'),
                on_click=_switch_mode,
                args=('crop',)
            )
        with ctrl_cols[2]:
            st.button(
                lang["clear_selection"],
                key="clear_manual_selection_bottom",
                disabled=(current == 'auto'),
                on_click=_clear_manual_crop_state
            )

    current_mode = st.session_state.get('mode', 'auto')

    if current_mode == 'auto':
        # st.info(lang["auto_mode"])
        classification_threshold = model_conf / 100.0
        detection_threshold = custom_conf / 100.0
        results = st.session_state.get('auto_detect_results', [])
        if results:
            # Build multi overlay
            overlay_multi = image.copy()
            draw = ImageDraw.Draw(overlay_multi)
            chosen = None
            
            # แสดงเลขเฉพาะเมื่อพบมากกว่า 1 ใบ
            show_numbers = len(results) > 1
            
            for idx, r in enumerate(results):
                if r['classification_confidence'] >= classification_threshold and chosen is None:
                    chosen = r
                bbox = r['bbox']
                passed = r['classification_confidence'] >= classification_threshold
                color = (0,255,0) if passed else (255,165,0)
                draw.rectangle([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']], outline=color, width=4)
                
                # แสดงเลขเฉพาะเมื่อมีมากกว่า 1 ใบ
                if show_numbers:
                    draw.text((bbox['left']+3, bbox['top']+3), f"{idx+1}", fill=color)
            if chosen is None:
                chosen = results[0]
            diseases_count = len(set(r['predicted_class'] for r in results))
            st.image(overlay_multi, use_container_width=True)
            # Update chosen into last_cropped_result for downstream blocks (unchanged logic)
            bbox = chosen['bbox']
            st.session_state.last_cropped_result = {
                'image': chosen['image'],
                'predicted_class': chosen['predicted_class'],
                'confidence': chosen['classification_confidence'],
                'crop_coords': {
                    'left': bbox['left'], 'top': bbox['top'], 'right': bbox['right'], 'bottom': bbox['bottom'],
                    'width': bbox['width'], 'height': bbox['height'], 'auto_detected': True
                }
            }
            st.session_state.last_rect_coords = f"auto-{bbox['left']}-{bbox['top']}-{bbox['right']}-{bbox['bottom']}"
            st.session_state.crop_done = True
            # Track chosen index for multi-GPT
            try:
                st.session_state.auto_detect_chosen_index = results.index(chosen)
            except Exception:
                st.session_state.auto_detect_chosen_index = 0
        else:
            # แสดงภาพต้นฉบับเมื่อไม่พบกรอบ
            st.image(image, use_container_width=True)
            # Clear result states เมื่อไม่มีการตรวจจับ
            st.session_state.last_cropped_result = None
            st.session_state.crop_done = False
        
        _render_mode_controls()

    else:
        st.info(lang["manual_mode"])
        rect = None
        crop_input_image = image
        scale_w = 1.0
        scale_h = 1.0
        crop_default_coords = None
        should_resize_image = True

        max_width_desktop = 720
        max_width_mobile = 360
        max_height_mobile = 520

        effective_max_width = max_width_mobile if is_mobile else max_width_desktop
        effective_max_height = max_height_mobile if is_mobile else None

        width_scale = effective_max_width / image.width if image.width > effective_max_width else 1.0
        height_scale = (effective_max_height / image.height) if (effective_max_height and image.height > effective_max_height) else 1.0
        scale = min(width_scale, height_scale)

        if scale < 1.0:
            crop_input_image = ImageOps.contain(
                image,
                (
                    int(effective_max_width),
                    int(effective_max_height) if effective_max_height else image.height
                )
            )
            scale_w = image.width / crop_input_image.width
            scale_h = image.height / crop_input_image.height
            should_resize_image = False
        else:
            crop_input_image = image
            should_resize_image = True

        if not st.session_state.get('last_rect_coords') or not str(st.session_state.get('last_rect_coords')).startswith('manual-'):
            margin = 0.2
            left = int(crop_input_image.width * margin)
            right = int(crop_input_image.width * (1 - margin))
            top = int(crop_input_image.height * margin)
            bottom = int(crop_input_image.height * (1 - margin))
            crop_default_coords = (left, right, top, bottom)

        try:
            rect = st_cropper(
                crop_input_image,
                realtime_update=True,
                box_color='#FF0000',
                aspect_ratio=None,
                return_type='box',
                key=f"manual_cropper_{st.session_state.get('manual_cropper_key', 0)}",
                should_resize_image=should_resize_image,
                default_coords=crop_default_coords
            )
        except Exception as e:
            st.warning(f"{lang['cropper_error']} {e}")
            rect = None

        if rect:
            if not should_resize_image:
                rect = {
                    'left': int(round(rect.get('left', 0) * scale_w)),
                    'top': int(round(rect.get('top', 0) * scale_h)),
                    'width': int(round(rect.get('width', 0) * scale_w)),
                    'height': int(round(rect.get('height', 0) * scale_h))
                }

            # rect อาจเป็น dict หรือ tuple
            if isinstance(rect, dict):
                x = int(rect.get('x', rect.get('left', 0)))
                y = int(rect.get('y', rect.get('top', 0)))
                w = int(rect.get('width', rect.get('w', 0)))
                h = int(rect.get('height', rect.get('h', 0)))
                l, t, r, b = x, y, x + w, y + h
            elif isinstance(rect, (list, tuple)) and len(rect) == 4:
                l, t, r, b = map(int, rect)
            else:
                rect = None

        if rect:
            W, H = image.size
            # ใช้ helper function _clamp_and_sort_bbox 
            l, t, r, b = _clamp_and_sort_bbox(l, t, r, b, W, H)
            if (r - l) < 10 or (b - t) < 10:
                st.warning(lang["crop_too_small"])
            else:
                crop_hash = f"manual-{l}-{t}-{r}-{b}"
                if crop_hash != st.session_state.get('last_rect_coords'):
                    manual_key = "manual_crop"
                    if 'per_box_gradcam_enabled' not in st.session_state:
                        st.session_state.per_box_gradcam_enabled = {}
                    if 'per_box_gradcam_heatmaps' not in st.session_state:
                        st.session_state.per_box_gradcam_heatmaps = {}

                    manual_on = st.session_state.per_box_gradcam_enabled.get(manual_key, False)
                    st.session_state.per_box_gradcam_heatmaps.pop(manual_key, None)
                    st.session_state.last_gradcam_heatmap = None

                    cropped_img = image.crop((l, t, r, b))
                    if manual_on:
                        predicted_class, confidence, heatmap = predict_with_gradcam(
                            model, cropped_img, transform, class_names, device, layer_name
                        )
                        st.session_state.per_box_gradcam_heatmaps[manual_key] = heatmap
                    else:
                        predicted_class, confidence = predict_image(model, cropped_img, transform, class_names, device)

                    st.session_state.last_cropped_result = {
                        'image': cropped_img,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'crop_coords': {
                            'left': l, 'top': t, 'right': r, 'bottom': b,
                            'width': r - l, 'height': b - t, 'auto_detected': False
                        }
                    }
                    st.session_state.last_rect_coords = crop_hash
                    st.session_state.gpt_called_for_current_result = False
                    st.session_state.crop_done = True
                    
                    # ล้าง GPT response เมื่อเปลี่ยนตำแหน่งครอป
                    if 'gpt_response_manual' in st.session_state:
                        del st.session_state.gpt_response_manual
                    if 'manual_gpt_response' in st.session_state:
                        del st.session_state.manual_gpt_response

                    # ไม่เรียก GPT อัตโนมัติในโหมด manual crop
                    # ให้ผู้ใช้กดปุ่มเรียก GPT เอง
                    st.session_state.trigger_gpt = False
        _render_mode_controls()

    # แสดงผลลัพธ์เมื่อมีการวิเคราะห์เสร็จแล้ว
    last_result = st.session_state.get('last_cropped_result')
    if last_result is not None and not st.session_state.get('skip_single_result_block'):
        result = last_result
        
        # Show cropped result
        st.markdown("---")
        st.subheader(lang["analysis_result"])

           # คู่มือการอ่านผล
        with st.expander(lang["gradcam_expander"]):
             st.text(f"{lang['gradcam_text']}")


        # Check if auto mode with multiple detections
        auto_results = st.session_state.get('auto_detect_results') or []
        if st.session_state.get('mode') == 'auto' and auto_results:

            # เตรียม state dict สำหรับแต่ละใบ
            if 'per_box_gradcam_enabled' not in st.session_state:
                st.session_state.per_box_gradcam_enabled = {}
            if 'per_box_gradcam_heatmaps' not in st.session_state:
                st.session_state.per_box_gradcam_heatmaps = {}

            classification_threshold = model_conf / 100.0

            total_leaves = len(auto_results)

            for idx, r in enumerate(auto_results):
                box_key = f"leaf_{idx}"
                crop_img = r['image']
                bbox = r['bbox']
                passed = r['classification_confidence'] >= classification_threshold

                base_leaf_label = lang['leaf_num'].strip()
                if total_leaves > 1:
                    leaf_label = f"{base_leaf_label} {idx+1}"
                else:
                    leaf_label = base_leaf_label
                
                # Container UI
                with st.container():
                    st.markdown(f"---")
                    header_cols = st.columns([4,1])
                    with header_cols[0]:
                        st.subheader(leaf_label)
                    with header_cols[1]:
                        # Toggle Grad-CAM เฉพาะใบ
                        current_on = st.session_state.per_box_gradcam_enabled.get(box_key, False)
                        new_on = st.toggle(lang["gradcam"], value=current_on, key=f"toggle_gc_{box_key}")
                        if new_on != current_on:
                            st.session_state.per_box_gradcam_enabled[box_key] = new_on
                            if new_on and box_key not in st.session_state.per_box_gradcam_heatmaps:
                                # คำนวณ heatmap เฉพาะใบนี้
                                try:
                                    _, _, heatmap_local = predict_with_gradcam(model, crop_img, transform, class_names, device, layer_name)
                                    st.session_state.per_box_gradcam_heatmaps[box_key] = heatmap_local
                                except Exception:
                                    st.session_state.per_box_gradcam_heatmaps[box_key] = None

                    # แสดงภาพครอป (ถ้ามี Grad-CAM เปิด ใส่ overlay เฉพาะครอป)
                    display_cols = st.columns([1,2])
                    with display_cols[0]:
                        # แสดงภาพเต็ม + กรอบตำแหน่งที่ตรวจ
                        base_img = image
                        if st.session_state.per_box_gradcam_enabled.get(box_key) and st.session_state.per_box_gradcam_heatmaps.get(box_key) is not None:
                            try:
                                # ใส่ Grad-CAM overlay บนภาพเต็ม
                                base_img = overlay_gradcam(image, crop_img, st.session_state.per_box_gradcam_heatmaps[box_key], alpha=0.5, crop_coords=bbox)
                            except Exception:
                                base_img = image
                        
                        # วาดกรอบบนภาพเต็ม
                        full_with_box = _draw_bbox_on_image(base_img, {
                            'left': bbox['left'], 'top': bbox['top'], 'right': bbox['right'], 'bottom': bbox['bottom']
                        }, color=(0,255,0) if passed else (255,165,0), width=4)
                        caption_txt = f"{leaf_label} - {lang['full_image_box']}"
                        if st.session_state.per_box_gradcam_enabled.get(box_key):
                            caption_txt += " + Grad-CAM"
                        
                        st.image(full_with_box, use_container_width=True)
                    with display_cols[1]:
                        classification_threshold = model_conf / 100.0
                        if r['classification_confidence'] >= classification_threshold:
                            st.write(f"**{lang['detected_disease']}** {r['predicted_class']}")
                            st.write(f"**{lang['confidence']}** {r['classification_confidence']:.2%}")
                            st.success(lang["clear_image"])
                        elif r['classification_confidence'] >= classification_threshold*0.6:
                            st.warning(lang["low_confidence"])
                            st.markdown(lang["low_confidence_tip"])
                        else:
                            st.error(lang["unclear_analysis"])
                            st.markdown(lang["unclear_analysis_tip"])
                        # เพิ่มปุ่มขอคำแนะนำ (เฉพาะเมื่อผ่าน threshold)
                        if r['classification_confidence'] >= classification_threshold:
                            # สร้าง key สำหรับเก็บ response ของแต่ละใบ
                            gpt_key = f"gpt_response_leaf_{idx}"
                            
                            with st.expander(lang["request_advice"]):
                                if st.button(lang["expert_advice"], key=f"gpt_btn_{idx}"):
                                    with st.spinner(lang["analyzing"]):
                                        try:
                                            gpt_response = call_gpt(
                                                r['predicted_class'], 
                                                r['classification_confidence'], 
                                                st.session_state.selected_language
                                            )
                                            st.session_state[gpt_key] = gpt_response

                                        except Exception as e:
                                            st.error(f"เกิดข้อผิดพลาดในการเรียก GPT: {str(e)}")

                                
                                # แสดงผลลัพธ์ GPT ถ้ามี
                                if gpt_key in st.session_state:
                                    st.markdown(f"**{lang['ai_advice']}**")
                                    st.success(lang["analysis_complete"])
                                    st.markdown(
                                        f"<div style='max-height:320px; overflow-y:auto; border:1px solid #ccc; padding:0.75rem; border-radius:6px; background-color:rgba(0,0,0,0.02); white-space:pre-wrap;'>{st.session_state[gpt_key]}</div>",
                                        unsafe_allow_html=True
                                    )
        else:
            # Manual mode display with Grad-CAM toggle (similar to auto mode)
            with st.container():
                st.markdown("---")
                header_cols = st.columns([4,1])
                with header_cols[0]:
                    st.subheader(lang["analysis_results"])
                with header_cols[1]:
                    # Toggle Grad-CAM for manual crop
                    manual_key = "manual_crop"
                    if 'per_box_gradcam_enabled' not in st.session_state:
                        st.session_state.per_box_gradcam_enabled = {}
                    if 'per_box_gradcam_heatmaps' not in st.session_state:
                        st.session_state.per_box_gradcam_heatmaps = {}
                    
                    current_on = st.session_state.per_box_gradcam_enabled.get(manual_key, False)
                    new_on = st.toggle("Grad-CAM", value=current_on, key=f"toggle_gc_{manual_key}")
                    if new_on != current_on:
                        st.session_state.per_box_gradcam_enabled[manual_key] = new_on
                        if new_on and manual_key not in st.session_state.per_box_gradcam_heatmaps:
                            # คำนวณ heatmap สำหรับ manual crop
                            try:
                                _, _, heatmap_local = predict_with_gradcam(model, result['image'], transform, class_names, device, layer_name)
                                st.session_state.per_box_gradcam_heatmaps[manual_key] = heatmap_local
                            except Exception:
                                st.session_state.per_box_gradcam_heatmaps[manual_key] = None
                        if not new_on:
                            st.session_state.per_box_gradcam_heatmaps.pop(manual_key, None)

                    if st.session_state.per_box_gradcam_enabled.get(manual_key, False) and manual_key not in st.session_state.per_box_gradcam_heatmaps:
                        try:
                            _, _, heatmap_local = predict_with_gradcam(model, result['image'], transform, class_names, device, layer_name)
                            st.session_state.per_box_gradcam_heatmaps[manual_key] = heatmap_local
                        except Exception:
                            st.session_state.per_box_gradcam_heatmaps[manual_key] = None

                # แสดงภาพ
                display_cols = st.columns([1,2])
                with display_cols[0]:
                    # Display FULL image (optionally Grad-CAM overlay) with bbox
                    bbox = result.get('crop_coords', {})
                    base_img = image
                    if st.session_state.per_box_gradcam_enabled.get(manual_key) and st.session_state.per_box_gradcam_heatmaps.get(manual_key) is not None:
                        try:
                            base_img = overlay_gradcam(image, result['image'], st.session_state.per_box_gradcam_heatmaps[manual_key], alpha=0.5, crop_coords=bbox)
                        except Exception:
                            base_img = image
                    
                    if all(k in bbox for k in ['left','top','right','bottom']):
                        full_with_box = _draw_bbox_on_image(base_img, {
                            'left': bbox['left'], 'top': bbox['top'], 'right': bbox['right'], 'bottom': bbox['bottom']
                        }, color=(0,255,0) if result['confidence'] >= (model_conf/100.0) else (255,165,0), width=4)
                    else:
                        full_with_box = base_img
                    
                    caption_txt = lang["manual_caption"]
                    if st.session_state.per_box_gradcam_enabled.get(manual_key):
                        caption_txt += " + Grad-CAM"
                    
                    st.image(full_with_box, use_container_width=True)
                
                with display_cols[1]:
                    # Prediction results manually cropped or single detection
                    classification_threshold = model_conf / 100.0
                    if result['confidence'] >= classification_threshold:
                        st.write(f"**{lang['detected_disease']}** {result['predicted_class']}")
                        st.write(f"**{lang['confidence']}** {result['confidence']:.2%}")
                        st.success(lang["clear_image"])
                    elif result['confidence'] >= classification_threshold*0.6:
                        st.warning(lang["low_confidence"])
                        st.markdown(lang["low_confidence_tip"])
                    else:
                        st.error(lang["unclear_analysis"])
                        st.markdown(lang["unclear_analysis_tip"])

                    # เพิ่มปุ่มขอคำแนะนำสำหรับ manual crop (เฉพาะเมื่อผ่าน threshold)
                    if result['confidence'] >= classification_threshold:
                        # สร้าง key สำหรับเก็บ response ของ manual crop
                        manual_gpt_key = "gpt_response_manual"
                        
                        with st.expander(lang["request_advice"]):
                            if st.button(lang["expert_advice"], key="manual_gpt_btn"):
                                with st.spinner(lang["analyzing"]):
                                    try:
                                        gpt_response = call_gpt(result['predicted_class'], result['confidence'], st.session_state.selected_language)
                                        st.session_state[manual_gpt_key] = gpt_response
                                    except Exception as e:
                                        st.error(f"เกิดข้อผิดพลาดในการเรียก GPT: {str(e)}")
                            
                            # แสดงผลลัพธ์ GPT ถ้ามี
                            if manual_gpt_key in st.session_state:
                                st.markdown(f"**{lang['ai_advice']}**")
                                st.success(lang["analysis_complete"])
                                st.write(st.session_state[manual_gpt_key])



    import io
    from typing import Optional, Tuple
    from PIL import Image, ImageDraw

    def _to_pil(img):
        """Ensure image is a PIL.Image."""
        if isinstance(img, Image.Image):
            return img
        try:
            # numpy array -> PIL
            return Image.fromarray(img)
        except Exception:
            raise TypeError("overlay_gradcam() should return PIL.Image or numpy array.")

    def _bbox_from_crop_coords(crop_coords: dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Normalize crop_coords into (left, top, right, bottom) in ORIGINAL image coordinates.
        Supports several key conventions: 
        - x, y, w, h
        - left, top, right, bottom
        - xmin, ymin, xmax, ymax
        Returns None if not enough info.
        """
        if not crop_coords:
            return None

        # Case 1: x, y, w, h (or width/height)
        if all(k in crop_coords for k in ("x", "y")) and (("w" in crop_coords) or ("width" in crop_coords)) and (("h" in crop_coords) or ("height" in crop_coords)):
            x = float(crop_coords["x"])
            y = float(crop_coords["y"])
            w = float(crop_coords.get("w", crop_coords.get("width")))
            h = float(crop_coords.get("h", crop_coords.get("height")))
            return (int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))

        # Case 2: left, top, right, bottom
        if all(k in crop_coords for k in ("left", "top", "right", "bottom")):
            return (
                int(round(float(crop_coords["left"]))),
                int(round(float(crop_coords["top"]))),
                int(round(float(crop_coords["right"]))),
                int(round(float(crop_coords["bottom"]))),
            )

        # Case 3: xmin, ymin, xmax, ymax
        if all(k in crop_coords for k in ("xmin", "ymin", "xmax", "ymax")):
            return (
                int(round(float(crop_coords["xmin"]))),
                int(round(float(crop_coords["ymin"]))),
                int(round(float(crop_coords["xmax"]))),
                int(round(float(crop_coords["ymax"]))),
            )

        return None

    def _scale_bbox(bbox, src_size, dst_size):
        """
        Scale bbox from src_size (W, H) space into dst_size (W, H) space.
        bbox: (l, t, r, b)
        """
        l, t, r, b = bbox
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        fx = dst_w / float(src_w)
        fy = dst_h / float(src_h)
        l2 = int(round(l * fx))
        r2 = int(round(r * fx))
        t2 = int(round(t * fy))
        b2 = int(round(b * fy))
        return (l2, t2, r2, b2)

    def _clamp_bbox(bbox, w, h):
        """Clamp bbox to image bounds."""
        l, t, r, b = bbox
        l = max(0, min(l, w - 1))
        r = max(0, min(r, w - 1))
        t = max(0, min(t, h - 1))
        b = max(0, min(b, h - 1))
        # Ensure proper ordering
        if r < l: l, r = r, l
        if b < t: t, b = b, t
        return (l, t, r, b)

