import os;
import dotenv;
from openai import AzureOpenAI,OpenAI;
from pydantic import BaseModel, Field;
from typing import Literal
import base64
from mimetypes import guess_type
import argparse;
import pathlib;
from tqdm.auto import tqdm;
import time;
import csv;
from google import genai;
import mimetypes;
import json;
from google.api_core import retry
import anthropic;
import torch;
import torchvision;
import torch.nn as nn;
from PIL import Image;

dotenv.load_dotenv(dotenv.find_dotenv(), override=True);
endpoint = os.getenv("AZURE_API_FOUNDRY_ENDPOINT")
subscription_key = os.getenv("AZURE_API_FOUNDRY")
api_version = "2024-12-01-preview"

LABELS = ["healthy", "leaf sheath and leaf spot", "virus", "rust", "northern corn leaf blight", "downy mildew"];
SYSTEM_PROMPT = open("./prompts/system2.txt").read();
USER_PROMPT = open("./prompts/user2.txt").read();


class CornLeafDiseaseResponse(BaseModel):
    response : Literal["healthy",
                       "leaf sheath and leaf spot",
                       "virus",
                       "rust",
                       "northern corn leaf blight",
                       "downy mildew"] = \
        Field(..., description="The type of corn leaf disease identified.");

gemini_schema = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    properties={
        "response": genai.types.Schema(type=genai.types.Type.STRING, enum=LABELS),
    },
    required=["response"],
);

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}";

def resnet_tf(sz = 224):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(sz),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])


def classify_with_chatgpt(image_path: str, model_name: str, client: OpenAI) -> str:
    data_url = local_image_to_data_url(image_path)

    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "response_format": CornLeafDiseaseResponse,
        "seed": 42,
    }

    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.parse(**kwargs)

    return resp.choices[0].message.parsed.response


# https://www.alibabacloud.com/help/en/model-studio/getting-started/models
def classify_with_qwen(
    image_path: str,
    model_name: str,        
    client
) -> str:

    data_url = local_image_to_data_url(image_path)

    # หมายเหตุ: โหมด JSON ต้องมีคำว่า "JSON" ใน system หรือ user (เราใส่ทั้งสองฝั่งเพื่อความชัวร์)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"{USER_PROMPT}\n\n"
                        "Valid labels: " + ", ".join(LABELS) + ". "
                        "Return JSON ONLY (no prose, no markdown, no code fences)."
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},  # JSON mode
        temperature=0.0,
        seed=42,                                 # รองรับ seed ตามเอกสาร
    )

    content = completion.choices[0].message.content or "{}"

    # กันกรณีบางรุ่นเผลอใส่ code fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # เผื่อรูปแบบ ```json\n{...}\n```
        if "\n" in cleaned:
            cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))

    data = json.loads(cleaned)
    parsed = CornLeafDiseaseResponse(**data)
    return parsed.response

def classify_with_claude(image_path: str, model_name: str, anthro) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt_text = USER_PROMPT if isinstance(USER_PROMPT, str) else str(USER_PROMPT)

    # --- tool + schema ---
    tool = {
        "name": "submit_classification",
        "description": "Return classification strictly as schema",
        "input_schema": {
            "type": "object",
            "properties": {
                "response": {"type": "string", "enum": LABELS}
            },
            "required": ["response"],
            "additionalProperties": False
        }
    }

    # --- เรียก Messages API (ใช้ SYSTEM_PROMPT จากไฟล์ของคุณ) ---
    msg = anthro.messages.create(
        model=model_name,
        max_tokens=256,
        temperature=0,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_classification"},
        system=f"{SYSTEM_PROMPT}\n\nYou must respond ONLY by calling the tool with a payload that matches the schema.",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64}
                },
                {
                    "type": "text",
                    "text": f"{prompt_text}\n\nReturn ONLY by calling the tool `submit_classification`."
                }
            ]
        }]
    )

    # --- ดึงผลจาก tool_use ---
    for block in (msg.content or []):
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_classification":
            return str(block.input.get("response", "healthy")).lower()

    # Fallback (ไม่คาดหวังว่าจะถึงตรงนี้เพราะบังคับ tool_choice แล้ว)
    for block in (msg.content or []):
        if getattr(block, "type", None) == "text":
            txt = (block.text or "").strip()
            try:
                data = json.loads(txt)
                if isinstance(data, dict) and "response" in data:
                    return str(data["response"]).lower()
            except Exception:
                pass
            return (txt.split()[0].lower() if txt else "healthy")

    return "healthy"

# Retry logic
def is_retryable(e) -> bool:
    if retry.if_transient_error(e):
        return True
    elif isinstance(e, genai.errors.ClientError) and e.code == 429:
        return True
    elif isinstance(e, genai.errors.ServerError) and e.code == 503:
        return True
    else:
        return False


# Image classification with retry
@retry.Retry(predicate=is_retryable)
def classify_with_gemini(image_path: str, model_name: str, client) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    content = {
        "role": "user",
        "parts": [
            {"text": f"{USER_PROMPT}\n\nReturn ONLY a JSON object with a key 'response' and value from this list: {LABELS}"},
            {"inline_data": {"mime_type": mime, "data": b64}},
        ],
    }

    resp = client.models.generate_content(
        model=model_name,
        contents=[content],
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            seed=42,
            system_instruction=SYSTEM_PROMPT
        ),
    )

    # Parse response
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        txt = getattr(resp, "text", None)
        if not txt and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            txt = "".join(getattr(p, "text", "") for p in parts)
        try:
            parsed = json.loads(txt or "{}")
        except json.JSONDecodeError:
            raise ValueError("Failed to decode JSON from Gemini response.")

    label = parsed.get("response", "")
    if label not in LABELS:
        raise ValueError(f"Unexpected label: {label}")
    return label

@torch.inference_mode()
def use_resnet18(image_path: str,tf : torchvision.transforms):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    logits = model(x)            # [1, num_classes]
    probs = torch.softmax(logits, dim=1)[0]  # [num_classes]

    top_id = int(torch.argmax(probs).item())
    top_label = id2class.get(top_id, str(top_id))
    top_p = float(probs[top_id].item())

    return top_label, top_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='/mnt/c/fusion/corn_project/All_Crops/test/')
    parser.add_argument("--model", type=str, default="BlueNJ");
    parser.add_argument("--log", type=str, default="results2.csv")
    parser.add_argument("--rpm", type=int, default=0, help="Rate limit (requests per minute). 0 means no limit.")
    parser.add_argument("--size",type=int,default=224);
    parser.add_argument("--weight_path",type=str,default="/mnt/c/fusion/corn_project/model_training/resnet18VsOthers/resnet18_224_cosine_croponly_6_b32/checkpoints/best.pth")
    args = parser.parse_args();

    state = torch.load(args.weight_path, map_location="cpu")
    class2id = {k.lower(): v for k, v in state["class2id"].items()}

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(state["class2id"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    id2class = {v: k for k, v in class2id.items()}

    print(class2id)

    if "gpt-4.1" in str(args.model).lower():
        client = AzureOpenAI(api_key=subscription_key, azure_endpoint=endpoint, api_version=api_version)
    elif "gemini" in str(args.model).lower():
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"));
    elif "claude" in str(args.model).lower():
        client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"));
    elif "qwen" in str(args.model).lower():
        client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        );
    elif "gpt-4o" in str(args.model).lower() or "gpt-5" in str(args.model).lower():
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"));
    elif "resnet18" in str(args.model).lower():
        client = resnet_tf(args.size)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    
    print(f"🤖 Using model: {args.model}");

    all_images = []
    for sub in pathlib.Path(args.image_dir).iterdir():
        if sub.is_dir() and sub.name.lower().split("_")[0] in class2id.keys():
            for img in sub.iterdir():
                all_images.append((img, sub.name.split("_")[0].lower()))

    out = open(args.log,"w",newline="",encoding="utf-8")
    if not "resnet18" in str(args.model).lower():
        writer = csv.writer(out); writer.writerow(["image","gt","prediction"]);
    else:
        writer = csv.writer(out); writer.writerow(["image","gt","prediction","confidence"]);

    for img, gt in tqdm(all_images, desc="Classifying", unit="img"):
        if "gpt-4.1" in str(args.model).lower():
            data_url = local_image_to_data_url(img)
            resp = client.chat.completions.parse(
                model=args.model,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":[
                        {"type":"text","text":USER_PROMPT},
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":data_url
                            }
                        }
                    ]}
                ],
                response_format=CornLeafDiseaseResponse,
                temperature=0.0,
                seed=42,
            );
            pred = resp.choices[0].message.parsed.response
        elif "gemini" in args.model.lower():
            pred = classify_with_gemini(str(img), args.model, client);
        elif "claude" in args.model.lower():
            pred = classify_with_claude(str(img), args.model, client);
        elif "qwen" in args.model.lower():
            pred = classify_with_qwen(str(img), args.model, client);
        elif "gpt-4o" in args.model.lower() or "gpt-5" in str(args.model).lower():
            pred = classify_with_chatgpt(str(img), args.model, client);
        elif "resnet18" in str(args.model).lower():
            pred,p = use_resnet18(str(img),client);

        if not "resnet18" in str(args.model).lower():
            writer.writerow([img.name, gt, pred]);
        else:
            writer.writerow([img.name, gt, pred, p])

        out.flush()
        if args.rpm>0: time.sleep(60/args.rpm)  # rate limit
    out.close()
    print(f"✅ Done. Log saved to {args.log}")
            

# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model gpt-4.1 --log chatbot_prediction/gpt4_1_224.csv --rpm 60
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model gemini-2.5-flash --log chatbot_prediction/gemini_2_5_flash_224.csv --rpm 50
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model gemini-2.5-pro --log chatbot_prediction/gemini_2_5_pro_224.csv --rpm 50
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model claude-sonnet-4-20250514 --log chatbot_prediction/claude-sonnet-4-20250514_224.csv --rpm 50
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model qwen3-vl-235b-a22b-instruct --log chatbot_prediction/qwen3-vl-235b-a22b-instruct_224.csv --rpm 50
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model gpt-4o --log chatbot_prediction/gpt-4o_224.csv --rpm 60
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model gpt-5 --log chatbot_prediction/gpt-5_224.csv --rpm 120
# /bin/python3 /mnt/c/fusion/corn_project/agent_testv2.py --image_dir /mnt/c/fusion/corn_project/All_Crops/test --model resnet18 --log chatbot_prediction/resnet18_224.csv --rpm 0 --weight_path /mnt/c/fusion/corn_project/model_training/resnet18VsOthers/resnet18_224_cosine_croponly_6_b32/checkpoints/best.pth