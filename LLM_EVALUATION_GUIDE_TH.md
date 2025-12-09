# คู่มือการประเมิน LLM สำหรับการจำแนกโรคข้าวโพด

**โครงการ:** Intelligent AgriDiagnose  
**Repository:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/  
**อัปเดตล่าสุด:** 9 ธันวาคม 2568

---

## สารบัญ

1. [ภาพรวม](#ภาพรวม)
2. [โมเดลที่ประเมิน](#โมเดลที่ประเมิน)
3. [สคริปต์การประเมิน](#สคริปต์การประเมิน)
4. [สรุปผลลัพธ์](#สรุปผลลัพธ์)
5. [วิธีรันการประเมิน](#วิธีรันการประเมิน)
6. [Prompts](#prompts)
7. [การวิเคราะห์](#การวิเคราะห์)

---

## 1. ภาพรวม

เอกสารนี้อธิบายการประเมินโมเดล Large Language Models (LLMs) ต่างๆ สำหรับการจำแนกโรคใบข้าวโพดโดยใช้ความสามารถด้าน vision การประเมินเปรียบเทียบ:

- **Traditional CNN**: ResNet-18 ที่เทรนบน dataset โรคข้าวโพด
- **โมเดล GPT**: GPT-4o, GPT-4.1-mini, GPT-5
- **Claude**: Claude Sonnet 4
- **Gemini**: Gemini 2.5 Flash, Gemini 2.5 Pro
- **Qwen**: Qwen3-VL-235B

### ขนาดภาพที่ทดสอบ
- **224×224 พิกเซล**: ขนาดมาตรฐาน

---

## 2. โมเดลที่ประเมิน

### 2.1 Traditional CNN

| โมเดล | สถาปัตยกรรม | ขนาด Input | การเทรน |
|-------|-------------|-----------|---------|
| ResNet-18 | CNN with transfer learning | 224×224 | Fine-tuned บน dataset โรคข้าวโพด |

### 2.2 โมเดล OpenAI

| โมเดล | เวอร์ชัน | API | ขนาด Input |
|-------|---------|-----|-----------|
| GPT-4o | ล่าสุด | Azure OpenAI | 224×224, 224×224 |
| GPT-4.1-mini | ล่าสุด | Azure OpenAI | 224×224 |
| GPT-5 | Preview | OpenAI | 224×224, 224×224 |

### 2.3 โมเดล Anthropic

| โมเดล | เวอร์ชัน | API | ขนาด Input |
|-------|---------|-----|-----------|
| Claude Sonnet 4 | 2025-05-14 | Anthropic | 224×224, 224×224 |

### 2.4 โมเดล Google

| โมเดล | เวอร์ชัน | API | ขนาด Input |
|-------|---------|-----|-----------|
| Gemini 2.5 Flash | ล่าสุด | Google AI | 224×224 |
| Gemini 2.5 Pro | ล่าสุด | Google AI | 224×224 |

### 2.5 โมเดล Qwen

| โมเดล | เวอร์ชัน | API | ขนาด Input |
|-------|---------|-----|-----------|
| Qwen3-VL-235B | a22b-instruct | Alibaba Cloud | 224×224, 224×224 |

---

## 3. สคริปต์การประเมิน

### ไฟล์หลัก

| ไฟล์ | วัตถุประสงค์ |
|------|--------------|
| `agent_test.py` | สคริปต์ประเมินหลักสำหรับ LLM ทั้งหมด |
| `agent_testv2.py` | การทำงานแบบอื่น |
| `prompts/system2.txt` | System prompt สำหรับ LLMs |
| `prompts/user2.txt` | User prompt template |

### โครงสร้างสคริปต์

```python
# agent_test.py ส่วนประกอบหลัก

# 1. Model Response Schema
class CornLeafDiseaseResponse(BaseModel):
    response: Literal[
        "rust", "blight", "spot", "virus", 
        "mildew", "healthy", "abnormality", 
        "brownspot", "curl", "smut"
    ]

# 2. ฟังก์ชันจำแนก
def classify_with_chatgpt(image_path, model_name, client) -> str
def classify_with_claude(image_path, model_name, client) -> str
def classify_with_gemini(image_path, model_name, client) -> str
def classify_with_qwen(image_path, model_name, client) -> str

# 3. Image Encoding
def encode_image(image_path: str) -> str:
    """Encode ภาพเป็น base64 สำหรับส่ง API"""
    
# 4. Main Evaluation Loop
for each test image:
    - โหลดภาพ
    - เรียก LLM API ที่เหมาะสม
    - Parse response
    - เปรียบเทียบกับ ground truth
    - บันทึกผลลัพธ์ลง CSV
```

### โรคที่รองรับ

```python
LABELS = [
    "blight",      # Northern Corn Leaf Blight
    "rust",        # สนิมใบข้าวโพด
    "virus",       # โรคไวรัส
    "mildew",      # โรคราน้ำค้าง
    "healthy",     # สุขภาพดี
    "spot"         # โรคจุดใบ
]
```

---

## 4. สรุปผลลัพธ์

### 4.1 ไฟล์ผลลัพธ์

ผลลัพธ์ทั้งหมดจัดเก็บใน `chatbot_prediction/`:

```
chatbot_prediction/
├── resnet18_224.csv              # ResNet-18 baseline (224×224)
├── resnet18.csv                   # ResNet-18 baseline (224×224)
├── gpt-4o_224.csv                # ผลลัพธ์ GPT-4o (224×224)
├── gpt-4o.csv                     # ผลลัพธ์ GPT-4o (224×224)
├── gpt4_1_224.csv                # ผลลัพธ์ GPT-4.1-mini (224×224)
├── gpt-5_224.csv                 # ผลลัพธ์ GPT-5 (224×224)
├── gpt-5.csv                      # ผลลัพธ์ GPT-5 (224×224)
├── claude-sonnet-4-20250514_224.csv  # Claude Sonnet 4 (224×224)
├── claude-sonnet-4-20250514.csv      # Claude Sonnet 4 (224×224)
├── gemini_2_5_flash_224.csv      # Gemini 2.5 Flash (224×224)
├── gemini_2_5_pro_224.csv        # Gemini 2.5 Pro (224×224)
├── qwen3-vl-235b-a22b-instruct_224.csv  # Qwen3-VL (224×224)
└── qwen3-vl-235b-a22b-instruct.csv      # Qwen3-VL (224×224)
```

### 4.2 รูปแบบ CSV

แต่ละไฟล์ CSV ประกอบด้วย:

```csv
image,gt,prediction,confidence
Blight_crop_0005.jpg,blight,blight,0.7785653
Healthy_crop_0013.jpg,healthy,healthy,0.9670238
Rust_crop_0008.jpg,rust,rust,0.8765339
```

**คอลัมน์:**
- `image`: ชื่อไฟล์ภาพทดสอบ
- `gt`: Ground truth label
- `prediction`: การทำนายของโมเดล
- `confidence`: คะแนนความเชื่อมั่น (0-1)

### 4.3 การวิเคราะห์เบื้องต้น

จากไฟล์ CSV:

**ประสิทธิภาพ ResNet-18 (224×224):**
- Baseline ที่แข็งแกร่งด้วยคะแนนความเชื่อมั่นสูง
- มีการจำแนกผิดบ้างระหว่างโรคที่คล้ายกัน
- ตัวอย่าง: บางภาพ blight จำแนกผิดเป็น healthy หรือ rust

**ข้อสังเกตทั่วไป:**
- โมเดลทำงานได้ดีบนใบที่มีสุขภาพดี
- สับสนระหว่างโรคที่มีลักษณะคล้ายกัน (เช่น blight vs rust)
- Mildew บางครั้งสับสนกับ virus
- คะแนนความเชื่อมั่นแตกต่างกันมากระหว่างโมเดล

---

## 5. วิธีรันการประเมิน

### 5.1 ข้อกำหนดเบื้องต้น

```bash
# ติดตั้ง dependencies
pip install openai anthropic google-generativeai torch torchvision pillow tqdm python-dotenv pydantic

# ตั้งค่า environment variables
export AZURE_API_FOUNDRY_ENDPOINT="your_azure_endpoint"
export AZURE_API_FOUNDRY="your_azure_api_key"
export GEMINI_API_KEY="your_gemini_key"
export CLAUDE_API_KEY="your_claude_key"
export QWEN_API_KEY="your_qwen_key"
```

### 5.2 การใช้งานพื้นฐาน

```bash
# ทดสอบด้วย GPT-4o
python agent_test.py \
    --model "gpt-4o" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/gpt4o_results.csv" \
    --img_size 224

# ทดสอบด้วย Claude Sonnet 4
python agent_test.py \
    --model "claude-sonnet-4-20250514" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/claude_results.csv" \
    --img_size 224

# ทดสอบด้วย Gemini 2.5 Pro
python agent_test.py \
    --model "gemini-2.5-pro" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/gemini_results.csv" \
    --img_size 224

# ทดสอบด้วย Qwen3-VL
python agent_test.py \
    --model "qwen3-vl-235b-a22b-instruct" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/qwen_results.csv" \
    --img_size 224
```

### 5.3 พารามิเตอร์ Command-Line

```python
parser.add_argument("--model", type=str, required=True,
                   help="ชื่อโมเดล: gpt-4o, gpt-5, claude-sonnet-4, gemini-2.5-pro, etc.")
parser.add_argument("--test_dir", type=str, required=True,
                   help="ไดเรกทอรีที่มีภาพทดสอบ")
parser.add_argument("--output_csv", type=str, default="results.csv",
                   help="เส้นทางไฟล์ CSV output")
parser.add_argument("--img_size", type=int, default=224,
                   help="ขนาดภาพ (224 หรือ 224)")
parser.add_argument("--selected_classes", type=str,
                   default="rust,blight,spot,virus,mildew,healthy",
                   help="รายการ classes ที่จะประเมิน คั่นด้วยคอมม่า")
```

### 5.4 การประมวลผลแบบ Batch

```bash
# รันการประเมินสำหรับโมเดลทั้งหมด
for model in "gpt-4o" "claude-sonnet-4-20250514" "gemini-2.5-pro" "qwen3-vl-235b-a22b-instruct"
do
    python agent_test.py \
        --model $model \
        --test_dir "./All_Crops/test" \
        --output_csv "./chatbot_prediction/${model}_224.csv" \
        --img_size 224
done
```

---

## 6. Prompts

### 6.1 System Prompt

อยู่ใน `prompts/system2.txt`:

```
คุณเป็น AI ผู้ช่วยด้านการเกษตรที่เชี่ยวชาญในการวินิจฉัยโรคใบข้าวโพด
งานของคุณคือวิเคราะห์ภาพใบข้าวโพดและระบุโรคด้วยความแม่นยำสูง

โรคที่มี:
- rust: สนิมใบข้าวโพด (จุดสีส้ม-น้ำตาล)
- blight: โรคใบไหม้ (แผลยาว)
- spot: โรคจุดใบ (จุดกลม)
- virus: โรคไวรัส (ลายด่างหรือแถบ)
- mildew: โรคราน้ำค้าง (รากวมสีขาว-เทา)
- healthy: ไม่มีอาการโรค

ให้ระบุเฉพาะชื่อโรคจากรายการข้างต้น แม่นยำและมั่นใจในการวินิจฉัย
```

### 6.2 User Prompt

อยู่ใน `prompts/user2.txt`:

```
วิเคราะห์ภาพใบข้าวโพดนี้และระบุโรคที่พบ
ตอบด้วยชื่อโรคเพียงอย่างเดียวจาก: rust, blight, spot, virus, mildew, หรือ healthy

ชื่อโรค:
```

### 6.3 Structured Output

โมเดลทั้งหมดถูกตั้งค่าให้คืนค่า JSON แบบมีโครงสร้าง:

```json
{
  "response": "rust"
}
```

สิ่งนี้ทำให้การ parse สอดคล้องกันระหว่าง LLM providers ต่างๆ

---

## 7. การวิเคราะห์

### 7.1 เมตริกการเปรียบเทียบ

คำนวณ accuracy metrics จากไฟล์ CSV:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# โหลดผลลัพธ์
df = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')

# คำนวณ accuracy
accuracy = accuracy_score(df['gt'], df['prediction'])
print(f"Accuracy: {accuracy:.4f}")

# สร้าง classification report
report = classification_report(df['gt'], df['prediction'])
print(report)
```

### 7.2 การวิเคราะห์ Confidence

```python
# วิเคราะห์คะแนนความเชื่อมั่น
df = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')

# ค่าเฉลี่ย confidence
avg_conf = df['confidence'].mean()
print(f"Average Confidence: {avg_conf:.4f}")

# Confidence ตามความถูกต้อง
correct = df[df['gt'] == df['prediction']]['confidence'].mean()
incorrect = df[df['gt'] != df['prediction']]['confidence'].mean()
print(f"Confidence (ถูกต้อง): {correct:.4f}")
print(f"Confidence (ผิด): {incorrect:.4f}")
```

### 7.3 ประสิทธิภาพแต่ละ Class

```python
# Accuracy แต่ละโรค
for disease in df['gt'].unique():
    disease_df = df[df['gt'] == disease]
    disease_acc = (disease_df['gt'] == disease_df['prediction']).mean()
    print(f"{disease}: {disease_acc:.4f}")
```

### 7.4 การวิเคราะห์ Confusion

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# สร้าง confusion matrix
cm = confusion_matrix(df['gt'], df['prediction'])
diseases = sorted(df['gt'].unique())

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=diseases, yticklabels=diseases)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - GPT-4o')
plt.tight_layout()
plt.savefig('confusion_matrix_gpt4o.png')
```

### 7.5 ผลกระทบของขนาดภาพ

เปรียบเทียบผลลัพธ์ระหว่าง 224×224 และ 224×224:

```python
# โหลดทั้งสองขนาด
df_224 = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')
df_224 = pd.read_csv('chatbot_prediction/gpt-4o.csv')

# คำนวณ accuracies
acc_224 = accuracy_score(df_224['gt'], df_224['prediction'])
acc_224 = accuracy_score(df_224['gt'], df_224['prediction'])

print(f"224×224 Accuracy: {acc_224:.4f}")
print(f"224×224 Accuracy: {acc_224:.4f}")
print(f"การปรับปรุง: {(acc_224 - acc_224):.4f}")
```

---

## 8. การตั้งค่า API

### 8.1 OpenAI (GPT-4o, GPT-4.1, GPT-5)

```python
# GPT-4.1 (Azure)
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_FOUNDRY"),
    azure_endpoint=os.getenv("AZURE_API_FOUNDRY_ENDPOINT"),
    api_version="2024-12-01-preview"
)

# GPT-4o, GPT-5 (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### 8.2 Claude (Anthropic)

```python
import anthropic
client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)
```

### 8.3 Gemini (Google)

```python
from google import genai
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
```

### 8.4 Qwen (Alibaba Cloud)

```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

---

## 9. ประมาณการต้นทุน

### 9.1 ต้นทุน API (โดยประมาณ)

| โมเดล | ต้นทุนต่อ 1K ภาพ (224px) | ต้นทุนต่อ 1K ภาพ (224px) |
|-------|--------------------------|--------------------------|
| GPT-4o | $2-5 | $5-10 |
| GPT-4.1-mini | $1-2 | $2-4 |
| GPT-5 | $5-10 | $10-20 |
| Claude Sonnet 4 | $3-6 | $6-12 |
| Gemini 2.5 Flash | $0.50-1 | $1-2 |
| Gemini 2.5 Pro | $2-4 | $4-8 |
| Qwen3-VL | $1-3 | $2-6 |

*หมายเหตุ: ต้นทุนเป็นการประมาณและแตกต่างกันตามภูมิภาคและระดับการใช้งาน*

---

## 10. สรุป

framework การประเมินนี้ให้การเปรียบเทียบโมเดล vision-language ที่ทันสมัยที่สุดสำหรับการจำแนกโรคพืช ผลลัพธ์สามารถช่วย:

- **การเลือกโมเดล**: เลือกโมเดลที่เหมาะสมสำหรับ production
- **การเพิ่มประสิทธิภาพต้นทุน**: สมดุลระหว่าง accuracy กับต้นทุน API
- **ระบบ Hybrid**: รวม CNN แบบดั้งเดิมกับ LLMs เพื่อผลลัพธ์ที่ดีที่สุด
- **การวิจัยในอนาคต**: ระบุพื้นที่สำหรับการปรับปรุง

---

**Repository:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/  
**ติดต่อ:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/issues

**อัปเดตล่าสุด:** 9 ธันวาคม 2568
