# LLM Evaluation for Corn Disease Classification

**Project:** Intelligent AgriDiagnose  
**Repository:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/  
**Last Updated:** December 9, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluated Models](#evaluated-models)
3. [Evaluation Script](#evaluation-script)
4. [Results Summary](#results-summary)
5. [How to Run Evaluation](#how-to-run-evaluation)
6. [Prompts](#prompts)
7. [Analysis](#analysis)

---

## 1. Overview

This document describes the evaluation of various Large Language Models (LLMs) for corn leaf disease classification using vision capabilities. The evaluation compares:

- **Traditional CNN**: ResNet-18 trained on corn disease dataset
- **GPT Models**: GPT-4o, GPT-4.1-mini, GPT-5
- **Claude**: Claude Sonnet 4
- **Gemini**: Gemini 2.5 Flash, Gemini 2.5 Pro
- **Qwen**: Qwen3-VL-235B

### Image Sizes Tested
- **224×224 pixels**: Standard size

---

## 2. Evaluated Models

### 2.1 Traditional CNN

| Model | Architecture | Input Size | Training |
|-------|--------------|------------|----------|
| ResNet-18 | CNN with transfer learning | 224×224 | Fine-tuned on corn disease dataset |

### 2.2 OpenAI Models

| Model | Version | API | Input Size |
|-------|---------|-----|------------|
| GPT-4o | Latest | Azure OpenAI | 224×224, 224×224 |
| GPT-4.1-mini | Latest | Azure OpenAI | 224×224 |
| GPT-5 | Preview | OpenAI | 224×224, 224×224 |

### 2.3 Anthropic Models

| Model | Version | API | Input Size |
|-------|---------|-----|------------|
| Claude Sonnet 4 | 2025-05-14 | Anthropic | 224×224, 224×224 |

### 2.4 Google Models

| Model | Version | API | Input Size |
|-------|---------|-----|------------|
| Gemini 2.5 Flash | Latest | Google AI | 224×224 |
| Gemini 2.5 Pro | Latest | Google AI | 224×224 |

### 2.5 Qwen Models

| Model | Version | API | Input Size |
|-------|---------|-----|------------|
| Qwen3-VL-235B | a22b-instruct | Alibaba Cloud | 224×224, 224×224 |

---

## 3. Evaluation Script

### Core Files

| File | Purpose |
|------|---------|
| `agent_test.py` | Main evaluation script for all LLM models |
| `agent_testv2.py` | Alternative evaluation implementation |
| `prompts/system2.txt` | System prompt for LLMs |
| `prompts/user2.txt` | User prompt template |

### Script Structure

```python
# agent_test.py key components

# 1. Model Response Schema
class CornLeafDiseaseResponse(BaseModel):
    response: Literal[
        "rust", "blight", "spot", "virus", 
        "mildew", "healthy", "abnormality", 
        "brownspot", "curl", "smut"
    ]

# 2. Classification Functions
def classify_with_chatgpt(image_path, model_name, client) -> str
def classify_with_claude(image_path, model_name, client) -> str
def classify_with_gemini(image_path, model_name, client) -> str
def classify_with_qwen(image_path, model_name, client) -> str

# 3. Image Encoding
def encode_image(image_path: str) -> str:
    """Encode image to base64 for API submission"""
    
# 4. Main Evaluation Loop
for each test image:
    - Load image
    - Call appropriate LLM API
    - Parse response
    - Compare with ground truth
    - Save results to CSV
```

### Supported Disease Classes

```python
LABELS = [
    "blight",      # Northern Corn Leaf Blight
    "rust",        # Rust
    "virus",       # Virus infections
    "mildew",      # Downy Mildew
    "healthy",     # No disease
    "spot"         # Leaf Spot
]
```

---

## 4. Results Summary

### 4.1 Available Result Files

All results are stored in `chatbot_prediction/` directory:

```
chatbot_prediction/
├── resnet18_224.csv              # ResNet-18 baseline (224×224)
├── resnet18.csv                   # ResNet-18 baseline (224×224)
├── gpt-4o_224.csv                # GPT-4o results (224×224)
├── gpt-4o.csv                     # GPT-4o results (224×224)
├── gpt4_1_224.csv                # GPT-4.1-mini results (224×224)
├── gpt-5_224.csv                 # GPT-5 results (224×224)
├── gpt-5.csv                      # GPT-5 results (224×224)
├── claude-sonnet-4-20250514_224.csv  # Claude Sonnet 4 (224×224)
├── claude-sonnet-4-20250514.csv      # Claude Sonnet 4 (224×224)
├── gemini_2_5_flash_224.csv      # Gemini 2.5 Flash (224×224)
├── gemini_2_5_pro_224.csv        # Gemini 2.5 Pro (224×224)
├── qwen3-vl-235b-a22b-instruct_224.csv  # Qwen3-VL (224×224)
└── qwen3-vl-235b-a22b-instruct.csv      # Qwen3-VL (224×224)
```

### 4.2 CSV Format

Each CSV file contains:

```csv
image,gt,prediction,confidence
Blight_crop_0005.jpg,blight,blight,0.7785653
Healthy_crop_0013.jpg,healthy,healthy,0.9670238
Rust_crop_0008.jpg,rust,rust,0.8765339
```

**Columns:**
- `image`: Filename of test image
- `gt`: Ground truth label
- `prediction`: Model's predicted label
- `confidence`: Confidence score (0-1)

### 4.3 Preliminary Analysis

Based on the CSV files:

**ResNet-18 (224×224) Performance:**
- Strong baseline with high confidence scores
- Occasional misclassifications between similar diseases
- Example: Some blight images misclassified as healthy or rust

**General Observations:**
- Models generally perform well on healthy leaves
- Confusion between visually similar diseases (e.g., blight vs rust)
- Mildew sometimes confused with virus
- Confidence scores vary significantly across models

---

## 5. How to Run Evaluation

### 5.1 Prerequisites

```bash
# Install dependencies
pip install openai anthropic google-generativeai torch torchvision pillow tqdm python-dotenv pydantic

# Set up environment variables
export AZURE_API_FOUNDRY_ENDPOINT="your_azure_endpoint"
export AZURE_API_FOUNDRY="your_azure_api_key"
export GEMINI_API_KEY="your_gemini_key"
export CLAUDE_API_KEY="your_claude_key"
export QWEN_API_KEY="your_qwen_key"
```

### 5.2 Basic Usage

```bash
# Test with GPT-4o
python agent_test.py \
    --model "gpt-4o" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/gpt4o_results.csv" \
    --img_size 224

# Test with Claude Sonnet 4
python agent_test.py \
    --model "claude-sonnet-4-20250514" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/claude_results.csv" \
    --img_size 224

# Test with Gemini 2.5 Pro
python agent_test.py \
    --model "gemini-2.5-pro" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/gemini_results.csv" \
    --img_size 224

# Test with Qwen3-VL
python agent_test.py \
    --model "qwen3-vl-235b-a22b-instruct" \
    --test_dir "./All_Crops/test" \
    --output_csv "./results/qwen_results.csv" \
    --img_size 224
```

### 5.3 Command-Line Arguments

```python
parser.add_argument("--model", type=str, required=True,
                   help="Model name: gpt-4o, gpt-5, claude-sonnet-4, gemini-2.5-pro, etc.")
parser.add_argument("--test_dir", type=str, required=True,
                   help="Directory containing test images")
parser.add_argument("--output_csv", type=str, default="results.csv",
                   help="Output CSV file path")
parser.add_argument("--img_size", type=int, default=224,
                   help="Image size (224 or 224)")
parser.add_argument("--selected_classes", type=str,
                   default="rust,blight,spot,virus,mildew,healthy",
                   help="Comma-separated list of classes to evaluate")
```

### 5.4 Batch Processing

```bash
# Run evaluation for all models
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

Located in `prompts/system2.txt`:

```
You are an expert agricultural AI assistant specializing in corn leaf disease diagnosis. 
Your task is to analyze images of corn leaves and identify diseases with high accuracy.

Available disease classes:
- rust: Orange/brown pustules on leaves
- blight: Northern Corn Leaf Blight with elongated lesions
- spot: Leaf spot diseases with circular spots
- virus: Various viral infections causing mottling or streaking
- mildew: Downy mildew with white/gray mold
- healthy: No visible disease symptoms

Provide only the disease name from the above list. Be precise and confident in your diagnosis.
```

### 6.2 User Prompt

Located in `prompts/user2.txt`:

```
Analyze this corn leaf image and identify the disease present. 
Respond with ONLY one of these disease names: rust, blight, spot, virus, mildew, or healthy.

Disease name:
```

### 6.3 Structured Output

All models are configured to return structured JSON:

```json
{
  "response": "rust"
}
```

This ensures consistent parsing across different LLM providers.

---

## 7. Analysis

### 7.1 Comparison Metrics

To compute accuracy metrics from CSV files:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load results
df = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')

# Calculate accuracy
accuracy = accuracy_score(df['gt'], df['prediction'])
print(f"Accuracy: {accuracy:.4f}")

# Generate classification report
report = classification_report(df['gt'], df['prediction'])
print(report)
```

### 7.2 Confidence Analysis

```python
# Analyze confidence scores
df = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')

# Average confidence
avg_conf = df['confidence'].mean()
print(f"Average Confidence: {avg_conf:.4f}")

# Confidence by correctness
correct = df[df['gt'] == df['prediction']]['confidence'].mean()
incorrect = df[df['gt'] != df['prediction']]['confidence'].mean()
print(f"Confidence (Correct): {correct:.4f}")
print(f"Confidence (Incorrect): {incorrect:.4f}")
```

### 7.3 Per-Class Performance

```python
# Per-class accuracy
for disease in df['gt'].unique():
    disease_df = df[df['gt'] == disease]
    disease_acc = (disease_df['gt'] == disease_df['prediction']).mean()
    print(f"{disease}: {disease_acc:.4f}")
```

### 7.4 Confusion Analysis

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
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

### 7.5 Image Size Impact

Compare results between 224×224 and 224×224:

```python
# Load both sizes
df_224 = pd.read_csv('chatbot_prediction/gpt-4o_224.csv')
df_224 = pd.read_csv('chatbot_prediction/gpt-4o.csv')

# Calculate accuracies
acc_224 = accuracy_score(df_224['gt'], df_224['prediction'])
acc_224 = accuracy_score(df_224['gt'], df_224['prediction'])

print(f"224×224 Accuracy: {acc_224:.4f}")
print(f"224×224 Accuracy: {acc_224:.4f}")
print(f"Improvement: {(acc_224 - acc_224):.4f}")
```

---

## 8. API Configuration

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

## 9. Cost Estimation

### 9.1 API Costs (Approximate)

| Model | Cost per 1K images (224px) | Cost per 1K images (224px) |
|-------|---------------------------|---------------------------|
| GPT-4o | $2-5 | $5-10 |
| GPT-4.1-mini | $1-2 | $2-4 |
| GPT-5 | $5-10 | $10-20 |
| Claude Sonnet 4 | $3-6 | $6-12 |
| Gemini 2.5 Flash | $0.50-1 | $1-2 |
| Gemini 2.5 Pro | $2-4 | $4-8 |
| Qwen3-VL | $1-3 | $2-6 |

*Note: Costs are estimates and vary by region and usage tier.*

### 9.2 Test Dataset Size

```bash
# Count test images
find All_Crops/test -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
# ~200 images per class × 6 classes = ~1200 images
```

---

## 10. Error Handling

### 10.1 Common Errors

```python
# Rate limiting
try:
    response = client.chat.completions.create(...)
except Exception as e:
    if "rate_limit" in str(e).lower():
        time.sleep(60)  # Wait 1 minute
        response = client.chat.completions.create(...)

# Invalid image format
def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

# Response parsing
def parse_response(response_text: str) -> str:
    try:
        data = json.loads(response_text)
        return data.get("response", "unknown")
    except json.JSONDecodeError:
        # Fallback to text extraction
        return extract_disease_name(response_text)
```

---

## 11. Future Work

### 11.1 Potential Improvements

1. **Ensemble Methods**
   - Combine predictions from multiple models
   - Weighted voting based on confidence scores

2. **Few-Shot Learning**
   - Provide example images in prompts
   - Fine-tune prompts with challenging cases

3. **Chain-of-Thought Prompting**
   - Ask models to explain reasoning
   - Improve accuracy through step-by-step analysis

4. **Multi-Modal Fusion**
   - Combine LLM predictions with ResNet-18
   - Use LLM for uncertainty cases

5. **Active Learning**
   - Identify low-confidence predictions
   - Request human expert review

### 11.2 Additional Evaluations

- **Robustness Testing**: Image corruptions, occlusions
- **Adversarial Examples**: Test model reliability
- **Cross-Dataset Evaluation**: Generalization to other corn datasets
- **Real-Time Performance**: Latency measurements
- **Cost-Benefit Analysis**: Accuracy vs API costs

---

## 12. Conclusion

This evaluation framework provides a comprehensive comparison of state-of-the-art vision-language models for agricultural disease classification. The results can guide:

- **Model Selection**: Choose optimal model for production deployment
- **Cost Optimization**: Balance accuracy with API costs
- **Hybrid Systems**: Combine traditional CNN with LLMs for best results
- **Future Research**: Identify areas for improvement

---

**Repository:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/  
**Contact:** https://github.com/panthakan-fusionsol/Intelligent-AgriDiagnose/issues

**Last Updated:** December 9, 2025
