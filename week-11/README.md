# Week 11 - HuggingFace Advanced

> 🎯 Master advanced HuggingFace techniques including fine-tuning, PEFT/LoRA, quantization, embeddings, and multimodal models

## References

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [BitsAndBytes Quantization](https://huggingface.co/docs/transformers/main_classes/quantization)
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace Hub](https://huggingface.co/docs/hub)
- [Accelerate Library](https://huggingface.co/docs/accelerate)
- [HuggingFace Course](https://huggingface.co/course)

---

## Installation

```bash
# Navigate to project directory
cd week-11

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install core dependencies
uv add transformers torch accelerate datasets

# Install PEFT for efficient fine-tuning
uv add peft

# Install quantization support
uv add bitsandbytes

# Install sentence transformers
uv add sentence-transformers

# Install evaluation
uv add evaluate scikit-learn

# Install for vision tasks
uv add pillow timm

# Install for audio tasks
uv add torchaudio librosa

# Optional: Install HuggingFace Hub CLI
uv add huggingface_hub
```

---

## Concepts

### 1. Model Architecture Deep Dive

Understanding the transformer architecture is crucial for advanced usage.

#### Attention Mechanism

```python
"""
The attention mechanism allows models to focus on relevant parts of the input.
"""
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# Get attention weights
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len)
attention_weights = outputs.attentions

print(f"Number of layers: {len(attention_weights)}")
print(f"Attention shape: {attention_weights[0].shape}")

# Visualize attention for specific layer and head
layer_idx, head_idx = 11, 0
attention_matrix = attention_weights[layer_idx][0, head_idx].detach()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for i, token in enumerate(tokens):
    print(f"{token}: {attention_matrix[i].tolist()[:5]}...")
```

#### Hidden States Access

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

# Get hidden states from all layers
hidden_states = outputs.hidden_states  # 13 tensors for BERT-base (embeddings + 12 layers)

# Embedding layer output
embedding_output = hidden_states[0]
print(f"Embedding shape: {embedding_output.shape}")

# Last layer output (same as outputs.last_hidden_state)
last_layer = hidden_states[-1]
print(f"Last layer shape: {last_layer.shape}")

# Get CLS token representation
cls_embedding = last_layer[:, 0, :]  # [batch, hidden_dim]
print(f"CLS embedding shape: {cls_embedding.shape}")
```

---

### 2. Embeddings and Sentence Transformers

Embeddings convert text to dense vectors for similarity search and retrieval.

#### Basic Embeddings with Mean Pooling

```python
import torch
from transformers import AutoModel, AutoTokenizer

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = [
    "Machine learning is fascinating",
    "I love artificial intelligence",
    "The weather is nice today"
]

# Encode sentences
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])

# Cosine similarity
from torch.nn.functional import cosine_similarity

sim_01 = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
sim_02 = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))

print(f"Similarity (ML vs AI): {sim_01.item():.4f}")  # Higher
print(f"Similarity (ML vs Weather): {sim_02.item():.4f}")  # Lower
```

#### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Machine learning is fascinating",
    "I love artificial intelligence",
    "The weather is nice today"
]

# Get embeddings
embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Calculate similarity
from sentence_transformers import util

similarities = util.cos_sim(embeddings, embeddings)
print(f"Similarity matrix:\n{similarities}")

# Semantic search
query = "AI and deep learning"
query_embedding = model.encode(query)

corpus_embeddings = model.encode([
    "Neural networks are the foundation of deep learning",
    "The stock market crashed yesterday",
    "Transformers revolutionized NLP"
])

hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)
for hit in hits[0]:
    print(f"Score: {hit['score']:.4f}")
```

---

### 3. Fine-Tuning Fundamentals

Fine-tuning adapts pre-trained models to specific tasks.

#### Traditional Full Fine-Tuning

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("imdb", split="train[:1000]")
test_dataset = load_dataset("imdb", split="test[:200]")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = dataset.map(tokenize_function, batched=True)
eval_dataset = test_dataset.map(tokenize_function, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()
```

---

### 4. PEFT - Parameter-Efficient Fine-Tuning

PEFT methods train only a small subset of parameters, reducing compute requirements.

#### LoRA (Low-Rank Adaptation)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank of update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # Which modules to adapt
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# trainable params: 296,450 || all params: 67,251,458 || trainable%: 0.44

# Training is same as regular fine-tuning
# ... use Trainer as before ...

# Save and load PEFT model
peft_model.save_pretrained("./lora-model")

# Load later
from peft import PeftModel
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
loaded_model = PeftModel.from_pretrained(base_model, "./lora-model")
```

#### QLoRA (Quantized LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

---

### 5. Quantization

Quantization reduces model size and inference time.

#### Dynamic Quantization (CPU)

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Compare sizes
import os

torch.save(model.state_dict(), "original.pt")
torch.save(quantized_model.state_dict(), "quantized.pt")

original_size = os.path.getsize("original.pt") / 1e6
quantized_size = os.path.getsize("quantized.pt") / 1e6

print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")

# Cleanup
os.remove("original.pt")
os.remove("quantized.pt")
```

#### BitsAndBytes Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit quantization (even smaller)
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config_4bit,
    device_map="auto"
)

# Memory comparison
print(f"8-bit memory: {model_8bit.get_memory_footprint() / 1e9:.2f} GB")
print(f"4-bit memory: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
```

---

### 6. Vision Transformers

HuggingFace supports various vision models.

#### Image Classification

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch

# Load model
model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process and predict
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Get prediction
logits = outputs.logits
predicted_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_idx]
confidence = torch.softmax(logits, dim=-1)[0, predicted_idx].item()

print(f"Predicted: {predicted_label} (confidence: {confidence:.2%})")
```

#### Image Feature Extraction

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load images
images = [Image.new("RGB", (224, 224), color) for color in ["red", "blue", "red"]]

# Extract features
inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]  # CLS token

# Compare similarity
from torch.nn.functional import cosine_similarity

sim_01 = cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
sim_02 = cosine_similarity(features[0].unsqueeze(0), features[2].unsqueeze(0))

print(f"Red vs Blue: {sim_01.item():.4f}")
print(f"Red vs Red: {sim_02.item():.4f}")
```

---

### 7. Audio Models

HuggingFace supports speech recognition and audio classification.

#### Speech Recognition (Whisper)

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load audio file
audio_path = "path/to/audio.wav"
audio, sr = librosa.load(audio_path, sr=16000)

# Process
input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(f"Transcription: {transcription[0]}")
```

#### Audio Classification

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa

# Load model
model_name = "superb/wav2vec2-base-superb-ks"  # Keyword spotting
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Load and process audio
audio, sr = librosa.load("audio.wav", sr=16000)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

# Classify
with torch.no_grad():
    outputs = model(**inputs)
    predicted_idx = outputs.logits.argmax(-1).item()

label = model.config.id2label[predicted_idx]
print(f"Predicted keyword: {label}")
```

---

### 8. Model Hub and Sharing

Working with the HuggingFace Hub.

#### Uploading Models

```python
from huggingface_hub import HfApi, login

# Login (run once)
login()  # Will prompt for token

# Push model to Hub
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")

# Or push with specific commit message
model.push_to_hub(
    "your-username/model-name",
    commit_message="Add fine-tuned model"
)
```

#### Creating Model Cards

```python
from huggingface_hub import ModelCard, ModelCardData

card_data = ModelCardData(
    language="en",
    license="mit",
    library_name="transformers",
    tags=["text-classification", "sentiment-analysis"],
    datasets=["imdb"],
    metrics=["accuracy"],
    base_model="distilbert-base-uncased"
)

card = ModelCard.from_template(
    card_data,
    model_id="your-username/model-name",
    model_description="A fine-tuned DistilBERT for sentiment analysis",
    developers="Your Name",
    training_details="Fine-tuned on IMDB dataset for 3 epochs"
)

card.push_to_hub("your-username/model-name")
```

---

### 9. Accelerate for Distributed Training

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# Initialize accelerator
accelerator = Accelerator()

# Load model and data
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = load_dataset("imdb", split="train[:1000]")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

dataloader = DataLoader(tokenized, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Prepare for distributed training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop
model.train()
for epoch in range(3):
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed")

# Save model (only on main process)
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), "model.pt")
```

---

### 10. Advanced Pipeline Usage

#### Custom Pipeline

```python
from transformers import Pipeline, AutoTokenizer, AutoModel
from transformers.pipelines import PIPELINE_REGISTRY
import torch

class SentenceEmbeddingPipeline(Pipeline):
    """Custom pipeline for sentence embeddings."""

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs.last_hidden_state

    def postprocess(self, model_outputs):
        # Mean pooling
        return model_outputs.mean(dim=1).squeeze().tolist()


# Register custom pipeline
PIPELINE_REGISTRY.register_pipeline(
    "sentence-embedding",
    pipeline_class=SentenceEmbeddingPipeline,
    pt_model=AutoModel
)

# Use custom pipeline
from transformers import pipeline

embedder = pipeline(
    "sentence-embedding",
    model="bert-base-uncased"
)

embedding = embedder("This is a test sentence")
print(f"Embedding dimension: {len(embedding)}")
```

---

## Exercises Structure

The exercises for this week follow the progressive difficulty pattern:

- **Exercise 1 (Basic)**: [exercises/exercise_basic_1_embeddings.py](exercises/exercise_basic_1_embeddings.py)
  - Mean pooling implementation
  - Sentence similarity
  - Semantic search
  - Attention analysis

- **Exercise 2 (Intermediate)**: [exercises/exercise_intermediate_2_finetuning.py](exercises/exercise_intermediate_2_finetuning.py)
  - Data preparation
  - Training loop
  - Evaluation metrics
  - Model checkpointing

- **Exercise 3 (Advanced)**: [exercises/exercise_advanced_3_peft.py](exercises/exercise_advanced_3_peft.py)
  - LoRA configuration
  - PEFT model creation
  - QLoRA setup
  - Model merging

---

## Weekly Project

Build a **Multi-Modal Similarity Search System** that:

1. Processes text and images
2. Creates embeddings for both modalities
3. Supports cross-modal search
4. Uses quantized models for efficiency
5. Includes a custom pipeline

Project file: [exercises/project_pipeline.py](exercises/project_pipeline.py)

---

## Interview Questions

1. **Explain the difference between mean pooling, CLS token, and max pooling for sentence embeddings. When would you use each?**

2. **What is LoRA and why is it more efficient than full fine-tuning? Explain the rank decomposition concept.**

3. **Describe the trade-offs between 4-bit and 8-bit quantization. When would you choose each?**

4. **How do attention weights help interpret model decisions? What are the limitations of attention-based interpretability?**

5. **Explain the concept of "catastrophic forgetting" in fine-tuning and how PEFT methods help mitigate it.**

6. **What is the difference between static and dynamic quantization? When would you use each?**

7. **How does the BitsAndBytes library achieve 4-bit quantization? Explain NF4 (Normal Float 4).**

8. **Compare Sentence Transformers with regular BERT embeddings. What makes Sentence Transformers better for semantic similarity?**

9. **What is the purpose of `prepare_model_for_kbit_training()` in QLoRA? What does it modify?**

10. **Explain how Vision Transformers (ViT) differ from traditional CNNs. What are patch embeddings?**

---

## Takeaway Checklist

After completing Week 11, you should be able to:

- [ ] Extract embeddings using mean pooling and CLS tokens
- [ ] Build semantic search systems with Sentence Transformers
- [ ] Fine-tune models using the Trainer API
- [ ] Implement LoRA adapters for efficient fine-tuning
- [ ] Set up QLoRA for large model fine-tuning
- [ ] Apply quantization to reduce model size
- [ ] Work with Vision Transformers for image tasks
- [ ] Use audio models for speech recognition
- [ ] Share models on HuggingFace Hub
- [ ] Create custom pipelines

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 12](../week-12/README.md)**
