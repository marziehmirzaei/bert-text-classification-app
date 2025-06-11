# BERT Text Classification App

A text classification application using a fine-tuned BERT model.

## 🚀 Quick Start

### Use the model directly (easiest):
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="MARAZI/bert-text-classifier")
result = classifier("Your text here")
print(result)
