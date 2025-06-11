from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = "my_trained_bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Labels – update to your actual labels
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", result="Please enter text.")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()

    label = label_map.get(pred_class, "Unknown")
    return render_template("index.html", result=label)

if __name__ == "__main__":
    app.run(debug=True)
