from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "./further_fine_tuned_model2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use the model for inference
model.eval()

# Example texts for prediction
texts = [
    "Very bad movie, good scenography but other things were terrible"
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    sentiment = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
    confidence = max(predictions[0]).item()
    
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print()

print("Model training, evaluation, and saving completed successfully.")