from app.config import model_config
import torch

def tokenize_fn(batch):
    tokenizer = model_config.get_tokenizer()
    return tokenizer(batch['sentence'], truncation=True)

def predict_sentiment(text: str):
    tokenizer = model_config.get_tokenizer()
    model = model_config.get_model()

    # Tokenize the input text using the tokenize_fn
    inputs = tokenize_fn({"sentence": text})

    # Convert to tensor format
    input_tensors = {key: torch.tensor(value).unsqueeze(0) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**input_tensors)

    # Get the predicted label and score
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    max_prob, prediction = torch.max(probabilities, dim=1)

    # Convert the prediction to a label using the model's label map
    id2label = model.config.id2label  # Assuming this is in the model's config
    predicted_label = id2label[prediction.item()]

    return {"label": predicted_label, "score": max_prob.item()}



