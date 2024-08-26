from transformers import AutoTokenizer, DistilBertForSequenceClassification
import os

class ModelConfig:
    def __init__(self, checkpoint='distilbert-base-cased'):
        self.checkpoint = checkpoint
        self.tokenizer = None
        self.model = None
    
    def load_model_and_tokenizer(self):
        # Construct the absolute path to the models directory
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype="auto",
            use_safetensors=True  
        )

    def get_tokenizer(self):
        if not self.tokenizer:
            self.load_model_and_tokenizer()
        return self.tokenizer

    def get_model(self):
        if not self.model:
            self.load_model_and_tokenizer()
        return self.model

# Initialize the model configuration with the checkpoint
model_config = ModelConfig()

