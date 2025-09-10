from transformers import AutoTokenizer, CamembertForSequenceClassification
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import dill


# Charger le modèle et le tokenizer depuis Hugging Face
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Définir le prompt
prompt = "Once upon a time"

# Tokenizer le prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Générer du texte
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)

# Décoder et afficher le texte généré
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
