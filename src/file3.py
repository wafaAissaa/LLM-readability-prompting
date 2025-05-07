from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessor
import torch

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Custom logits processor to add external scores
class AdditiveLogitsProcessor(LogitsProcessor):
    def __init__(self, external_scores: torch.Tensor):
        self.external_scores = external_scores  # shape: (vocab_size,)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (batch_size, vocab_size)
        # external_scores: (vocab_size,)
        return scores + self.external_scores  # elementwise addition

# Create external score tensor (e.g., bias toward A2 tokens)
a2_vocab = ["house", "dog", "cat", "is", "the", "a", "big", "small", "blue", "happy", ".", "beautiful", "green", "and"]
external_scores = torch.full((len(tokenizer),), fill_value=-10.0)  # Penalize all tokens

for word in a2_vocab:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    for t in tokens:
        external_scores[t] = 5.0  # Boost A2 vocab tokens

# Move to model device
external_scores = external_scores.to(model.device)

# Wrap in LogitsProcessorList
from transformers import LogitsProcessorList
logits_processor = LogitsProcessorList([
    AdditiveLogitsProcessor(external_scores)
])

# Input
input_text = "translate English to French: The house is green and beautiful."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# Generate
outputs = model.generate(
    input_ids=input_ids,
    max_length=50,
    num_beams=5,
    logits_processor=logits_processor,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))