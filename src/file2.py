from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessorList, MinLengthLogitsProcessor, BeamScorer
from transformers.generation.beam_search import BeamSearchScorer
import torch

class ConstrainedBeamSearchScorer(BeamScorer):
    def __init__(self, batch_size, num_beams, device):
        self.beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device
        )

    def process(self, input_ids, next_scores, next_tokens, next_indices, pad_token_id=None, eos_token_id=None):
        return self.beam_scorer.process(
            input_ids, next_scores, next_tokens, next_indices,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

    def is_done(self, best_score, cur_len):
        return self.beam_scorer.is_done(best_score, cur_len)

    def finalize(self, input_ids, final_beam_scores, final_beam_tokens, final_beam_indices, pad_token_id=None, eos_token_id=None):
        return self.beam_scorer.finalize(
            input_ids, final_beam_scores, final_beam_tokens, final_beam_indices,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Encode input
input_text = "translate English to French: The house is wonderful."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)

# Set parameters
num_beams = 5
beam_scorer = ConstrainedBeamSearchScorer(batch_size=1, num_beams=num_beams, device=model.device)

# Optional: apply constraints (e.g., minimum length)
logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(10, eos_token_id=tokenizer.eos_token_id)])

# Generate with custom beam search scorer
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        beam_scorer=beam_scorer,
        logits_processor=logits_processor,
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
