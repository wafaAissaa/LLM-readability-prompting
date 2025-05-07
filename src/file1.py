from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to French: How old are you?"
force_words = ["Madame"]

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids


outputs = model.generate(
    input_ids,
    force_words_ids=force_words_ids,
    num_beams=10,
    num_return_sequences=3,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
    decoded = tokenizer.decode(output, skip_special_tokens=False)
    print(f"[{i+1}] {decoded}")