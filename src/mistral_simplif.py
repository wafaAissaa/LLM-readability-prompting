from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from utils import ft_download_data, download_data ,get_balanced_dataframe, download_difficulty_estimation
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import os
from tqdm import tqdm as console_tqdm
from accelerate import dispatch_model

# ------------------------------ CREATE DATASET ------------------------------ #
from datasets import Dataset


def format_data(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    training: bool = True,
):
    # Create conversation
    print("Create conversation...")

    def create_conversation(row):
        conversation = [
            {
                "role": "system",
                "content": "Vous êtes un modèle de langage naturel capable de simplifier des phrases en français. La phrase simplifiée doit avoir un sens aussi proche que possible de la phrase originale, mais elle est d'un niveau inférieur du CECRL et donc plus facile à comprendre. Par exemple, si une phrase est au niveau C1 du CECRL, simplifiez-la en B2. Si elle se situe au niveau B2, simplifiez-la en B1. Si elle se situe au niveau B1, simplifiez-la en A2. Si le niveau A2 est atteint, simplifiez en A1.",
            }
        ]
        if training:
            conversation.extend(
                [
                    {
                        "role": "user",
                        "content": f"""Voici une phrase en français de niveau CECRL {['A2', 'B1', 'B2', 'C1', 'C2'][row['index'] % 5]} à simplifier :
                    \"\"\"{row['Original sentence']}\"\"\"
                    Donne moi une phrase simplifiée au niveau CECRL {['A1', 'A2', 'B1', 'B2', 'C1'][row['index'] % 5]} tout en conservant au maximum son sens original
                    """,
                    },
                    {
                        "role": "assistant",
                        "content": f"{row['Simplified sentence']}",
                    },
                ]
            )
        else:
            reduced_difficulty = {
                "A1": "A1",
                "A2": "A1",
                "B1": "A2",
                "B2": "B1",
                "C1": "B2",
                "C2": "C1",
                "level1": "level1",
                "level2": "level1",
                "level3": "level2",
                "level4": "level3",
            }
            conversation.append(
                {
                    "role": "user",
                    "content": f"""Voici une phrase en français de niveau {row['Difficulty']} à simplifier :
                    \"\"\"{row['Sentence']}\"\"\"
                    Donne moi une phrase simplifiée au niveau {reduced_difficulty[row['Difficulty']]} tout en conservant au maximum son sens original
                    """,
                }
            )

        return conversation

    # Create dataset
    print("Create dataset...")
    conversation_list = (
        df.reset_index()
        .apply(create_conversation, axis=1)
        .rename("conversation")
        .to_list()
    )
    dataset = Dataset.from_dict({"chat": conversation_list})

    # Format dataset
    print("Format dataset...")
    formatted_dataset = dataset.map(
        lambda x: {
            "formatted_chat": tokenizer.apply_chat_template(
                x["chat"], tokenize=False, add_generation_prompt=True
            )
        }
    )

    return formatted_dataset

# ------------------------------ ENCODE DATASET ------------------------------ #
import torch
from tqdm import notebook as notebook_tqdm


def encode_dataset(dataset: Dataset, tokenizer: AutoTokenizer):
    # Determine max length
    print("Determine max length...")
    max_length = max(
        [
            len(tokenizer.encode(chat))
            for chat in notebook_tqdm.tqdm(dataset["formatted_chat"])
        ]
    )

    # Encode dataset
    print("Encode dataset...")
    encoded_dataset = dataset.map(
        lambda x: tokenizer(
            x["formatted_chat"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        ),
        batched=True,
    )

    # Create labels
    print("Create labels...")
    encoded_dataset = encoded_dataset.map(
        lambda x: {
            "labels": x["input_ids"],
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
        },
        batched=True,
    )

    # Create dataset ready for training
    print("Create dataset ready for training...")
    encoded_dataset = Dataset.from_dict(
        {
            "input_ids": torch.tensor(encoded_dataset["input_ids"]),
            "attention_mask": torch.tensor(encoded_dataset["attention_mask"]),
            "labels": torch.tensor(encoded_dataset["labels"]),
        }
    )

    # Set format
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return encoded_dataset


def download_tokenizer(model_name: str , training: bool = True):
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        truncation_side="left",
        add_eos_token=training,
        add_bos_token=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_name: str):
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", use_cache=False
        )
    except:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="cuda", use_cache=False
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="cpu", use_cache=False
            )

    # Configure model
    config = LoraConfig(
        r=64,  # Plus r est grand, plus le modèle est précis mais plus il est lent
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.config.use_cache = False

    return model



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import dispatch_model

    # Load the model and tokenizer from Hugging Face Hub
    model_name = "OloriBern/Mistral-7B-French-Simplification"
    offload_dir = "./offload"  # Directory for offloading model parts to disk

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, trust_remote_code=True)

    # Offload model parts using Accelerate
    model = dispatch_model(model, device_map="auto", offload_dir=offload_dir)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Now you can use the model and tokenizer for inference
    input_text = "Exemple de texte pour la simplification."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate simplified text
    outputs = model.generate(**inputs)
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(simplified_text)

    '''pwd = '..'
    MODEL = "bofenghuang/vigostral-7b-chat"
    zero_shot = False

    # Load data
    df = download_difficulty_estimation(pwd)
    test_df = get_balanced_dataframe(df, nbr=100)

    # Charger tokenizer
    tokenizer = download_tokenizer(MODEL, training=False)

    # Create dataset
    dataset = format_data(test_df, tokenizer, training=False)

    # Encode dataset
    encoded_dataset = encode_dataset(dataset, tokenizer)

    """# Load model
    path = os.path.join(
        pwd,
        "models",
        "difficulty_estimation",
        MODEL.replace("/", "_"),
    )"""

    path = "mistral_simplification_trained"

    # Clone model checkpoint
    print("model exists %s" % os.path.exists(os.path.join(pwd, path)))
    if not os.path.exists(os.path.join(pwd, path)):
        snapshot_download(
            repo_id="OloriBern/Mistral-7B-French-Simplification",
            local_dir=os.path.join(pwd, "mistral_simplification_trained"),
            revision="main",
            repo_type="model",
        )

    if zero_shot:
        model = load_model(MODEL)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(pwd, "mistral_simplification_trained"),
            device_map="auto",
            offload_dir="./offload",
            use_cache=False,
            trust_remote_code=True,
        )
        # Load normally first
        base_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(pwd, "mistral_simplification_trained"),
            trust_remote_code=True,
            use_cache=False,
        )

        # Then dispatch
        base_model = dispatch_model(
            base_model,
            device_map="auto",
            offload_dir="./offload"
        )
        print(base_model)
        model = PeftModel.from_pretrained(
            base_model, os.path.join(pwd, "mistral_simplification_trained")
        )

    # Move everything to GPU
    model.to("cuda")
    test_loader = DataLoader(encoded_dataset, batch_size=16)

    # Generate predictions
    with torch.no_grad():
        model.eval()
        predictions_ids = []

        for batch in console_tqdm(test_loader):
            input_ids_batch = batch["input_ids"].to("cuda")
            attention_mask_batch = batch["attention_mask"].to("cuda")

            outputs = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_length=max(128, input_ids_batch.shape[1] * 2),
                num_return_sequences=1,
            )

            predictions_ids.extend(outputs)
        predictions = [
            tokenizer.decode(prediction, skip_special_tokens=True)
            for prediction in predictions_ids
        ]
        predictions_series = pd.Series(predictions)'''