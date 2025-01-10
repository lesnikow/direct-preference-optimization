"""
Generate responses from a PyTorch language model for dialogue evaluation.

This module loads a saved PyTorch model and generates responses for a test dataset
of dialogue prompts. Supports batch processing and customizable generation parameters.

Usage:
    python make_responses.py \
        --model_path path/to/model \
        --data_path path/to/test_data.json \
        --output_path responses.json \
        [--batch_size 8] \
        [--max_length 512] \
        [--temperature 0.7] \
        [--device cuda]
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from tqdm import tqdm


class DialogueDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]

        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "prompt": prompt,
        }


def generate_responses(
    model_path: str,
    test_data: List[Dict],
    batch_size: int = 8,
    max_length: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
    **generation_kwargs
) -> List[str]:
    """Generate responses using a saved model."""

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Prepare dataset and dataloader
    dataset = DialogueDataset(test_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    responses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate responses
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs
            )

            # Decode and clean up responses
            for i, output in enumerate(outputs):
                # Find where the prompt ends
                prompt_length = len(
                    tokenizer.encode(batch["prompt"][i], add_special_tokens=True)
                )

                # Only take the generated part
                response = tokenizer.decode(
                    output[prompt_length:], skip_special_tokens=True
                ).strip()

                responses.append(response)

    return responses


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load test data
    with open(args.data_path) as f:
        test_data = json.load(f)

    # Generate responses
    responses = generate_responses(
        model_path=args.model_path,
        test_data=test_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device,
    )

    # Save responses
    with open(args.output_path, "w") as f:
        json.dump(responses, f, indent=2)


if __name__ == "__main__":
    main()
