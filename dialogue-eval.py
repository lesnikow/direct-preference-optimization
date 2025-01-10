"""
Evaluate dialogue model responses using GPT-4 as a judge, following the methodology from
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

import json
import openai
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from make-responses import generate_responses


class DialogueEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.prompt_template = (
            "For the following query to a chatbot, which response is more helpful?\n\n"
            "Query: {query}\n\n"
            "Response A:\n{response_a}\n\n"
            "Response B:\n{response_b}\n\n"
            "FIRST provide a one-sentence comparison of the two responses and explain "
            'which you feel is more helpful. SECOND, on a new line, state only "A" '
            'or "B" to indicate which response is more helpful. Your response should '
            "use the format:\n\n"
            "Comparison: <one-sentence comparison and explanation>\n"
            'More helpful: <"A" or "B">'
        )

    def get_gpt4_judgment(
        self, query: str, response_a: str, response_b: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get GPT-4's judgment between two responses."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt_template.format(
                            query=query, response_a=response_a, response_b=response_b
                        ),
                    }
                ],
                temperature=0,
            )
            judgment = response.choices[0].message.content

            lines = judgment.strip().split("\n")
            comparison = lines[0].replace("Comparison: ", "")
            choice = lines[1].replace("More helpful: ", "").strip()

            return comparison, choice
        except Exception as e:
            print(f"Error getting judgment: {e}")
            return None, None

    def evaluate_model(
        self,
        test_data: List[Dict],
        model_path: str,
        baseline_path: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
        temperature: float = 0.7,
        device: str = "cuda",
        sample_size: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate responses from specified model against baseline.
        If baseline_path is None, uses ground truth responses from test_data.
        """
        # Get model responses
        model_responses = generate_responses(
            model_path=model_path,
            test_data=test_data,
            batch_size=batch_size,
            max_length=max_length,
            temperature=temperature,
            device=device,
        )

        # Get baseline responses
        if baseline_path:
            baseline_responses = generate_responses(
                model_path=baseline_path,
                test_data=test_data,
                batch_size=batch_size,
                max_length=max_length,
                temperature=temperature,
                device=device,
            )
        else:
            baseline_responses = [d["chosen"] for d in test_data]

        # Subsample if requested
        if sample_size and sample_size < len(test_data):
            indices = np.random.choice(len(test_data), sample_size, replace=False)
            test_data = [test_data[i] for i in indices]
            model_responses = [model_responses[i] for i in indices]
            baseline_responses = [baseline_responses[i] for i in indices]

        wins = 0
        comparisons = []

        for i, (data, model_resp, baseline_resp) in enumerate(
            tqdm(zip(test_data, model_responses, baseline_responses))
        ):
            # Randomly assign model response to A or B
            if np.random.random() < 0.5:
                resp_a, resp_b = model_resp, baseline_resp
                model_is_a = True
            else:
                resp_a, resp_b = baseline_resp, model_resp
                model_is_a = False

            comparison, choice = self.get_gpt4_judgment(data["prompt"], resp_a, resp_b)

            if choice in ["A", "B"]:
                model_won = (model_is_a and choice == "A") or (
                    not model_is_a and choice == "B"
                )
                wins += int(model_won)

                comparisons.append(
                    {
                        "prompt": data["prompt"],
                        "model_response": model_resp,
                        "baseline_response": baseline_resp,
                        "comparison": comparison,
                        "model_won": model_won,
                    }
                )

        results = {
            "win_rate": wins / len(test_data),
            "total_comparisons": len(test_data),
            "comparisons": comparisons,
        }

        return results


def load_anthropic_hh_data(path: str) -> List[Dict]:
    """Load the Anthropic HH dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--baseline_path")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--openai_key", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_size", type=int)
    args = parser.parse_args()

    # Load test data
    test_data = load_anthropic_hh_data(args.data_path)

    # Initialize evaluator
    evaluator = DialogueEvaluator(args.openai_key)

    # Run evaluation
    results = evaluator.evaluate_model(
        test_data=test_data,
        model_path=args.model_path,
        baseline_path=args.baseline_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device,
        sample_size=args.sample_size,
    )

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
