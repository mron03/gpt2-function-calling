import argparse
import json

from gpt2fc.config import MODEL_CONFIGS, get_device
from gpt2fc.inference.generate import get_tokenizer, load_finetuned_model, run_inference
from gpt2fc.inference.parser import extract_functioncall
from gpt2fc.training.data import format_entry, load_glaive, split_glaive


def compute_metrics(results):
    """Metrics over samples whose ground truth contains a function call.

    All accuracies use the full function-call sample count as denominator,
    so unparseable predictions count as errors.
    """
    fc_results = [r for r in results if r["gt_fc"] is not None]
    total = len(results)
    fc_count = len(fc_results)

    if fc_count == 0:
        return {
            "total_samples": total,
            "fc_samples": 0,
            "non_fc_samples": total,
            "parse_rate": None,
            "fn_name_acc": None,
            "args_key_acc": None,
            "exact_match": None,
        }

    parsed = [r for r in fc_results if r["pred_fc"] is not None]

    def keys_match(r):
        pred_args = r["pred_fc"].get("arguments", {})
        gt_args = r["gt_fc"].get("arguments", {})
        if not isinstance(pred_args, dict) or not isinstance(gt_args, dict):
            return False
        return set(pred_args.keys()) == set(gt_args.keys())

    parse_rate = len(parsed) / fc_count
    fn_name_acc = sum(1 for r in parsed if r["pred_fc"].get("name") == r["gt_fc"].get("name")) / fc_count
    args_key_acc = sum(1 for r in parsed if keys_match(r)) / fc_count
    exact_match = sum(1 for r in parsed if r["pred_fc"] == r["gt_fc"]) / fc_count

    return {
        "total_samples": total,
        "fc_samples": fc_count,
        "non_fc_samples": total - fc_count,
        "parse_rate": round(parse_rate, 4),
        "fn_name_acc": round(fn_name_acc, 4),
        "args_key_acc": round(args_key_acc, 4),
        "exact_match": round(exact_match, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GPT-2 on Glaive function calling")
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--model-size",     default="355M", choices=list(MODEL_CONFIGS))
    parser.add_argument("--num-samples",    type=int, default=200, help="Test samples to evaluate (-1 = full test split)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-json",    default="results/eval-results.json")
    parser.add_argument("--device",         default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    model = load_finetuned_model(args.checkpoint, args.model_size, device)
    print(f"Loaded {args.model_size} checkpoint from {args.checkpoint}")

    tokenizer = get_tokenizer()

    print("Loading Glaive dataset...")
    data = load_glaive()
    _, _, test_data = split_glaive(data)
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    print(f"Evaluating on {len(test_data)} test samples")

    results = []
    for i, entry in enumerate(test_data):
        instruction, response = format_entry(entry)
        if not response:
            continue
        prediction = run_inference(model, tokenizer, instruction, args.max_new_tokens, device)
        results.append({
            "prompt": instruction,
            "ground_truth": response,
            "prediction": prediction,
            "gt_fc": extract_functioncall(response),
            "pred_fc": extract_functioncall(prediction),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(test_data)}]")

    metrics = compute_metrics(results)

    print("\n=== Results ===")
    for k, v in metrics.items():
        pct = f"  ({v * 100:.1f}%)" if isinstance(v, float) else ""
        print(f"  {k:<20} {v}{pct}")

    with open(args.output_json, "w") as f:
        json.dump({"metrics": metrics, "samples": results}, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
