from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm.auto import tqdm


DEFAULT_LAYERS = list(range(15, 31))
DEFAULT_SAE_SNAPSHOT = "pts"
TEXT_COLUMN = "text"

ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:therapist|client|patient|counsell?or|psychologist|psychiatrist|"
    r"doctor|user|assistant|interviewer|interviewee|speaker\s*[a-z0-9]+)\s*:\s*",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "a", "all", "also", "am", "an", "and", "are", "as", "at", "be", "but",
    "by", "can", "could", "did", "do", "does", "done", "for", "from", "get",
    "go", "going", "got", "had", "has", "have", "he", "her", "here", "him",
    "his", "how", "i", "if", "in", "is", "it", "its", "just", "know", "me",
    "my", "no", "nope", "not", "of", "oh", "ok", "okay", "on", "one", "or",
    "our", "really", "she", "should", "so", "that", "the", "their", "them",
    "then", "there", "they", "think", "this", "to", "very", "want", "was", "we",
    "were", "what", "when", "where", "who", "why", "will", "with", "would",
    "yeah", "yes", "you", "your",
}
CONTRACTION_PARTS = {
    "'s", "'m", "'re", "'ve", "'d", "'ll", "'t", "n't", "s", "m", "re", "ve",
    "d", "ll", "t", "nt", "don", "doesn", "didn", "isn", "aren", "wasn",
    "weren", "couldn", "wouldn", "shouldn", "won", "cant", "cannot", "hasn",
    "haven", "hadn",
}
PUNCT_CHARS = set(".,;:!?-()[]{}\"'*/\\|_+=<>@#$%^&~")


@dataclass
class DatasetFeatureStats:
    total_tokens: int = 0
    activation_counts: Counter = field(default_factory=Counter)
    activation_value_sums: defaultdict = field(default_factory=lambda: defaultdict(float))
    activation_value_maxes: dict[int, float] = field(default_factory=dict)
    observed_token_hits: Counter = field(default_factory=Counter)
    noise_token_hits: Counter = field(default_factory=Counter)
    content_token_counts: defaultdict = field(default_factory=lambda: defaultdict(Counter))

    def update(
        self,
        token_ids: list[int],
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
        tokenizer,
    ) -> None:
        self.total_tokens += len(token_ids)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        for token, ids, values in zip(tokens, feature_ids.tolist(), feature_values.tolist()):
            normalized_token = normalize_token(token)
            noise = is_noise_token(normalized_token)
            for feature_id, activation_value in zip(ids, values):
                feature_id = int(feature_id)
                activation_value = float(activation_value)
                self.activation_counts[feature_id] += 1
                self.activation_value_sums[feature_id] += activation_value
                previous_max = self.activation_value_maxes.get(feature_id, -math.inf)
                self.activation_value_maxes[feature_id] = max(previous_max, activation_value)
                self.observed_token_hits[feature_id] += 1
                if noise:
                    self.noise_token_hits[feature_id] += 1
                elif normalized_token:
                    self.content_token_counts[feature_id][normalized_token] += 1

    def frequency_per_1k(self, feature_id: int) -> float:
        return self.activation_counts[feature_id] / max(self.total_tokens, 1) * 1000.0

    def noise_ratio(self, feature_id: int) -> float:
        return self.noise_token_hits[feature_id] / max(self.observed_token_hits[feature_id], 1)

    def top_content_tokens(self, feature_id: int) -> str:
        return "; ".join(
            f"{token}:{count}"
            for token, count in self.content_token_counts[feature_id].most_common(20)
        )


def clean_dialogue_turn(turn: object) -> str:
    text = "" if turn is None else str(turn).strip()
    return ROLE_PREFIX_RE.sub("", text, count=1).strip()


def clean_dialogue_text(text: object) -> str:
    lines = (clean_dialogue_turn(line) for line in str(text).splitlines())
    return "\n".join(line for line in lines if line)


def clean_control_text(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.replace("_comma_", ",").replace("_comma", ",")
    return clean_dialogue_turn(" ".join(text.split()))


def join_turns(turns: Iterable[object]) -> str:
    cleaned = (clean_dialogue_turn(turn) for turn in turns)
    return "\n".join(turn for turn in cleaned if turn)


def conversation_text_from_payload(payload: dict) -> str:
    turns = payload.get("full_conversation") or payload.get("conversation")
    if turns is None and payload.get("three_turn_sequences"):
        seen = []
        seen_set = set()
        for sequence in payload["three_turn_sequences"]:
            for turn in sequence:
                if turn not in seen_set:
                    seen.append(turn)
                    seen_set.add(turn)
        turns = seen

    if isinstance(turns, list):
        return join_turns(turns)
    if isinstance(turns, str):
        return clean_dialogue_text(turns)
    raise ValueError(
        "Conversation JSON must contain full_conversation, conversation, "
        "or three_turn_sequences"
    )


def find_conversation_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "conversations",
        data_dir / "download" / "conversations",
        data_dir / ".cache" / "huggingface" / "download" / "conversations",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find PTSD conversations under {data_dir}")


def load_ptsd_texts(data_dir: Path) -> list[str]:
    conversation_dir = find_conversation_dir(data_dir)
    paths = sorted(conversation_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No PTSD conversation JSON files found in {conversation_dir}")

    texts = []
    for path in tqdm(paths, desc="load PTSD conversations"):
        with path.open("r", encoding="utf-8") as file:
            text = conversation_text_from_payload(json.load(file))
        if text:
            texts.append(text)
    return texts


def load_daily_texts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DailyDialog file: {path}")

    texts = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            text = join_turns(clean_control_text(turn) for turn in line.split("__eou__"))
            if text:
                texts.append(text)
    return texts


def iter_empathetic_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if len(row) >= 8:
                yield dict(zip(header[:8], row[:8]))


def load_empathetic_texts(root: Path) -> list[str]:
    paths = [root / "train.csv", root / "valid.csv", root / "test.csv"]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing EmpatheticDialogues files: {missing}")

    texts = []
    for path in paths:
        grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for item in iter_empathetic_rows(path):
            grouped[str(item["conv_id"])].append(item)
        for items in grouped.values():
            items.sort(key=lambda item: int(item["utterance_idx"]))
            text = join_turns(clean_control_text(item["utterance"]) for item in items)
            if text:
                texts.append(text)
    return texts


def normalize_token(token: object) -> str:
    token = "" if token is None else str(token)
    return (
        token.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", " ")
        .replace("<0x0A>", " ")
        .strip()
        .lower()
    )


def is_noise_token(token: str) -> bool:
    if not token or token in STOPWORDS or token in CONTRACTION_PARTS:
        return True
    if token.isdigit():
        return True
    if len(token) == 1 and not token.isalpha():
        return True
    return all(character in PUNCT_CHARS for character in token)


def parse_layers(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = (int(part.strip()) for part in value.split("-", 1))
        return list(range(start, end + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def infer_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_path(value: str, project_dir: Path, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else project_dir / path


def resolve_sae_snapshot_dir(
    value: str,
    project_dir: Path,
    layers: Iterable[int],
) -> Path:
    if value:
        candidate = resolve_path(value, project_dir, Path(value))
        missing = [layer for layer in layers if not (candidate / f"layer{layer}.sae.pt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"SAE snapshot {candidate} is missing checkpoints for layers: {missing}"
            )
        return candidate

    snapshots_root = project_dir / "qwen3-8b-SAE" / "snapshots"
    preferred = snapshots_root / DEFAULT_SAE_SNAPSHOT
    candidates = [preferred]
    if snapshots_root.is_dir():
        candidates.extend(
            path for path in sorted(snapshots_root.iterdir())
            if path.is_dir() and path != preferred
        )

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if all((candidate / f"layer{layer}.sae.pt").is_file() for layer in layers):
            return candidate

    raise FileNotFoundError(
        "Could not find one SAE snapshot containing every requested layer. "
        f"Checked: {checked}. You can pass --sae-snapshot-dir explicitly."
    )


def tokenize_corpora(tokenizer, corpora: dict[str, list[str]]) -> dict[str, list[list[int]]]:
    tokenized = {}
    for dataset_name, texts in corpora.items():
        conversations = []
        for text in tqdm(texts, desc=f"tokenize {dataset_name}"):
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if token_ids:
                conversations.append(token_ids)
        tokenized[dataset_name] = conversations
        print(
            f"{dataset_name}: conversations={len(conversations)} "
            f"tokens={sum(map(len, conversations))}"
        )
    return tokenized


def make_windows(
    tokenized_corpora: dict[str, list[list[int]]],
    token_budget: int | None,
    window_length: int,
    seed: int,
) -> list[tuple[str, list[int]]]:
    if token_budget is None:
        token_budget = min(
            sum(map(len, conversations))
            for conversations in tokenized_corpora.values()
        )

    windows = []
    for dataset_name, source_conversations in tokenized_corpora.items():
        conversations = list(source_conversations)
        random.Random(seed).shuffle(conversations)
        remaining = token_budget
        selected_tokens = 0
        for conversation in conversations:
            if remaining <= 0:
                break
            selected = conversation[:remaining]
            remaining -= len(selected)
            selected_tokens += len(selected)
            for start in range(0, len(selected), window_length):
                window = selected[start : start + window_length]
                if window:
                    windows.append((dataset_name, window))
        print(
            f"layer seed={seed} {dataset_name}: selected_tokens={selected_tokens} "
            f"target={token_budget}"
        )

    random.Random(seed).shuffle(windows)
    return windows


def extract_layer_stats(
    model,
    tokenizer,
    layer: int,
    sae_snapshot_dir: Path,
    windows: list[tuple[str, list[int]]],
    top_k: int,
    dtype: torch.dtype,
) -> dict[str, DatasetFeatureStats]:
    layer_module = model.model.layers[layer]
    layer_device = next(layer_module.parameters()).device
    sae_path = sae_snapshot_dir / f"layer{layer}.sae.pt"
    if not sae_path.is_file():
        raise FileNotFoundError(f"Missing SAE checkpoint: {sae_path}")

    sae_state = torch.load(sae_path, map_location="cpu")
    w_enc = sae_state["W_enc"].to(device=layer_device, dtype=dtype).contiguous()
    b_enc = sae_state["b_enc"].to(device=layer_device, dtype=dtype).contiguous()
    del sae_state

    captured: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inputs, output):
        captured["residual"] = output[0] if isinstance(output, tuple) else output

    handle = layer_module.register_forward_hook(capture_hook)
    input_device = model.get_input_embeddings().weight.device
    stats = {name: DatasetFeatureStats() for name in ("ptsd", "daily", "empathetic")}
    try:
        for dataset_name, token_ids in tqdm(windows, desc=f"extract layer {layer}"):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=input_device)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                residual = captured["residual"].to(device=layer_device, dtype=dtype)
                pre_acts = residual @ w_enc.T + b_enc
                feature_values, feature_ids = torch.topk(pre_acts, k=top_k, dim=-1)

            stats[dataset_name].update(
                token_ids,
                feature_ids[0].detach().cpu(),
                feature_values[0].detach().float().cpu(),
                tokenizer,
            )
            captured.clear()
            del input_ids, residual, pre_acts, feature_values, feature_ids
    finally:
        handle.remove()

    del w_enc, b_enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return stats


def save_layer_summaries(layer_summaries: list[dict], output_root: Path) -> Path:
    ranked = sorted(
        layer_summaries,
        key=lambda item: item["mean_ptsd_minus_controls_per_1k"],
        reverse=True,
    )
    summary_path = output_root / "layer_ptsd_feature_difference_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(ranked, file, ensure_ascii=False, indent=2)
    return summary_path


def comparison_table(
    ptsd: DatasetFeatureStats,
    control: DatasetFeatureStats,
    control_name: str,
) -> pd.DataFrame:
    feature_ids = sorted(set(ptsd.activation_counts) | set(control.activation_counts))
    rows = []
    for feature_id in feature_ids:
        ptsd_frequency = ptsd.frequency_per_1k(feature_id)
        control_frequency = control.frequency_per_1k(feature_id)
        rows.append(
            {
                "comparison": f"ptsd_vs_{control_name}",
                "feature_id": feature_id,
                "ptsd_activation_count": ptsd.activation_counts[feature_id],
                f"{control_name}_activation_count": control.activation_counts[feature_id],
                "ptsd_activation_frequency_per_1k_tokens": ptsd_frequency,
                f"{control_name}_activation_frequency_per_1k_tokens": control_frequency,
                "activation_frequency_diff_ptsd_minus_control": (
                    ptsd_frequency - control_frequency
                ),
                "ptsd_noise_ratio": ptsd.noise_ratio(feature_id),
                "ptsd_top_content_tokens": ptsd.top_content_tokens(feature_id),
                f"{control_name}_noise_ratio": control.noise_ratio(feature_id),
                f"{control_name}_top_content_tokens": control.top_content_tokens(feature_id),
                "direction": "ptsd_specific",
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "activation_frequency_diff_ptsd_minus_control",
            "ptsd_activation_frequency_per_1k_tokens",
        ],
        ascending=False,
    ).reset_index(drop=True)


def save_layer_results(
    layer: int,
    stats: dict[str, DatasetFeatureStats],
    output_root: Path,
    comparison_top_n: int,
    final_feature_count: int,
) -> dict:
    output_dir = output_root / f"ptsd_control_sae_layer{layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_column = "activation_frequency_diff_ptsd_minus_control"

    daily = comparison_table(stats["ptsd"], stats["daily"], "daily")
    empathetic = comparison_table(stats["ptsd"], stats["empathetic"], "empathetic")
    daily_top = daily.head(comparison_top_n).copy()
    empathetic_top = empathetic.head(comparison_top_n).copy()

    daily_top.to_csv(output_dir / f"ptsd_vs_daily_top{comparison_top_n}.csv", index=False)
    empathetic_top.to_csv(
        output_dir / f"ptsd_vs_empathetic_top{comparison_top_n}.csv",
        index=False,
    )

    merged_top = pd.concat(
        [daily_top[["feature_id", diff_column]], empathetic_top[["feature_id", diff_column]]],
        ignore_index=True,
    )
    feature_ids = (
        merged_top.groupby("feature_id", as_index=False)[diff_column]
        .mean()
        .sort_values(diff_column, ascending=False)
        .head(final_feature_count)["feature_id"]
        .astype(int)
        .tolist()
    )
    feature_path = output_dir / f"layer_{layer}_PTSDfeature.json"
    with feature_path.open("w", encoding="utf-8") as file:
        json.dump(feature_ids, file, ensure_ascii=False)

    daily_diffs = daily.set_index("feature_id")[diff_column]
    empathetic_diffs = empathetic.set_index("feature_id")[diff_column]
    selected_daily = [float(daily_diffs.get(feature_id, 0.0)) for feature_id in feature_ids]
    selected_empathetic = [
        float(empathetic_diffs.get(feature_id, 0.0)) for feature_id in feature_ids
    ]
    summary = {
        "layer": layer,
        "top_feature_count": len(feature_ids),
        "mean_ptsd_minus_daily_per_1k": sum(selected_daily) / len(selected_daily),
        "mean_ptsd_minus_empathetic_per_1k": (
            sum(selected_empathetic) / len(selected_empathetic)
        ),
        "mean_ptsd_minus_controls_per_1k": (
            sum(selected_daily + selected_empathetic)
            / len(selected_daily + selected_empathetic)
        ),
    }
    print(f"saved layer {layer}: {feature_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Qwen3 SAE features that activate more frequently in PTSD text"
    )
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--sae-snapshot-dir", default="")
    parser.add_argument("--ptsd-data-dir", default="thousand-voices-trauma")
    parser.add_argument("--daily-data-path", default="daily_dialogues/dialogues_text.txt")
    parser.add_argument("--empathetic-data-dir", default="empathetic_dialogues")
    parser.add_argument("--output-dir", default="features_new")
    parser.add_argument("--layers", default="15-30")
    parser.add_argument("--tokens-per-dataset", type=int, default=80000)
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--sae-top-k", type=int, default=50)
    parser.add_argument("--comparison-top-n", type=int, default=50)
    parser.add_argument("--final-feature-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_dir = Path(args.project_dir).resolve()
    if (
        not (project_dir / "thousand-voices-trauma").exists()
        and (project_dir.parent / "thousand-voices-trauma").exists()
    ):
        project_dir = project_dir.parent

    layers = parse_layers(args.layers)
    dtype = infer_dtype(args.dtype)
    model_dir = resolve_path(args.model_dir, project_dir, project_dir / "qwen3-8b-base")
    sae_snapshot_dir = resolve_sae_snapshot_dir(args.sae_snapshot_dir, project_dir, layers)
    ptsd_data_dir = resolve_path(
        args.ptsd_data_dir,
        project_dir,
        project_dir / "thousand-voices-trauma",
    )
    daily_data_path = resolve_path(
        args.daily_data_path,
        project_dir,
        project_dir / "daily_dialogues" / "dialogues_text.txt",
    )
    empathetic_data_dir = resolve_path(
        args.empathetic_data_dir,
        project_dir,
        project_dir / "empathetic_dialogues",
    )
    output_root = resolve_path(args.output_dir, project_dir, project_dir / "features_new")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    torch.set_grad_enabled(False)
    print("SAE snapshot directory:", sae_snapshot_dir)

    corpora = {
        "ptsd": load_ptsd_texts(ptsd_data_dir),
        "daily": load_daily_texts(daily_data_path),
        "empathetic": load_empathetic_texts(empathetic_data_dir),
    }
    tokenized_corpora = tokenize_corpora(tokenizer, corpora)
    output_root.mkdir(parents=True, exist_ok=True)

    layer_summaries = []
    token_budget = args.tokens_per_dataset if args.tokens_per_dataset > 0 else None
    for layer in layers:
        layer_seed = args.seed + layer
        windows = make_windows(
            tokenized_corpora,
            token_budget=token_budget,
            window_length=args.window_length,
            seed=layer_seed,
        )
        stats = extract_layer_stats(
            model,
            tokenizer,
            layer=layer,
            sae_snapshot_dir=sae_snapshot_dir,
            windows=windows,
            top_k=args.sae_top_k,
            dtype=dtype,
        )
        layer_summary = save_layer_results(
            layer,
            stats,
            output_root=output_root,
            comparison_top_n=args.comparison_top_n,
            final_feature_count=args.final_feature_count,
        )
        layer_summaries.append(layer_summary)
        summary_path = save_layer_summaries(layer_summaries, output_root)
        del stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    layer_summaries.sort(
        key=lambda item: item["mean_ptsd_minus_controls_per_1k"],
        reverse=True,
    )
    summary_path = save_layer_summaries(layer_summaries, output_root)

    print("\nLayer ranking:")
    print(pd.DataFrame(layer_summaries).to_string(index=False))
    print("best intervention layer:", layer_summaries[0]["layer"])
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
