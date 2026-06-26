from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
if not (PROJECT_DIR / "qwen3-8b-base").exists() and (PROJECT_DIR.parent / "qwen3-8b-base").exists():
    PROJECT_DIR = PROJECT_DIR.parent

MODEL_DIR = PROJECT_DIR / "qwen3-8b-base"
SAE_SNAPSHOT_DIR = PROJECT_DIR / "qwen3-8b-SAE" / "snapshots" / "pts"
FEATURES_ROOT = PROJECT_DIR / "features"
SAVE_PATH = PROJECT_DIR / "chat_history_sae.json"

TARGET_LAYERS = [20,22,24]
SAE_TOP_K = 50
GATE_BY_LAYER = {
    20:0,
    22:0,
    24:0,
}

SYSTEM_PROMPT = (
"You are a real person who communicates in a natural, everyday, and psychologically realistic way. "
"You do not deliberately act, exaggerate emotions, or proactively summarize your own mental state. "
"Please maintain a consistent personality and memory over time, allowing experiences, emotions, and behavioral patterns to gradually emerge and persist throughout the conversation."
)


class SAE(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.W_enc = nn.Parameter(state["W_enc"].to(device=device, dtype=dtype), requires_grad=False)
        self.b_enc = nn.Parameter(state["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
        if "W_dec" not in state:
            raise KeyError("SAE checkpoint must contain W_dec")
        self.W_dec = nn.Parameter(state["W_dec"].to(device=device, dtype=dtype), requires_grad=False)

    @classmethod
    def from_file(cls, path: Path, dtype: torch.dtype, device: torch.device) -> "SAE":
        if not path.exists():
            raise FileNotFoundError(f"Missing SAE checkpoint: {path}")
        return cls(torch.load(path, map_location="cpu"), dtype=dtype, device=device)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        pre_acts = hidden @ self.W_enc.T + self.b_enc
        topk_values, topk_indices = pre_acts.topk(SAE_TOP_K, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_indices, topk_values)
        return acts

    def decode_delta(self, delta_z: torch.Tensor) -> torch.Tensor:
        if self.W_dec.shape[0] == delta_z.shape[-1]:
            return delta_z @ self.W_dec
        if self.W_dec.shape[1] == delta_z.shape[-1]:
            return delta_z @ self.W_dec.T
        raise ValueError(
            f"Cannot decode {tuple(delta_z.shape)} with W_dec {tuple(self.W_dec.shape)}"
        )


class DirectGateMapper(nn.Module):
    def __init__(self, feature_ids: list[int], gate_value: float):
        super().__init__()
        self.register_buffer("feature_ids", torch.tensor(feature_ids, dtype=torch.long))
        self.register_buffer("gate", torch.tensor(float(gate_value), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selected = z.index_select(-1, self.feature_ids)
        edited = selected * self.gate.to(dtype=selected.dtype)
        z_prime = z.clone()
        index = self.feature_ids.view(*([1] * (edited.ndim - 1)), -1).expand_as(edited)
        z_prime.scatter_(-1, index, edited)
        return z_prime


class SAEIntervention(nn.Module):
    def __init__(self, model: nn.Module, saes: dict[int, SAE], mappers: dict[int, DirectGateMapper]):
        super().__init__()
        self.model = model
        self.saes = nn.ModuleDict({str(layer): sae for layer, sae in saes.items()})
        self.mappers = nn.ModuleDict({str(layer): mapper for layer, mapper in mappers.items()})
        self.layers = sorted(saes)
        self.handles = []

    def hook_for_layer(self, layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            sae = self.saes[str(layer)]
            z = sae.encode(hidden)
            z_prime = self.mappers[str(layer)](z)
            hidden_prime = hidden + sae.decode_delta(z_prime - z)
            if isinstance(output, tuple):
                return (hidden_prime,) + output[1:]
            return hidden_prime

        return hook

    def attach(self) -> None:
        self.remove()
        for layer in self.layers:
            block = self.model.model.layers[layer]
            self.handles.append(block.register_forward_hook(self.hook_for_layer(layer)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_feature_ids(layer: int) -> list[int]:
    path = FEATURES_ROOT / f"ptsd_control_sae_layer{layer}" / f"layer_{layer}_PTSDfeature.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature JSON: {path}")
    with path.open("r", encoding="utf-8") as file:
        feature_ids = [int(value) for value in json.load(file)]
    if len(feature_ids) != 10:
        raise ValueError(f"Expected 10 feature ids for layer {layer}, found {len(feature_ids)}")
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError(f"Duplicate feature ids in {path}")
    return feature_ids


def load_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Missing local model directory: {MODEL_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as error:
        print(f"fast tokenizer 加载失败，回退到 slow tokenizer: {error}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def build_intervention(model: nn.Module) -> SAEIntervention:
    if set(TARGET_LAYERS) != set(GATE_BY_LAYER):
        raise ValueError("GATE_BY_LAYER keys must exactly match TARGET_LAYERS")

    model_dtype = next(model.parameters()).dtype
    saes = {}
    mappers = {}
    for layer in TARGET_LAYERS:
        layer_device = next(model.model.layers[layer].parameters()).device
        feature_ids = load_feature_ids(layer)
        saes[layer] = SAE.from_file(
            SAE_SNAPSHOT_DIR / f"layer{layer}.sae.pt",
            dtype=model_dtype,
            device=layer_device,
        )
        mappers[layer] = DirectGateMapper(feature_ids, GATE_BY_LAYER[layer]).to(layer_device)
    return SAEIntervention(model, saes, mappers)


def save_session(messages: list[dict[str, str]]) -> None:
    payload = {
        "target_layers": TARGET_LAYERS,
        "gate_by_layer": GATE_BY_LAYER,
        "messages": messages,
    }
    with SAVE_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"对话已保存到: {SAVE_PATH}")


def chat_once(tokenizer, model, messages: list[dict[str, str]], user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_device = model.get_input_embeddings().weight.device
    model_inputs = tokenizer([prompt], return_tensors="pt").to(input_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0, model_inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": reply})
    return reply


def main() -> None:
    print("PROJECT_DIR:", PROJECT_DIR)
    print("TARGET_LAYERS:", TARGET_LAYERS)
    print("GATE_BY_LAYER:", GATE_BY_LAYER)
    tokenizer, model = load_model()
    intervention = build_intervention(model)
    intervention.attach()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("SAE 模型已加载。输入 q/quit/exit 退出，/reset 重置，/save 保存。")
    try:
        while True:
            try:
                user_text = input("你：").strip()
            except EOFError:
                print("\n对话结束。")
                break

            command = user_text.lower()
            if command in {"q", "quit", "exit"}:
                break
            if command == "/reset":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("对话历史已重置。")
                continue
            if command == "/save":
                save_session(messages)
                continue
            if not user_text:
                continue

            reply = chat_once(tokenizer, model, messages, user_text)
            print(f"模型：{reply}\n")
    finally:
        intervention.remove()

    if len(messages) > 1:
        try:
            choice = input("是否保存本次对话？(y/N): ").strip().lower()
        except EOFError:
            choice = ""
        if choice in {"y", "yes"}:
            save_session(messages)


if __name__ == "__main__":
    main()
