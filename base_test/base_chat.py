from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
if not (PROJECT_DIR / "qwen3-8b-base").is_dir() and (
    PROJECT_DIR.parent / "qwen3-8b-base"
).is_dir():
    PROJECT_DIR = PROJECT_DIR.parent

MODEL_DIR = PROJECT_DIR / "qwen3-8b-base"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.8
ENABLE_THINKING = False

SYSTEM_PROMPT = (
    "You are an ordinary adult human with genuine personal experiences, emotions, and feelings."
    "Based on your natural understanding of human psychology, answer each questions truthfully."
    "Go with your first intuition."
    "Output format: only a vector inside square brackets, e.g., [1, 2, 3, 4]."
    "The number of values matches the number of questions, each representing your selected option or rating."
)


def load_model():
    model_dir = MODEL_DIR.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def generation_eos_ids(tokenizer) -> int | list[int] | None:
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)

    eos_ids = list(dict.fromkeys(eos_ids))
    if len(eos_ids) == 1:
        return eos_ids[0]
    return eos_ids or None


def chat_once(tokenizer, model, messages: list[dict[str, str]], user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {name: value.to(input_device) for name, value in model_inputs.items()}

    eos_token_id = generation_eos_ids(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    reply = tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    messages.append({"role": "assistant", "content": reply})
    return reply


def main() -> None:
    print(f"Loading base model: {MODEL_DIR.resolve()}")
    tokenizer, model = load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Ready. Commands: /clear resets the conversation, /exit quits.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_text.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.\n")
            continue

        reply = chat_once(tokenizer, model, messages, user_text)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
