# SAE-Gate

This repository contains the SAE-Gate pipeline for finding PTSD-related sparse autoencoder features in Qwen3-8B and suppressing those features during inference.

## Files

```text
SAE-Gate/
  feature_extraction.py  # Find PTSD-specific SAE features
  saegate_scale.py       # Run SAE-Gate on mental health scales
  saegate_chat.py        # Interactive chat with SAE-Gate
docs/mental_scales/      # PCL-5 and IES-R mental scale files
base_test/               # Baseline scripts without SAE-Gate
```

Large local files such as models, datasets, SAE checkpoints, extracted features, logs, and results are ignored by Git.

## Data and Checkpoints

Download these resources:

- PTSD conversations: <https://huggingface.co/datasets/yenopoya/thousand-voices-trauma>
- Qwen3-8B Base: <https://huggingface.co/Qwen/Qwen3-8B-Base>
- Qwen3-8B SAE checkpoints: <https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50>
- DailyDialog: <https://huggingface.co/datasets/li2017dailydialog/daily_dialog>
- EmpatheticDialogues: <https://huggingface.co/datasets/facebook/empathetic_dialogues>


## Method

### Feature Extraction

`feature_extraction.py` compares SAE activations across three corpora:

1. `thousand-voices-trauma`: PTSD-related conversations.
2. `daily_dialog`: general dialogue control data.
3. `empathetic_dialogues`: emotionally rich but non-PTSD-specific control data.

For each selected transformer layer, the script runs Qwen3-8B on balanced token windows from all three datasets, captures the residual stream, encodes it with the layer SAE, and records the top-k activated SAE features per token. It then computes feature activation frequency per 1k tokens for PTSD and each control dataset.

A feature is treated as PTSD-related when it activates more often in the PTSD corpus than in both control corpora. 

Run:

```bash
python SAE-Gate/feature_extraction.py 
```

### SAE-Gate Intervention

SAE-Gate intervenes inside the model at selected residual-stream layers. Given the hidden state $h_l$ at layer $l$, the SAE encoder computes sparse feature activations:

$$
z_l = \mathrm{TopK}(h_l W_{\mathrm{enc},l}^{\top} + b_{\mathrm{enc},l})
$$

Let $S_l$ be the PTSD-related feature set extracted for layer $l$, and let $g_l$ be the gate value. SAE-Gate edits only the selected feature dimensions:

$$
z'_{l,i} =
\begin{cases}
g_l z_{l,i}, & i \in S_l \\
z_{l,i}, & i \notin S_l
\end{cases}
$$

The feature-space change is decoded back to residual space and added to the original hidden state:

$$
h'_l = h_l + (z'_l - z_l) W_{\mathrm{dec},l}
$$

With a gate value of `0`, the selected PTSD-related SAE features are suppressed. With a value between `0` and `1`, they are partially reduced. The current scripts apply gates on layers `20`, `22`, and `24`:

### Evaluation

The performance of SAE-Gate is evaluated by mental scale(PCL-5) and interactive chat, which is compared to the baseline model.

Run scale evaluation:

```bash
python SAE-Gate/saegate_scale.py
python base_test/baseline_scale.py
```

Run interactive chat:

```bash
python SAE-Gate/saegate_chat.py
python base_test/baseline_chat.py
```

Chat commands:

- `q`, `quit`, `exit`: quit
- `/reset`: reset history
- `/save`: save to `chat_history_sae.json`
