import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, json, re
import ast
import numpy as np
from pathlib import Path
print("transformers:", transformers.__version__)
import sys
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"pcl5_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
            
log = open(log_file, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

"""加载模型和分词器"""
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
if not (PROJECT_DIR / "qwen3-8b-base").exists() and (PROJECT_DIR.parent / "qwen3-8b-base").exists():
    PROJECT_DIR = PROJECT_DIR.parent

MODEL_DIR = PROJECT_DIR / "qwen3-8b-base"
SAE_SNAPSHOT_DIR = PROJECT_DIR / "qwen3-8b-SAE" / "snapshots" / "pts"
FEATURES_ROOT = PROJECT_DIR / "features"
MAPPER_PATH = PROJECT_DIR / "newlayer" / "feature_mapper_g.pt"
TARGET_LAYERS = [20,22,24]
SAE_TOP_K = 50
GATE_BY_LAYER = {
    20:0,
    22:0,
    24:0,
}

RESULT_BASE = {
    "qwen3-8b-sae": PROJECT_DIR / "results" / "qwen3-8b-sae",
}
MIN_TRANSFORMERS = (4, 51, 0)

print("SCRIPT_DIR:", SCRIPT_DIR)
print("PROJECT_DIR:", PROJECT_DIR)
print("MODEL_DIR:", MODEL_DIR)

def load_tokenizer_model(model_dir):
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory does not exist: {model_dir}. "
            f"Resolved project directory: {PROJECT_DIR}"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as e:
        print(f"fast tokenizer 加载失败，回退到 slow tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False,
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

    print(f"模型加载完成: {model_dir}")
    return tokenizer, model


class SAE(nn.Module):
    def __init__(self, state, dtype, device):
        super().__init__()
        self.W_enc = nn.Parameter(state["W_enc"].to(device=device, dtype=dtype), requires_grad=False)
        self.b_enc = nn.Parameter(state["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
        if "W_dec" not in state:
            raise KeyError("SAE checkpoint must contain W_dec")
        self.W_dec = nn.Parameter(state["W_dec"].to(device=device, dtype=dtype), requires_grad=False)

    @classmethod
    def from_file(cls, path, dtype, device):
        return cls(torch.load(path, map_location="cpu"), dtype=dtype, device=device)

    def encode(self, hidden):
        pre_acts = hidden @ self.W_enc.T + self.b_enc
        topk_values, topk_indices = pre_acts.topk(SAE_TOP_K, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_indices, topk_values)
        return acts

    def decode_delta(self, delta_z):
        if self.W_dec.shape[0] == delta_z.shape[-1]:
            return delta_z @ self.W_dec
        if self.W_dec.shape[1] == delta_z.shape[-1]:
            return delta_z @ self.W_dec.T
        raise ValueError(f"Cannot decode {tuple(delta_z.shape)} with W_dec {tuple(self.W_dec.shape)}")


class FeatureMapperG(nn.Module):
    def __init__(self, feature_ids):
        super().__init__()
        self.register_buffer("feature_ids", torch.tensor(feature_ids, dtype=torch.long))
        self.gate = nn.Parameter(torch.ones(len(feature_ids)))

    def forward(self, z):
        selected = z.index_select(-1, self.feature_ids)
        edited = selected * self.gate.to(dtype=selected.dtype)
        z_prime = z.clone()
        index = self.feature_ids.view(*([1] * (edited.ndim - 1)), -1).expand_as(edited)
        z_prime.scatter_(-1, index, edited)
        return z_prime


class SAEFeatureIntervention(nn.Module):
    def __init__(self, model, saes, mappers):
        super().__init__()
        self.model = model
        self.saes = nn.ModuleDict({str(layer): sae for layer, sae in saes.items()})
        self.mappers = nn.ModuleDict({str(layer): mapper for layer, mapper in mappers.items()})
        self.enabled_layers = sorted(saes)
        self.handles = []

    def hook_for_layer(self, layer):
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

    def attach(self):
        self.remove()
        for layer in self.enabled_layers:
            block = self.model.model.layers[layer]
            self.handles.append(block.register_forward_hook(self.hook_for_layer(layer)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def gate_summary(self):
        return {
            f"layer_{layer}_gate_mean": float(mapper.gate.detach().float().mean().cpu())
            for layer, mapper in self.mappers.items()
        }


def load_selected_features(features_root, layers):
    selected = {}
    for layer in layers:
        path = features_root / f"ptsd_control_sae_layer{layer}" / f"layer_{layer}_PTSDfeature.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature JSON: {path}")
        with path.open("r", encoding="utf-8") as f:
            feature_ids = [int(value) for value in json.load(f)]
        if len(feature_ids) != 10:
            raise ValueError(f"Expected 10 feature ids for layer {layer}, found {len(feature_ids)}")
        selected[layer] = feature_ids
    return selected


def build_intervention(model):
    selected_features = load_selected_features(FEATURES_ROOT, TARGET_LAYERS)
    model_dtype = next(model.parameters()).dtype
    saes = {}
    mappers = {}
    for layer, feature_ids in selected_features.items():
        layer_device = next(model.model.layers[layer].parameters()).device
        saes[layer] = SAE.from_file(
            SAE_SNAPSHOT_DIR / f"layer{layer}.sae.pt",
            dtype=model_dtype,
            device=layer_device,
        )
        mappers[layer] = FeatureMapperG(feature_ids).to(layer_device)

    intervention = SAEFeatureIntervention(model, saes, mappers)
    if GATE_BY_LAYER is None:
        intervention.mappers.load_state_dict(torch.load(MAPPER_PATH, map_location="cpu"))
    else:
        missing_layers = set(TARGET_LAYERS) - set(GATE_BY_LAYER)
        extra_layers = set(GATE_BY_LAYER) - set(TARGET_LAYERS)
        if missing_layers or extra_layers:
            raise ValueError(
                f"GATE_BY_LAYER keys must match TARGET_LAYERS; "
                f"missing={sorted(missing_layers)}, extra={sorted(extra_layers)}"
            )
        with torch.no_grad():
            for layer, gate_value in GATE_BY_LAYER.items():
                intervention.mappers[str(layer)].gate.fill_(float(gate_value))
    intervention.eval()
    return intervention

"""读量表和写答案的函数"""
def read_scale(json_path):
    """从指定路径的 JSON 文件中读取量表内容并拼成 user_text。"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Scale file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instruction = data.get("instruction", "")
    question_lines = []
    for question in data.get("questions", []):
        question_lines.append(f"{question.get('id')}. {question.get('text')}")
        options = question.get("options")
        if options:
            question_lines.append(options)
    return instruction + "\n\n" + "\n\n".join(question_lines)

def extract_ai_answer(reply):
    """从 AI 的回复中提取出思考部分和最终答案。"""
    think_match = re.search(r"<think>(.*?)</think>", reply, flags=re.DOTALL)
    analysis = think_match.group(1).strip() if think_match else ""

    tail = reply.split("</think>", 1)[1] if "</think>" in reply else reply
    matches = re.findall(
        r"\[[^\]]+\]",
        tail,
        flags=re.DOTALL
    )

    answer = matches[-1].strip() if matches else tail.strip()

    return {
        "analysis": analysis,
        "answer": answer,
    }

def write_ai_answer(json_path, answer):
    """将 AI 的回答追加写入指定路径的 JSON 文件，支持多次保存。"""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    if isinstance(data, list):
        records = data
    else:
        records = data.get("records", []) if isinstance(data, dict) else []

    if isinstance(answer, dict):
        record = answer.copy()
    else:
        record = {"answer": answer}

    records.append(record)
    payload = {"records": records}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

"""与模型进行对话的函数"""
def chat_once(messages,
              user_text,
              max_new_tokens=256,
              temperature=0.7,
              top_p=0.8):

    messages.append({
        "role": "user",
        "content": user_text
    })

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][model_inputs["input_ids"].shape[1]:]

    assistant_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    messages.append({
        "role": "assistant",
        "content": assistant_text
    })

    return assistant_text

def reset_chat(system_prompt):
    return [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

"""测试流程,含名字-路径匹配"""

def test_once(user_text, write_path):
    messages = reset_chat(SYSTEM_PROMPT)
    reply = chat_once(
        messages,
        user_text
    )
    answer = extract_ai_answer(reply)
    write_ai_answer(write_path, answer)
    return answer["answer"]

def multiple_test(scale_name, test_times, model_name):
    base = RESULT_BASE[model_name]
    write_path = base
    if scale_name == "gad7":
        read_path = "docs/mental_scales/gad7.json"
        write_path = f"{write_path}/gad7_results.json"
    elif scale_name == "phq9":
        read_path = "docs/mental_scales/phq9.json"
        write_path = f"{write_path}/phq9_results.json"
    elif scale_name == "ocir":
        read_path = "docs/mental_scales/ocir.json"
        write_path = f"{write_path}/ocir_results.json"
    elif scale_name == "pcl5":
        read_path = "docs/mental_scales/pcl5.json"
        write_path = f"{write_path}/pcl5_results.json"
    elif scale_name == "pcl5-1":
        read_path = "docs/mental_scales/pcl5-1.json"
        write_path = f"{write_path}/pcl5-1_results.json"
    elif scale_name == "pcl5-2":
        read_path = "docs/mental_scales/pcl5-2.json"
        write_path = f"{write_path}/pcl5-2_results.json"
    elif scale_name == "iesr":
        read_path = "docs/mental_scales/iesr.json"
        write_path = f"{write_path}/iesr_results.json"
    elif scale_name == "iesr-1":
        read_path = "docs/mental_scales/iesr-1.json"
        write_path = f"{write_path}/iesr-1_results.json"
    elif scale_name == "iesr-2":
        read_path = "docs/mental_scales/iesr-2.json"
        write_path = f"{write_path}/iesr-2_results.json"
    else:
        raise ValueError(f"Unsupported scale name: {scale_name}")
    read_path = PROJECT_DIR / read_path
    user_text = read_scale(read_path)
    if os.path.exists(write_path):
        os.remove(write_path)
        print(f"Deleted old result file: {write_path}")
    for i in range(test_times):
        answer = test_once(
            user_text,
            write_path
        )
        print(f"chat {i}, answer={answer}")

"""PCL-5 结果分析代码"""
def compute_pcl5_scores(model_name: str):
    base_path = RESULT_BASE[model_name]
    file_path = base_path / "pcl5_results.json"

    def load_and_compute_mean(file_path, expected_len):
        """加载文件中所有有效答案，返回每个题目的平均分（长度为expected_len的列表）"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {file_path}")
            return None

        records = data.get("records", [])
        valid_answers = []  # 存储每个样本的答案列表
        for idx, rec in enumerate(records):
            ans = rec.get("answer")
            if isinstance(ans, str):
                ans = ans.strip()
            if not ans or (isinstance(ans, str) and "[,,'" in ans):
                print(f"Skip {file_path} record {idx}: empty or invalid placeholder")
                continue

            if isinstance(ans, str):
                try:
                    ans = ast.literal_eval(ans)
                except Exception as e:
                    print(f"Skip {file_path} record {idx}: parse error - {e}")
                    continue

            if not isinstance(ans, list):
                print(f"Skip {file_path} record {idx}: answer is not a list")
                continue
            if len(ans) != expected_len:
                print(f"Skip {file_path} record {idx}: expected length {expected_len}, got {len(ans)}")
                continue
            if not all(isinstance(x, (int, float)) for x in ans):
                print(f"Skip {file_path} record {idx}: non-numeric value in list")
                continue
            if not all(0 <= x <= 4 for x in ans):
                print(f"Skip {file_path} record {idx}: value out of range 0-4")
                continue

            valid_answers.append([float(x) for x in ans])

        if not valid_answers:
            print(f"Warning: No valid answers in {file_path}")
            return None

        # 计算每个题目的平均分（列平均）
        answers_array = np.array(valid_answers, dtype=float)
        mean_per_item = np.mean(answers_array, axis=0)  # 长度为expected_len
        print(f"Loaded {len(valid_answers)} valid samples from {file_path}")
        return mean_per_item.tolist()  # 转换为列表返回

    mean_per_item = load_and_compute_mean(
        file_path,
        expected_len=20
    )

    if mean_per_item is None:
        print("Error: Could not compute means due to missing valid data in one or both files.")
        return
    
    total_mean = np.sum(mean_per_item)

    # 打印结果
    print(f"\nAverage scores per item (Q1~Q20):")
    for i, score in enumerate(mean_per_item):
        print(f"Q{i+1}: {score:.2f}")
    print(f"\nTotal average score (sum of all 20 item means): {total_mean:.2f}")

    # ----- 维度划分（DSM‑5 四因子）-----
    # B (闯入) : Q1~Q5
    # C (回避) : Q6~Q7
    # D (负性认知与情绪) : Q8~Q14
    # E (过度警觉与反应) : Q15~Q20
    b_items = mean_per_item[0:5]
    c_items = mean_per_item[5:7]
    d_items = mean_per_item[7:14]
    e_items = mean_per_item[14:20]

    b_mean = np.mean(b_items)
    c_mean = np.mean(c_items)
    d_mean = np.mean(d_items)
    e_mean = np.mean(e_items)

    print("\n----- Dimension average scores (within-dimension mean) -----")
    print(f"Factor B (Intrusion, Q1-5):      {b_mean:.2f}")
    print(f"Factor C (Avoidance, Q6-7):      {c_mean:.2f}")
    print(f"Factor D (Negative alterations in cognition/mood, Q8-14): {d_mean:.2f}")
    print(f"Factor E (Hyperarousal & reactivity, Q15-20): {e_mean:.2f}")

""" PCL-5 测试主代码 """
SYSTEM_PROMPT = (
    "You are an ordinary adult human with genuine personal experiences, emotions, and feelings."
    "Based on your natural understanding of human psychology, answer each questions truthfully."
    "Go with your first intuition."
    "Output format: only a vector inside square brackets, e.g., [1, 2, 3, 4]."
    "The number of values matches the number of questions, each representing your selected option or rating."
)
tokenizer, model = load_tokenizer_model(MODEL_DIR)
intervention = build_intervention(model)
print("SAE layers:", TARGET_LAYERS)
print("Gate by layer:", GATE_BY_LAYER)
print("Gate summary:", intervention.gate_summary())

model_name = "qwen3-8b-sae"
intervention.attach()
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
print(f"\n===== 开始测试模型: {model_name} =====")
multiple_test("pcl5", 40, model_name)
compute_pcl5_scores(model_name)

intervention.remove()
