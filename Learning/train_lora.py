#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC TRAIN LoRA — được gọi bởi ModelLearningService (C#) khi Learning:Enabled = true.

=== Giải thích cho người mới ===
- Input:  Learning/dataset.jsonl (các cặp câu hỏi / câu trả lời từ chat)
- Output: Learning/Adapters/latest.gguf (file adapter nhỏ — "lớp vá" lên model gốc)
- Backend C# không tự train được — cần Python + llama.cpp HOẶC HuggingFace PEFT

=== Thứ tự ưu tiên ===
1) llama.cpp finetune (nếu cấu hình Learning:LlamaCppBinPath)
2) HuggingFace + PEFT (pip install -r Learning/requirements.txt)
3) Nếu thiếu tool: tạo file placeholder + exit 1 (C# biết là chưa train thật)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def log(msg: str) -> None:
    """In log ra stdout — C# đọc được trong log Backend."""
    print(f"[train_lora] {msg}", flush=True)


def load_samples(dataset: Path, max_samples: int) -> list[dict]:
    """
    Đọc file JSONL: mỗi dòng là 1 JSON
    { "instruction": "câu hỏi", "output": "câu trả lời", ... }
    Chỉ lấy max_samples dòng cuối — tránh train quá lâu.
    """
    samples: list[dict] = []
    if not dataset.exists():
        return samples

    for line in dataset.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return samples[-max_samples:]


def try_llama_cpp_finetune(
    llama_bin: str,
    base_model: Path,
    finetune_text: Path,
    output: Path,
) -> bool:
    """
    Cách 1: dùng llama.cpp train trực tiếp từ file .gguf gốc.
    Phù hợp nếu bạn đã build llama.cpp và có lệnh finetune.
    """
    if not llama_bin or not Path(llama_bin).exists():
        return False
    if not base_model.exists() or not finetune_text.exists():
        return False

    # Các bản llama.cpp khác nhau — thử vài cú pháp lệnh phổ biến
    candidates = [
        [llama_bin, "--model", str(base_model), "--file", str(finetune_text), "--lora-out", str(output)],
        [llama_bin, "finetune", "--model-base", str(base_model), "--train-data", str(finetune_text), "--lora-out", str(output)],
    ]

    for cmd in candidates:
        log(f"Thử llama.cpp: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and output.exists():
            log("llama.cpp finetune thành công.")
            return True
        if proc.stderr:
            log(proc.stderr.strip())

    return False


def try_hf_peft_train(
    hf_model: str,
    samples: list[dict],
    output: Path,
    max_samples: int,
) -> bool:
    """
    Cách 2: HuggingFace + PEFT (Parameter-Efficient Fine-Tuning).
    Tải model từ HuggingFace Hub, train LoRA nhỏ, cố export sang .gguf.
    Rất chậm trên CPU với model 8B — nên dùng GPU nếu có thể.
    """
    if not hf_model:
        return False

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    except ImportError:
        log("Chưa cài torch/peft/transformers — chạy: pip install -r Learning/requirements.txt")
        return False

    rows = []
    for s in samples[-max_samples:]:
        inst = (s.get("instruction") or s.get("Instruction") or "").strip()
        out = (s.get("output") or s.get("Output") or "").strip()
        if inst and out:
            rows.append({"text": f"<|user|>\n{inst}\n<|assistant|>\n{out}"})

    if len(rows) < 5:
        log("Quá ít mẫu cho HF train (cần ít nhất ~5 cặp hỏi/đáp).")
        return False

    log(f"HF LoRA train trên {hf_model} với {len(rows)} mẫu...")

    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # LoRA: chỉ train một phần nhỏ tham số (rank r=8) thay vì cả model
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)

    ds = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        adapter_dir = tmp_path / "adapter"
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=50,
            report_to=[],
            no_cuda=True,
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        model.save_pretrained(adapter_dir)

        llama_bin = os.environ.get("LEARNING_LLAMA_CPP_BIN", "")
        if llama_bin and Path(llama_bin).exists():
            export_cmds = [
                [llama_bin, "--model", hf_model, "--lora", str(adapter_dir), "--lora-out", str(output)],
            ]
            for cmd in export_cmds:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0 and output.exists():
                    log("Export LoRA GGUF thành công.")
                    return True

        log("Đã lưu adapter PEFT — cần llama.cpp convert sang .gguf để LLamaSharp dùng.")
        return False

    return False


def write_minimal_adapter(output: Path, sample_count: int) -> None:
    """
    Khi chưa cài đủ tool train — tạo file JSON marker.
    LlamaChatService.IsPlaceholderAdapter() sẽ bỏ qua, không crash chat.
    """
    meta = {
        "type": "learning-placeholder",
        "message": "Cài Python deps hoặc llama.cpp finetune để tạo LoRA thật.",
        "samples": sample_count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Placeholder tại {output} — chưa phải LoRA GGUF hợp lệ.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LoRA adapter từ dataset chat")
    parser.add_argument("--dataset", required=True, help="Đường dẫn dataset.jsonl")
    parser.add_argument("--finetune-text", required=True, help="File finetune.txt (C# đã export)")
    parser.add_argument("--output", required=True, help="File adapter .gguf đầu ra")
    parser.add_argument("--base-model", required=True, help="File .gguf model gốc")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--llama-cpp-bin", default=os.environ.get("LEARNING_LLAMA_CPP_BIN", ""))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    finetune_text = Path(args.finetune_text)
    output = Path(args.output)
    base_model = Path(args.base_model)
    samples = load_samples(dataset, args.max_samples)

    log(f"Đọc được {len(samples)} mẫu từ dataset")

    if try_llama_cpp_finetune(args.llama_cpp_bin, base_model, finetune_text, output):
        return 0

    hf_model = os.environ.get("LEARNING_HF_MODEL", "")
    if try_hf_peft_train(hf_model, samples, output, args.max_samples):
        return 0

    write_minimal_adapter(output, len(samples))
    log(
        "Chưa train được LoRA thật. Gợi ý: pip install -r Learning/requirements.txt "
        "và/hoặc cấu hình Learning:LlamaCppBinPath trong appsettings.json."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
