#!/usr/bin/env python
"""Merge a LoRA-adapted checkpoint into a standalone full-precision model.

The script reproduces the layout used by the fully fine-tuned releases by
dumping sharded weights together with tokenizer and auxiliary assets so the
result can be loaded without providing ``--model-base`` at inference time.

中文简介：用于将 LoRA 训练得到的增量权重合并回完整模型目录，对齐完整微调版
发布的目录结构，使推理时无需再额外提供基座模型路径。
"""

# 主要步骤概览：
# 1. 解析命令行参数并创建输出目录。
# 2. 加载基座模型与分词器，必要时初始化多模态视觉模块。
# 3. 将 LoRA 权重折叠回稠密权重，同时恢复非 LoRA 的其余可训练参数。
# 4. 清理配置文件字段，更新模型元信息。
# 5. 保存权重分片、分词器资产与生成配置，得到可直接推理的完整模型。

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import torch
from types import SimpleNamespace
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from llamavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM
from llava.mm_utils import get_model_name_from_path


def parse_args():
    """解析命令行参数，获取合并所需的路径与控制项。"""

    parser = argparse.ArgumentParser(description="Merge LLaMA-VID LoRA weights into a full checkpoint")
    # LoRA 权重所在目录，通常是训练脚本输出的 adapter 目录。
    parser.add_argument("--model-path", required=True, help="Directory containing LoRA adapter weights")
    # 基座模型目录，用于补齐完整权重与分词器资产。
    parser.add_argument("--model-base", required=True, help="Directory containing the full base model")
    # 合并后的完整模型输出目录。
    parser.add_argument("--output-dir", required=True, help="Directory to save the merged full-precision model")
    # 控制 save_pretrained 切分权重文件的最大尺寸，默认 10GB。
    parser.add_argument("--max-shard-size", default="10GB", help="Shard size passed to save_pretrained")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device identifier used while materialising and merging the model (e.g. cuda:0)",
    )
    # 默认会清除量化配置，如需保留则传入该开关。
    parser.add_argument("--keep-quant-config", action="store_true", help="Retain quantization_config in config.json")
    return parser.parse_args()


def remove_config_entries(config, keys):
    """从配置对象中删除指定字段，避免保存后残留无效信息。"""

    for key in keys:
        if hasattr(config, key):
            try:
                delattr(config, key)
            except AttributeError:
                pass
        # 同步移除 __dict__/_internal_dict 中的同名字段，防止重复写回。
        if hasattr(config, "__dict__"):
            config.__dict__.pop(key, None)
        if hasattr(config, "_internal_dict"):
            config._internal_dict.pop(key, None)


def main():
    """脚本入口：执行 LoRA 合并、配置清理与结果持久化。"""

    args = parse_args()

    output_dir = Path(args.output_dir)
    # 确保输出目录存在，必要时递归创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name_from_path(args.model_path)
    # 模型名包含 "vid" 时视为多模态模型，需要初始化视觉组件。
    is_multimodal = "vid" in model_name.lower()

    device = torch.device(args.device)

    # 加载基座模型/分词器，并结合 LoRA/基座配置推断默认精度。
    tokenizer, model, cfg = load_base_model(args, is_multimodal, device)

    if is_multimodal:
        # 为多模态模型补全视觉塔、注意力模块等组件。
        setup_multimodal_modules(model, cfg, args, tokenizer, device)

    # 将 LoRA 增量合并回模型稠密权重。
    model = merge_lora(model, args.model_path, device)
    # 恢复 non_lora_trainables.bin 中存放的附加可训练参数（如视觉塔增量）。
    load_non_lora_trainables(model, args.model_path)
    # 确保视觉塔模型已加载到目标设备并转成正确 dtype。
    ensure_vision_tower_loaded(model, device)

    # 切换评估模式，避免推理阶段触发 dropout 等训练逻辑。
    model.eval()

    # 清理配置中的临时字段，写入新的模型类型与 dtype 信息。
    clean_config(model, output_dir, args.keep_quant_config, is_multimodal)

    # 保存权重、分词器及生成配置，形成完整可部署目录。
    save_checkpoint(model, tokenizer, args.model_base, output_dir, args.max_shard_size)

    print(f"Merged model saved to {output_dir}")


def load_base_model(args, is_multimodal: bool, device: torch.device) -> Tuple[AutoTokenizer, torch.nn.Module, AutoConfig]:
    """加载基座模型/分词器，并尽量复现训练时的环境设置。"""

    # model_kwargs 仿照训练阶段，指定默认精度与设备映射，减少额外拷贝。
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": infer_dtype(args.model_path if is_multimodal else args.model_base),
        "device_map": {"": device.type if device.index is None else f"{device.type}:{device.index}"}
        if device.type != "cpu" else None,
    }

    # device_map=None 需要直接移除，否则 transformers 会报错。
    if model_kwargs["device_map"] is None:
        model_kwargs.pop("device_map")

    if is_multimodal:
        # 多模态模型从 LoRA 目录加载 config 以获取视觉塔相关字段。
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        # LoRA 目录通常会额外导出 non_lora_trainables.bin，包含多模态投影器等权重。
        # 如果配置里缺失 pretrain_mm_mlp_adapter，则自动回退到该文件，避免用户手动修改配置。
        default_mm_adapter = Path(args.model_path) / "non_lora_trainables.bin"
        if getattr(config, "pretrain_mm_mlp_adapter", None) is None and default_mm_adapter.exists():
            config.pretrain_mm_mlp_adapter = str(default_mm_adapter)
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
        model = LlavaLlamaAttForCausalLM.from_pretrained(
            args.model_base,
            config=config,
            **model_kwargs,
        )
    else:
        # 纯语言模型完全使用基座目录中的配置。
        config = AutoConfig.from_pretrained(args.model_base, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            args.model_base,
            config=config,
            **model_kwargs,
        )

    return tokenizer, model, config


def setup_multimodal_modules(model, config, args, tokenizer, device):
    """初始化多模态模型所需的视觉塔、注意力模块等组件。"""

    if not hasattr(model, "get_model"):
        return

    vision_tower_path = getattr(config, "mm_vision_tower", None)
    if vision_tower_path is None:
        return

    # 使用 SimpleNamespace 构造与训练时一致的参数集合。
    model_args = SimpleNamespace(
        vision_tower=vision_tower_path,
        image_processor=getattr(config, "image_processor", None),
        mm_projector_type=getattr(config, "mm_projector_type", "linear"),
        mm_vision_select_layer=getattr(config, "mm_vision_select_layer", -2),
        mm_vision_select_feature=getattr(config, "mm_vision_select_feature", "patch"),
        pretrain_mm_mlp_adapter=getattr(config, "pretrain_mm_mlp_adapter", None),
        tune_mm_mlp_adapter=getattr(config, "tune_mm_mlp_adapter", False),
        freeze_mm_mlp_adapter=getattr(config, "freeze_mm_mlp_adapter", False),
        mm_use_im_start_end=getattr(config, "mm_use_im_start_end", False),
        mm_use_im_patch_token=getattr(config, "mm_use_im_patch_token", True),
        model_name_or_path=args.model_base,
        model_path=args.model_path,
        bert_type=getattr(config, "bert_type", "qformer_pretrain"),
        num_query=getattr(config, "num_query", 32),
        compress_type=getattr(config, "compress_type", None),
        pretrain_qformer=getattr(config, "pretrain_qformer", None),
    )

    # 初始化视觉塔与 Q-former，使其与训练时的权重布局保持一致。
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=None,
        max_token=getattr(config, "max_token", getattr(config, "model_max_length", 2048)),
    )

    # 初始化视觉 tokenizer，将图像特征映射到文本 token 序列。
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 初始化多模态注意力模块，for_eval=True 表示用于推理。
    model.get_model().initialize_attention_modules(model_args, for_eval=True)

    dtype = next(model.parameters()).dtype
    if hasattr(model.get_model(), "mm_projector"):
        # 将多模态投影器移动到目标设备，并保持与主模型一致的 dtype。
        model.get_model().mm_projector.to(device=device, dtype=dtype)


def infer_dtype(config: str) -> torch.dtype:
    """根据配置文件声明的 torch_dtype 推断默认使用的精度。"""

    # 默认返回 fp16；若配置声明为 bf16，则沿用 bf16 以避免精度转换。
    cfg = AutoConfig.from_pretrained(config, trust_remote_code=True)
    dtype = getattr(cfg, "torch_dtype", None)
    if isinstance(dtype, str) and dtype.lower() in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float16


def merge_lora(model: torch.nn.Module, lora_path: str, device: torch.device) -> torch.nn.Module:
    """将 LoRA 适配器合并进基座模型的稠密权重中。"""

    # 先将模型移动到目标设备，再加载 LoRA 适配器。
    model = model.to(device)
    peft_model = PeftModel.from_pretrained(model, lora_path)
    peft_model = peft_model.to(device)
    target_dtype = next(model.parameters()).dtype
    # merge_and_unload 会把 A/B 低秩矩阵折叠进线性层，并移除 LoRA 结构。
    merged = peft_model.merge_and_unload()
    return merged.to(device=device, dtype=target_dtype)


def load_non_lora_trainables(model: torch.nn.Module, lora_path: str) -> None:
    """加载 non_lora_trainables.bin 中存放的额外权重（如视觉塔或投影器增量）。"""

    non_lora_path = Path(lora_path) / "non_lora_trainables.bin"
    if not non_lora_path.exists():
        return

    # 读取补充权重并转换到目标 dtype，保证与主模型匹配。
    state_dict = torch.load(non_lora_path, map_location="cpu")
    target_dtype = next(model.parameters()).dtype
    converted = {k: v.to(dtype=target_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing:
        print(f"Warning: missing keys when loading non-LoRA weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading non-LoRA weights: {unexpected}")


def ensure_vision_tower_loaded(model: torch.nn.Module, device: torch.device) -> None:
    """确保多模态模型的视觉塔已正确加载并移动到目标设备。"""

    if not hasattr(model, "get_vision_tower"):
        return

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        return

    try:
        is_loaded = vision_tower.is_loaded
    except AttributeError:
        is_loaded = True

    if not is_loaded and hasattr(vision_tower, "load_model"):
        vision_tower.load_model()

    dtype = next(model.parameters()).dtype

    # 某些实现会将视觉塔封装为列表（例如 FSDP），需要逐个转移到设备上。
    towers = vision_tower if isinstance(vision_tower, (list, tuple)) else [vision_tower]
    for tower in towers:
        if hasattr(tower, "to"):
            tower.to(device=device, dtype=dtype)


def clean_config(model, output_dir: Path, keep_quant_config: bool, is_multimodal: bool) -> None:
    """清理配置文件并写入合并后所需的关键字段。"""

    config = model.config
    if not keep_quant_config:
        remove_config_entries(config, ["quantization_config"])
    remove_config_entries(config, ["pretrain_mm_mlp_adapter"])

    # 记录 dtype 与输出目录，方便推理框架识别这是完整模型而非 LoRA 适配器。
    dtype_name = str(next(model.parameters()).dtype).split(".")[-1]
    config.torch_dtype = dtype_name
    config._name_or_path = str(output_dir)

    if is_multimodal:
        config.model_type = "llava"
        config.architectures = ["LlavaLlamaAttForCausalLM"]
    else:
        config.model_type = "llama"


def save_checkpoint(model, tokenizer, model_base: str, output_dir: Path, max_shard_size: str) -> None:
    """保存合并后的模型权重、分词器资产以及生成配置。"""

    # safe_serialization=False 保持与官方发布一致的二进制权重格式。
    model.save_pretrained(
        output_dir,
        safe_serialization=False,
        max_shard_size=max_shard_size,
    )

    tokenizer.save_pretrained(output_dir)

    base_generation_config = Path(model_base) / "generation_config.json"
    if base_generation_config.exists():
        # 拷贝基座的生成配置，保证推理脚本仍能读取默认生成参数。
        shutil.copy2(base_generation_config, output_dir / "generation_config.json")


if __name__ == "__main__":
    main()
