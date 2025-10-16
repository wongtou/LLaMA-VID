from .language_model.llava_llama_vid import LlavaLlamaAttForCausalLM as LlavaVIDForCausalLM
from .language_model.llava_navid import LlavaLlamaAttForCausalLM as NaVidForCausalLM

MODEL_ARCH_REGISTRY = {
    "navid": NaVidForCausalLM,
    "vid": LlavaVIDForCausalLM,
}


def resolve_model_architecture(model_name: str):
    name = model_name.lower()
    for key, cls in MODEL_ARCH_REGISTRY.items():
        if key in name:
            return cls
    return LlavaVIDForCausalLM


LlavaLlamaAttForCausalLM = LlavaVIDForCausalLM
