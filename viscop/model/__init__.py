# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from .projector import load_mm_projector

from .viscop_qwen2 import ViSCoP_Qwen2ForCausalLM
from .viscop_qwen2 import ViSCoP_Qwen2Config

VLLMs = {
    "viscop_qwen2": ViSCoP_Qwen2ForCausalLM
}

VLLMConfigs = {
    "viscop_qwen2": ViSCoP_Qwen2Config
}


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None

    # NOTE: auto device_map by default
    # if want to put model into a single device, you can set device_map={"": "cuda:0"}
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    assert 'num_visual_probes' in config, 'num_visual_probes not specified in model config, you may be trying to load a model other than ViSCoP.'

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    # judge model type
    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "viscop_qwen2")

    # judge pretrain/finetune
    is_alignment = getattr(config, "tune_mm_mlp_adapter", False) or getattr(config, "is_alignment", False)
    print('Only projector tuned? ', is_alignment)

    # NOTE: lora/qlora model loading
    if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        # NOTE: remove qlora training quantization config 
        if hasattr(cfg_pretrained, 'quantization_config'):
            del cfg_pretrained.quantization_config
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)
        print('Loading base VLM...')

        if 'qwen2' in model_base.lower():
            model = ViSCoP_Qwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = ViSCoP_Qwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise FileNotFoundError(f'The passed model was trained with LoRA, but non_lora_trainables.bin not found in {model_path}. Please make sure the model was trained with lora.')
        
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        _ = model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None or '-base' in model_name.lower() or is_alignment:
        # NOTE: Base/Pretrain model loading
        print('Loading base VLM followed by tuned projector weights.')
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)

        model = ViSCoP_Qwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        # NOTE; loading vision-language projector
        mm_projector_weights = load_mm_projector(model_path, token=token)
        _ = model.load_state_dict(mm_projector_weights, strict=False)
    elif 'viscop' in model_type:
        # NOTE: SFT model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        model = ViSCoP_Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    processor = None

    if "viscop" in model_type:
        vision_encoder = model.get_vision_encoder()
        processor = vision_encoder.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
