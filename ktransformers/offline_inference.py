"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor

import argparse
from pathlib import Path
from contextlib import nullcontext
import triton.profiler as proton

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_prefill_size: int = 8192,
    # Add profiler related arguments
    profile: bool = False,
    profile_note: str = None,
    profiler_context: str = "python",
    profiler_backend: str = None,
    profiler_hook: str = "triton",
    profile_result_dir: str = None
):

    torch.set_grad_enabled(False)
    if use_cuda_graph:
        torch.profiler._utils._init_for_cuda_graphs()

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    content = """在遥远的翡翠森林里，住着各种各样的神奇生物。其中，有一只名叫露露的小狐狸，她与其他狐狸不同，天生长着一双晶莹剔透的翅膀。然而，这双翅膀却从未带她飞翔过。
    一天，森林里传来一个惊人的消息：藏在森林深处的魔法泉水干涸了，所有生物赖以生存的泉水即将枯竭。他们说，只有传说中的“天空之羽”才能唤醒泉水，让它重新流淌。然而，“天空之羽”藏在一座高耸入云的山峰上，没有任何动物能抵达那里。
    露露听到这个消息后，决定亲自去寻找“天空之羽”，即便她的翅膀无法飞翔，她也要尝试。最终，露露来到了传说中的高峰脚下，根本无法攀爬。她望着天空，心里充满了不甘：“如果我能飞起来，就不会被这座山挡住了……”
    正当她感到迷茫时，一只年迈的白鹰出现在她面前。
    “孩子，你为什么到这里来？”白鹰用苍老但慈祥的声音问道。
    露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。
    在遥远的翡翠森林里，住着各种各样的神奇生物。其中，有一只名叫露露的小狐狸，她与其他狐狸不同，天生长着一双晶莹剔透的翅膀。然而，这双翅膀却从未带她飞翔过。
    一天，森林里传来一个惊人的消息：藏在森林深处的魔法泉水干涸了，所有生物赖以生存的泉水即将枯竭。他们说，只有传说中的“天空之羽”才能唤醒泉水，让它重新流淌。然而，“天空之羽”藏在一座高耸入云的山峰上，没有任何动物能抵达那里。
    露露听到这个消息后，决定亲自去寻找“天空之羽”，即便她的翅膀无法飞翔，她也要尝试。最终，露露来到了传说中的高峰脚下，根本无法攀爬。她望着天空，心里充满了不甘：“如果我能飞起来，就不会被这座山挡住了……”
    正当她感到迷茫时，一只年迈的白鹰出现在她面前。
    “孩子，你为什么到这里来？”白鹰用苍老但慈祥的声音问道。
    露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。
        露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。想。
    请简述这个故事的内涵 故事的内涵这个故事的内涵写10000个字"""        
    messages = [{"role": "user", "content": content}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if force_think:
        token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
        input_tensor = torch.cat(
            [input_tensor, token_thinks], dim=1
        )
    if mode == 'long_context':
        assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
        "please change max_seq_len in  ~/.ktransformers/config.yaml"

    session_id = None
    if profile:
        if profile_result_dir:
            profile_result_path = Path(profile_result_dir)
            profile_result_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
            proton_name = str(profile_result_path / model_path.split('/')[-1])
        else:
            proton_name = model_path.split('/')[-1]
        if profile_note:
            proton_name = f"{proton_name}-{profile_note}"
        if use_cuda_graph:
            run_mode = "-cudagraph"
        else:
            run_mode = "-eager"
        proton_name = proton_name + run_mode + f"-ctx_{profiler_context}" + f"-backend_{profiler_backend}" + f"-hook_{profiler_hook}"
        print(f"Profile name: {proton_name}")
        session_id = proton.start(
                name=proton_name, 
                context=profiler_context,
                backend=profiler_backend,
                hook=profiler_hook
        )

    if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8 and device_manager.gpu_vendor == GPUVendor.NVIDIA:
        with proton.cpu_timed_scope("prefill_and_generate") if profile else nullcontext():
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
    else:
        with proton.cpu_timed_scope("prefill_and_generate") if profile else nullcontext():
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
            )
    
    if profile:
        proton.finalize(session_id)


if __name__ == "__main__":
    # fire.Fire(local_chat)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-path', type=str, default='deepseek-ai/DeepSeek-V2-Lite-Chat')
    # parser.add_argument('--gguf-path',  type=str, default="./local_dev/DeepSeek-V2-Lite-Chat-GGUF")
    parser.add_argument('--model-path', type=str, default='deepseek-ai/DeepSeek-V3')
    parser.add_argument('--gguf-path',  type=str, default="./local_dev/DeepSeek-V3-GGUF")
    parser.add_argument('--use-cuda-graph', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--optimize-config-path', type=str, default=None)
    parser.add_argument('--cpu-infer', type=int, default=Config().cpu_infer)
    parser.add_argument('--prompt-file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="normal", choices=["normal", "long_context"])
    parser.add_argument('--force-think', type=bool, default=False)
    parser.add_argument('--chunk-prefill-size', type=int, default=8192)

    # Proton related arguments
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-note',
        type=str,
        default=None,
        help=('custom note for this profile. Will be reflected in profile file name'))
    parser.add_argument(
        '--profiler-context',
        type=str,
        default="shadow",
        help=('Proton context. Can be shadow or python. By default shadow'))
    parser.add_argument(
        '--profiler-backend',
        type=str,
        default=None,
        help=('Proton backend. use "cupti_pcsampling" for instruction sampling. By default auto select'))
    parser.add_argument(
        '--profiler-hook',
        type=str,
        default="triton",
        help=('Proton hook. Currently only "triton" available. By default triton.'))
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the triton profiler output.'))

    args = parser.parse_args()
    local_chat(
        model_path=args.model_path,
        optimize_config_path=args.optimize_config_path,
        gguf_path=args.gguf_path,
        max_new_tokens=args.max_new_tokens,
        cpu_infer=args.cpu_infer,
        use_cuda_graph=args.use_cuda_graph,
        prompt_file=args.prompt_file,
        mode=args.mode,
        force_think=args.force_think,
        chunk_prefill_size=args.chunk_prefill_size,
        profile=args.profile,
        profile_note=args.profile_note,
        profiler_context=args.profiler_context,
        profiler_backend=args.profiler_backend,
        profiler_hook=args.profiler_hook,
        profile_result_dir=args.profile_result_dir
    )
