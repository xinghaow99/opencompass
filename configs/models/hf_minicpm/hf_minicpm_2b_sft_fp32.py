from opencompass.models import HuggingFaceCausalLM
import torch

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='minicpm-2b-sft-fp32',
        path="openbmb/MiniCPM-2B-sft-fp32",
        tokenizer_path='openbmb/MiniCPM-2B-sft-fp32',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
