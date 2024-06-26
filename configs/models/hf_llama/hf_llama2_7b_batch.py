from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-7b-hf',
        path="/remote-home/share/models/llama_v2_hf/7b",
        tokenizer_path='/remote-home/share/models/llama_v2_hf/7b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              
                              ),
        pad_token_id=0,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=3,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
