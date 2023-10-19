from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-70b-hf',
        path="/remote-home/share/llama_v2_hf/70b",
        tokenizer_path='/remote-home/share/llama_v2_hf/70b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              pad_token='[PAD]'
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
