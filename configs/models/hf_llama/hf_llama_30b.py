from opencompass.models import HuggingFaceCausalLM

models = [
    # LLaMA 30B
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-30b-hf',
        path="/remote-home/share/models/llama_hf/30B",
        tokenizer_path='/remote-home/share/models/llama_hf/30B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        pad_token_id=0,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=3,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
