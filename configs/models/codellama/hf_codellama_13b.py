from opencompass.models import HuggingFaceCausalLM

models = [
    # CodeLlama 13B
    dict(
        type=HuggingFaceCausalLM,
        abbr='CodeLlama-13b',
        path="/remote-home/share/codellama/hf_models/codellama/CodeLlama-13b-hf",
        tokenizer_path='/remote-home/share/codellama/hf_models/codellama/CodeLlama-13b-hf',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        batch_padding=True,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
