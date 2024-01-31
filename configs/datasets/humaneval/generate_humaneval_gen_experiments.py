import os

# Set the working directory to be the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generate_script(amateur_layer_idx, cd_alpha, cd_beta):
    template = f"""from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset, HumanEvaluator, humaneval_postprocess

humaneval_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\\n{{prompt}}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512, amateur_layer_idx="auto_refine", cd_alpha={cd_alpha}, cd_beta={cd_beta}))

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess),
)

humaneval_datasets = [
    dict(
        type=HFDataset,
        abbr='humaneval-refine-alpha-{str(cd_alpha).replace('.', '')}-beta-{str(cd_beta).replace('.', '')}',
        path='openai_humaneval',
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg)
]
    """

    file_name = f"humaneval_gen_refine_alpha_{str(cd_alpha).replace('.', '')}_beta_{str(cd_beta).replace('.', '')}.py"
    with open(file_name, "w") as f:
        f.write(template)

    print(f"File '{file_name}' has been generated!")

# Example of how to call the function:
# generate_script(25, 0.6, 0.5)
import numpy as np
if __name__ == "__main__":
    for layer in ["auto"]:
        for alpha in [0.1, 0.5, 1, 1.5, 2]:
            for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                generate_script(layer, alpha, beta)