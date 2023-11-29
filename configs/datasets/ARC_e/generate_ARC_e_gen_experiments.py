import os

# Set the working directory to be the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generate_script(amateur_layer_idx, cd_alpha, cd_beta):
    template = f"""from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

ARC_e_reader_cfg = dict(
    input_columns=["question", "textA", "textB", "textC", "textD"],
    output_column="answerKey")

ARC_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "Question: {{question}}\\nA. {{textA}}\\nB. {{textB}}\\nC. {{textC}}\\nD. {{textD}}\\nAnswer:"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, amateur_layer_idx={amateur_layer_idx}, cd_alpha={cd_alpha}, cd_beta={cd_beta}),
)

ARC_e_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

ARC_e_datasets = [
    dict(
        abbr="ARC-e-layer-{amateur_layer_idx}-alpha-{str(cd_alpha).replace('.', '')}-beta-{str(cd_beta).replace('.', '')}",
        type=ARCDataset,
        path="./data/ARC/ARC-e/ARC-Easy-Dev.jsonl",
        reader_cfg=ARC_e_reader_cfg,
        infer_cfg=ARC_e_infer_cfg,
        eval_cfg=ARC_e_eval_cfg,
    )
]

    """

    file_name = f"ARC_e_gen_layer_{amateur_layer_idx}_alpha_{str(cd_alpha).replace('.', '')}_beta_{str(cd_beta).replace('.', '')}.py"
    with open(file_name, "w") as f:
        f.write(template)

    print(f"File '{file_name}' has been generated!")

# Example of how to call the function:
# generate_script(25, 0.6, 0.5)
import numpy as np
if __name__ == "__main__":
    for layer in ["auto"]:
        for alpha in [0]:
            for beta in [0.25]:
                generate_script(layer, alpha, beta)