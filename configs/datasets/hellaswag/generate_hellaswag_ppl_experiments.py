import os

# Set the working directory to be the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generate_script(amateur_layer_idx, cd_alpha, cd_beta):
    template = f"""from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import hellaswagDataset

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_split='validation',
    test_split='validation')

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={{
            i: dict(round=[
                dict(role="HUMAN", prompt="{{ctx}}"),
                dict(role="BOT", prompt=f"{{{{{{chr(ord('A') + i)}}}}}}"),
            ])
            for i in range(4)
        }}),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, amateur_layer_idx={amateur_layer_idx}, cd_alpha={cd_alpha}, cd_beta={cd_beta}))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets = [
    dict(
        type=hellaswagDataset,
        path='hellaswag',
        abbr='hellaswag-layer-{amateur_layer_idx}-alpha-{str(cd_alpha).replace('.', '')}-beta-{str(cd_beta).replace('.', '')}',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
    """

    file_name = f"hellaswag_ppl_layer_{amateur_layer_idx}_alpha_{str(cd_alpha).replace('.', '')}_beta_{str(cd_beta).replace('.', '')}.py"
    with open(file_name, "w") as f:
        f.write(template)

    print(f"File '{file_name}' has been generated!")

# Example of how to call the function:
# generate_script(25, 0.6, 0.5)

if __name__ == "__main__":
    for layer in [10]:
        for alpha in [0]:
            for beta in [0, 0.25, 0.5]:
                generate_script(layer, alpha, beta)