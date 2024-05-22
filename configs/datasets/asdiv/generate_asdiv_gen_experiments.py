import os

# Set the working directory to be the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generate_script(amateur_layer_idx, cd_alpha, cd_beta):
    template = f"""from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset, asdiv_postprocess, asdiv_dataset_postprocess, ASDivEvaluator

asdiv_reader_cfg = dict(input_columns=['body', 'question'], output_column='answer')

asdiv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: Sarah has 15 apples. She gives 7 to her friend and then buys 10 more. How many apples does she have now?\\nAnswer:"),
                dict(role='BOT', prompt="Sarah starts with 15 apples. If she gives 7 to her friend, she is left with 15 apples - 7 apples = 8 apples. Later, Sarah buys 10 more apples. Therefore, her new total is her previous total of 8 apples + the 10 additional apples she bought = 18 apples. The answer is 18\\n"),

                dict(role='HUMAN', prompt="Question: A bookshop has 120 books. If 30% of them are fiction, how many fiction books are there?\\nAnswer:"),
                dict(role='BOT', prompt="In the bookshop, there are a total of 120 books. To find out how many of these are fiction, we calculate 30% of 120 books. To do this, we multiply 120 by 30%, or 0.30. So, the calculation is 120 books × 0.30 = 36 books. Therefore, there are 36 fiction books in the bookshop. The answer is 36\\n"),

                dict(role='HUMAN', prompt="Question: A classroom has 5 rows of chairs with 8 chairs in each row. If 3 chairs are removed, how many chairs are left?\\nAnswer:"),
                dict(role='BOT', prompt="Initially, the classroom contains 5 rows of chairs, with each row having 8 chairs. To find the total number of chairs, we multiply the number of rows by the number of chairs per row: 5 rows × 8 chairs/row = 40 chairs. If 3 chairs are then removed from this total, we subtract 3 from 40, which gives us 40 chairs - 3 chairs = 37 chairs. Therefore, there are 37 chairs left in the classroom. The answer is 37\\n"),

                dict(role='HUMAN', prompt="Question: If a train travels 300 miles in 5 hours, at the same speed, how long will it take to travel 180 miles?\\nAnswer:"),
                dict(role='BOT', prompt="The train covers a distance of 300 miles in 5 hours. To find the speed of the train, we divide the total distance by the total time taken. So, the train's speed is 300 miles / 5 hours = 60 miles per hour (mph). To find out how long it will take to travel 180 miles at this speed, we divide the distance by the speed: 180 miles / 60 mph. This gives us a travel time of 3 hours for the train to cover 180 miles. The answer is 3\\n"),

                dict(role='HUMAN', prompt="Question: {{body}} {{question}}\\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256, amateur_layer_idx="{layer}", cd_alpha={cd_alpha}, cd_beta={cd_beta}))

asdiv_eval_cfg = dict(evaluator=dict(type=ASDivEvaluator),
                      pred_postprocessor=dict(type=asdiv_postprocess),
                      dataset_postprocessor=dict(type=asdiv_dataset_postprocess))
asdiv_datasets = [
    dict(
        abbr='asdiv-{layer}-alpha-{str(cd_alpha).replace('.', '')}-beta-{str(cd_beta).replace('.', '')}',
        type=HFDataset,
        path='EleutherAI/asdiv',
        name='asdiv',
        split='validation',
        reader_cfg=asdiv_reader_cfg,
        infer_cfg=asdiv_infer_cfg,
        eval_cfg=asdiv_eval_cfg)
]
    """

    file_name = f"asdiv_gen_{layer}_alpha_{str(cd_alpha).replace('.', '')}_beta_{str(cd_beta).replace('.', '')}.py"
    with open(file_name, "w") as f:
        f.write(template)

    print(f"File '{file_name}' has been generated!")

# Example of how to call the function:
# generate_script(25, 0.6, 0.5)
import numpy as np
if __name__ == "__main__":
    for layer in ["refine_norm"]:
        for alpha in [1]:
            for beta in [0.9]:
                generate_script(layer, alpha, beta)