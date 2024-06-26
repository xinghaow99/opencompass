from opencompass.openicl.icl_prompt_template import PromptTemplate
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
        template={
            i: dict(round=[
                dict(role="HUMAN", prompt="{ctx}"),
                dict(role="BOT", prompt=f"{{{chr(ord('A') + i)}}}"),
            ])
            for i in range(4)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, amateur_layer_idx="auto_refine", cd_alpha=0.5, cd_beta=0.9))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets = [
    dict(
        abbr="hellaswag-ppl-refine",
        type=hellaswagDataset,
        path='hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
