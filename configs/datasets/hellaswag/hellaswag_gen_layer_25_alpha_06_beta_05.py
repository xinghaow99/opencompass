from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import hellaswagDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess

hellaswag_reader_cfg = dict(
    input_columns=["ctx", "A", "B", "C", "D"],
    output_column="label",
    test_split="validation")

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=("{ctx}\nQuestion: Which ending makes the most sense?\n"
                        "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
                        "You may choose from 'A', 'B', 'C', 'D'.\n"
                        "Answer:"),
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever, ),
    inferencer=dict(type=GenInferencer, max_out_len=1, amateur_layer_idx=25, cd_alpha=0.6, cd_beta=0.5),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        type=hellaswagDataset_V2,
        path="hellaswag",
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
