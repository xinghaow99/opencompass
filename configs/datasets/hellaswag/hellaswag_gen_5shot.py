from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import hellaswagDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess
hellaswag_reader_cfg = dict(
    input_columns=["ctx", "A", "B", "C", "D"],
    output_column="label",
    test_split="validation")

hellaswag_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=("{ctx}\nQuestion: Which ending makes the most sense?\n"
                        "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
                        "You may choose from 'A', 'B', 'C', 'D'.\n"
                        "Answer: "),
            ),
            dict(role="BOT", prompt="{label}\n")
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
            dict(
                role="HUMAN",
                prompt=("{ctx}\nQuestion: Which ending makes the most sense?\n"
                        "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
                        "You may choose from 'A', 'B', 'C', 'D'.\n"
                        "Answer: "),
            ),
        ]),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever, ice_num=5, fix_id_list=[1, 3, 5, 7, 9]),
    inferencer=dict(type=GenInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr="hellaswag-gen-5shot",
        type=hellaswagDataset_V2,
        path="hellaswag",
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
