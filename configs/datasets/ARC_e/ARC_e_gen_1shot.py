from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

_hint = "Answer the question by replying A, B, C or D."

ARC_e_reader_cfg = dict(
    input_columns=["question", "textA", "textB", "textC", "textD"],
    output_column="answerKey")

ARC_e_infer_cfg = dict(
    ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role="HUMAN",
                    prompt=
                    f"{_hint}\nQuestion: {{question}}\nA. {{textA}}\nB. {{textB}}\nC. {{textC}}\nD. {{textD}}\nAnswer:"
                ),
                dict(role="BOT", prompt="{target}\n")
            ]),
        ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:"
                )
            ], ),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever, ice_num=1, fix_id_list=[0]),
    inferencer=dict(type=GenInferencer),
)

ARC_e_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

ARC_e_datasets = [
    dict(
        abbr="ARC-e-1shot",
        type=ARCDataset,
        path="./data/ARC/ARC-e/ARC-Easy-Dev.jsonl",
        reader_cfg=ARC_e_reader_cfg,
        infer_cfg=ARC_e_infer_cfg,
        eval_cfg=ARC_e_eval_cfg,
    )
]
