from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets import XsumDataset

Xsum_reader_cfg = dict(input_columns=["dialogue"], output_column="summary")

Xsum_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=
                "Document: {dialogue}\nBased on the previous text, provide a brief single summary:"
            ),
            dict(role="BOT", prompt="{summary}\n")
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
                "Document: {dialogue}\nBased on the previous text, provide a brief single summary:"
            ),
        ]),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever, ice_num=3, fix_id_list=[0, 1, 2]),
    inferencer=dict(type=GenInferencer, amateur_layer_idx='auto_refine', cd_alpha=1, cd_beta=0.4),
)

Xsum_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type="Xsum"),
)

Xsum_datasets = [
    dict(
        type=XsumDataset,
        abbr="Xsum",
        path="./data/Xsum/dev.jsonl",
        reader_cfg=Xsum_reader_cfg,
        infer_cfg=Xsum_infer_cfg,
        eval_cfg=Xsum_eval_cfg,
    )
]
