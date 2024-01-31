from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset, strategyqa_pred_postprocess, strategyqa_dataset_postprocess


strategyqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test')

strategyqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='''Q: Do hamsters provide food for any animals?
A: Let's think step by step. Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Let's think step by step. Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.

Q: Hydrogen's atomic number squared exceeds number of Spice Girls?
A: Let's think step by step. Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no.

Q: Is it common to see frost during some college commencements?
A: Let's think step by step. College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.

Q: Could a llama birth twice during War in Vietnam (1945-46)?
A: Let's think step by step. The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.

Q: Would a pear sink in water?
A: Let's think step by step. The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.

Q: {question}'''),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128, amateur_layer_idx='auto_refine', cd_alpha=5, cd_beta=0.99))

strategyqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=strategyqa_pred_postprocess),
    dataset_postprocessor=dict(type=strategyqa_dataset_postprocess))

strategyqa_datasets = [
    dict(
        abbr='strategyqa-QA=format-refine-alpha-5-beta-099',
        type=HFDataset,
        path='wics/strategy-qa',
        reader_cfg=strategyqa_reader_cfg,
        infer_cfg=strategyqa_infer_cfg,
        eval_cfg=strategyqa_eval_cfg)
]

    