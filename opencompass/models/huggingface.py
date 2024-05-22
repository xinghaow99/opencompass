import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch_size: int,
    ):
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # compare the last len(stop) tokens
        lookback_ids_batch = input_ids[:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if done:
                continue
            self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


@MODELS.register_module()
class HuggingFace(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)
        print(self.tokenizer)
        print("tokenizer_kwargs: ", tokenizer_kwargs)
        print("self.pad_token_id: ", self.pad_token_id)
        print("self.tokenizer.pad_token_id: ", self.tokenizer.pad_token_id)
        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        min_out_len=min_out_len,
                                        stopping_criteria=stopping_criteria,
                                        **generation_kwargs)
        else:
            return sum(
                (self._single_generate(inputs=[input_],
                                       max_out_len=max_out_len,
                                       min_out_len=min_out_len,
                                       stopping_criteria=stopping_criteria,
                                       **generation_kwargs)
                 for input_ in inputs), [])

    def _batch_generate(self,
                        inputs: List[str],
                        max_out_len: int,
                        min_out_len: Optional[int] = None,
                        stopping_criteria: List[str] = [],
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            for i in range(len(inputs)):
                conv = get_conversation_template('vicuna')
                conv.append_message(conv.roles[0], inputs[i])
                conv.append_message(conv.roles[1], None)
                inputs[i] = conv.get_prompt()

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len -
                                                  max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          tokens['input_ids'].shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        return decodeds

    def _single_generate(self,
                         inputs: List[str],
                         max_out_len: int,
                         min_out_len: Optional[int] = None,
                         stopping_criteria: List[str] = [],
                         **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], inputs[0])
            conv.append_message(conv.roles[1], None)
            inputs = [conv.get_prompt()]

        if self.mode == 'mid':
            input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            if len(input_ids[0]) > self.max_seq_len - max_out_len:
                half = int((self.max_seq_len - max_out_len) / 2)
                inputs = [
                    self.tokenizer.decode(input_ids[0][:half],
                                          skip_special_tokens=True) +
                    self.tokenizer.decode(input_ids[0][-half:],
                                          skip_special_tokens=True)
                ]

        input_ids = self.tokenizer(inputs,
                                   truncation=True,
                                   max_length=self.max_seq_len -
                                   max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          input_ids.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # To accommodate the PeftModel, parameters should be passed in
        # key-value format for generate.
        outputs = self.model.generate(input_ids=input_ids,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        return decodeds

    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs[0], {'tokens': tokens}

    def get_cd_logits(self, inputs: List[str], amateur_layer_idx: int, cd_alpha: float, cd_beta: float):
        # print("getting cd logits with amateur layer idx: ", amateur_layer_idx, "cd_alpha: ", cd_alpha, "cd_beta: ", cd_beta)
        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens, output_hidden_states=True, return_dict=True)
        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        with torch.no_grad():
            expert_logits = outputs.logits
            amateur_layer_hidden_states = outputs.hidden_states[amateur_layer_idx]
            amateur_logits = self.model.lm_head(amateur_layer_hidden_states)
            if amateur_logits.device != expert_logits.device:
                amateur_logits = amateur_logits.to(expert_logits.device)

            cutoff = torch.log(cd_alpha*torch.ones_like(expert_logits)) + expert_logits.max(dim=-1, keepdim=True).values
            diffs = (1 + cd_beta)*expert_logits - cd_beta*amateur_logits
            diffs_min = diffs.min(dim=-1, keepdim=True).values
            cd_logits = torch.where(expert_logits < cutoff, diffs_min, diffs)
            same_pred_mask = torch.argmax(expert_logits, dim=-1, keepdim=True) == torch.argmax(amateur_logits, dim=-1, keepdim=True)
            same_pred_mask = same_pred_mask.expand_as(expert_logits)
            cd_logits = torch.where(same_pred_mask, expert_logits, cd_logits)
            # cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))

            # expert_probs = torch.softmax(expert_logits, dim=-1)
            # amateur_probs = torch.softmax(amateur_logits, dim=-1)
            # cd_logits = (1 + cd_beta)*expert_probs - cd_beta*(amateur_probs)


            # print("cd_logits: ", cd_logits)
            # print("expert_logits: ", expert_logits)
            # print("amateur_logits: ", amateur_logits)
        return cd_logits, {'tokens': tokens}
    def get_layer_entropy(self, layer_prob):
        eps = torch.finfo(layer_prob.dtype).tiny
        layer_prob = layer_prob.clamp(min=eps)
        layer_entropy = -torch.sum(layer_prob * torch.log(layer_prob), dim=-1)
        return layer_entropy
    
    def get_layer_entropy_log(self, layer_log_prob):
        layer_entropy = -torch.sum(layer_log_prob.exp() * layer_log_prob, dim=-1)
        return layer_entropy
    
    # def get_min_entropy_diff_layer(self, layer_prob):
    #     layer_entropy = self.get_layer_entropy(layer_prob)
    #     entropy_diff = torch.diff(layer_entropy, dim=0) # (num_layers-1, batch_size, seq_len)
    #     entropy_diff = entropy_diff.transpose(0, 1)
    #     start_layer = entropy_diff.argmin(dim=-1) + 1
    #     return start_layer
    def get_min_entropy_diff_layer(self, layer_prob):
        layer_entropy = self.get_layer_entropy(layer_prob)
        # print("layer_entropy shape:", layer_entropy.shape)  # Print shape of layer_entropy
        # print("layer_entropy: ", layer_entropy)
        entropy_diff = torch.diff(layer_entropy, dim=0).transpose(0, 1)  # (batch_size, num_layers-1, seq_len)
        # print("entropy_diff shape:", entropy_diff.shape)  # Print shape of entropy_diff
                
        start_layer = entropy_diff.argmin(dim=1) + 1
        # print("start_layer shape:", start_layer.shape)  # Print shape of start_layer
        return start_layer
    
    def get_min_entropy_diff_layer_log(self, layer_log_prob):
        # layer_entropy = self.get_layer_entropy_log(layer_log_prob)
        layer_entropy = -torch.sum(layer_log_prob.exp() * layer_log_prob, dim=-1)
        # print("layer_entropy shape:", layer_entropy.shape)  # Print shape of layer_entropy

        entropy_diff = torch.diff(layer_entropy, dim=0)  # (num_layers-1, batch_size, seq_len)
        # print("entropy_diff shape:", entropy_diff.shape)  # Print shape of entropy_diff
        
        entropy_diff = entropy_diff.transpose(0, 1)
        # print("transposed entropy_diff shape:", entropy_diff.shape)  # Print shape after transposing
        
        start_layer = entropy_diff.argmin(dim=1) + 1
        # print("start_layer shape:", start_layer.shape)  # Print shape of start_layer
        return start_layer
    
    def get_refined_probs(self, inputs: List[str], alpha: float, beta: float):
        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens, output_hidden_states=True, return_dict=True)
        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        with torch.no_grad():
            stacked_hidden_states = torch.stack(outputs.hidden_states, dim=0)
            # print("stacked_hidden_states shape:", stacked_hidden_states.shape)  # Print shape of stacked_hidden_states

            layer_logits = self.model.lm_head(stacked_hidden_states)
            # print("layer_logits shape:", layer_logits.shape)  # Print shape of layer_logits

            # layer_prob = torch.softmax(layer_logits, dim=-1)
            layer_prob = torch.nn.functional.log_softmax(layer_logits, dim=-1)
            # print("layer_prob shape:", layer_prob.shape)  # Print shape of layer_prob

            refine_start_layer_idx = self.get_min_entropy_diff_layer(layer_prob)
            # print("refine_start_layer_idx shape:", refine_start_layer_idx.shape)  # Print shape of refine_start_layer_idx
            # print("refine_start_layer_idx:", refine_start_layer_idx)  # Print value of refine_start_layer_idx
            num_layers = layer_prob.size(0)
            # print("num_layers:", num_layers)  # Print value of num_layers

            momentum = torch.zeros_like(layer_prob[-1], device=layer_prob.device)
            # print("momentum shape:", momentum.shape)  # Print shape of momentum

            # Assuming beta and alpha are defined elsewhere as scalars
            for t in range(1, num_layers):
                delta_p = layer_prob[t] - layer_prob[t-1]
                # print(f"delta_p shape at t={t}:", delta_p.shape)  # Print shape of delta_p at each time step

                mask = (torch.arange(num_layers, device=layer_prob.device).unsqueeze(0).unsqueeze(0) >= refine_start_layer_idx.unsqueeze(-1))
                # print(f"mask shape at t={t}:", mask.shape)  # Print shape of mask at each time step

                # Update momentum using the mask
                expanded_mask = mask[:, :, t].unsqueeze(-1).expand_as(momentum)
                # print(f"expanded_mask shape at t={t}:", expanded_mask.shape)  # Print shape of expanded_mask at each time step
                # print("expanded_mask:", expanded_mask)  # Print value of expanded_mask
                momentum = torch.where(expanded_mask, beta * momentum + (1 - beta) * delta_p, momentum)
                # print(f"momentum shape at t={t}:", momentum.shape)  # Print shape of momentum at each time step

            weight_factor = (num_layers - refine_start_layer_idx.float()) / num_layers
            # print("weight_factor shape:", weight_factor.shape)  # Print shape of weight_factor

            refined = layer_prob[-1] + alpha * (1 - weight_factor.unsqueeze(-1).expand_as(layer_prob[-1])) * momentum
            # print("refined shape:", refined.shape)  # Print shape of refined
        return refined, {'tokens': tokens}
    
    def get_refined_logits(self, inputs: List[str], alpha: float, beta: float):
        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens, output_hidden_states=True, return_dict=True)
        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        with torch.no_grad():
            stacked_hidden_states = torch.stack(outputs.hidden_states, dim=0)

            layer_logits = self.model.lm_head(stacked_hidden_states)

            layer_log_prob = torch.nn.functional.log_softmax(layer_logits, dim=-1)

            refine_start_layer_idx = self.get_min_entropy_diff_layer_log(layer_log_prob)
            num_layers = layer_logits.size(0)

            momentum = torch.zeros_like(layer_logits[-1], device=layer_logits.device)
            # weight_factor = (num_layers - refine_start_layer_idx.float()) / num_layers
            # weight_factor = (1 - weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1]))
            # beta = (1-weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1])) * beta
            # Assuming beta and alpha are defined elsewhere as scalars
            for t in range(1, num_layers):
                delta_p = layer_logits[t] - layer_logits[t-1]

                mask = (torch.arange(num_layers, device=layer_logits.device).unsqueeze(0).unsqueeze(0) >= refine_start_layer_idx.unsqueeze(-1))

                # Update momentum using the mask
                expanded_mask = mask[:, :, t].unsqueeze(-1).expand_as(momentum)
                momentum = torch.where(expanded_mask, beta * momentum + (1 - beta) * delta_p, momentum)

            # weight_factor = (num_layers - refine_start_layer_idx.float()) / num_layers
            # weight_factor = (1 - weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1]))
            # refined = layer_log_prob[-1] + alpha * weight_factor * momentum
            refined = layer_logits[-1] + alpha * momentum
        return refined, {'tokens': tokens}

    def get_refined_log_probs(self, inputs: List[str], alpha: float, beta: float):
        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens, output_hidden_states=True, return_dict=True)
        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        with torch.no_grad():
            # stacked_hidden_states = torch.stack(outputs.hidden_states, dim=0)

            # layer_logits = self.model.lm_head(torch.stack(outputs.hidden_states, dim=0))

            layer_log_prob = torch.nn.functional.log_softmax(self.model.lm_head(torch.stack(outputs.hidden_states, dim=0)), dim=-1)

            # refine_start_layer_idx = self.get_min_entropy_diff_layer_log(layer_log_prob)
            # layer_entropy = -torch.sum(layer_log_prob.exp() * layer_log_prob, dim=-1)
            layer_entropy = []

            # Iterate along the first dimension
            for log_prob in layer_log_prob:
                entropy = -torch.sum(log_prob.exp() * log_prob, dim=-1)
                layer_entropy.append(entropy)

            # Converting the list back to a tensor
            layer_entropy = torch.stack(layer_entropy)
            
            entropy_diff = torch.diff(layer_entropy, dim=0).transpose(0, 1)
        
            refine_start_layer_idx = entropy_diff.argmin(dim=1) + 1
            
            num_layers = layer_log_prob.size(0)

            momentum = torch.zeros_like(layer_log_prob[-1], device=layer_log_prob.device)
            # weight_factor = (num_layers - refine_start_layer_idx.float()) / num_layers
            # weight_factor = (1 - weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1]))
            # beta = (1-weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1])) * beta
            # Assuming beta and alpha are defined elsewhere as scalars
            for t in range(1, num_layers):
                delta_p = layer_log_prob[t].exp()  - layer_log_prob[t-1].exp()

                mask = (torch.arange(num_layers, device=layer_log_prob.device).unsqueeze(0).unsqueeze(0) >= refine_start_layer_idx.unsqueeze(-1))

                # Update momentum using the mask
                expanded_mask = mask[:, :, t].unsqueeze(-1).expand_as(momentum)
                momentum = torch.where(expanded_mask, beta * momentum + (1 - beta) * delta_p, momentum)

            # weight_factor = (num_layers - refine_start_layer_idx.float()) / num_layers
            # weight_factor = (1 - weight_factor.unsqueeze(-1).expand_as(layer_log_prob[-1]))
            # refined = layer_log_prob[-1] + alpha * weight_factor * momentum
            refined = layer_log_prob[-1].exp() + alpha * momentum
            refined = refined.clamp(min=torch.finfo(refined.dtype).tiny)
            refined /= refined.sum(dim=-1, keepdim=True)
            refined = torch.log(refined)
        return refined, {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None,
                **kwargs) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length, **kwargs)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length, **kwargs)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None,
                 **kwargs) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        amateur_layer_idx = kwargs.get('amateur_layer_idx', None)
        cd_alpha = kwargs.get('cd_alpha', None)
        cd_beta = kwargs.get('cd_beta', None)
        if isinstance(amateur_layer_idx, int):
            outputs, inputs = self.get_cd_logits(inputs, amateur_layer_idx, cd_alpha, cd_beta)
        elif isinstance(amateur_layer_idx, str):
            assert amateur_layer_idx in ["auto_refine", "auto_refine_log", "auto_refine_logits"]
            if amateur_layer_idx == "auto_refine":
                outputs, inputs = self.get_refined_probs(inputs, cd_alpha, cd_beta)
                raise NotImplementedError()
            elif amateur_layer_idx == "auto_refine_log":
                outputs, inputs = self.get_refined_log_probs(inputs, cd_alpha, cd_beta)
                shift_log_probs = outputs[..., :-1, :].contiguous().float()
                shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous().to(shift_log_probs.device)
                loss_fct = torch.nn.NLLLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(shift_log_probs.view(-1, shift_log_probs.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())
            elif amateur_layer_idx == "auto_refine_logits":
                outputs, inputs = self.get_refined_logits(inputs, cd_alpha, cd_beta)
                raise NotImplementedError()
        else:
            outputs, inputs = self.get_logits(inputs)
            shift_logits = outputs[..., :-1, :].contiguous().float()

            shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous().to(shift_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
        # print("loss: ", loss)
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get loglikelihood scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            conts (List[str]): A list of strings: slices after the space.
            NOT SUPPORT mask_length YET!
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of loglikelihood scores.
        """
        assert mask_length is None, 'Not support mask_length yet.'
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_loglikelihood(inputs, conts)
        else:
            return np.concatenate([
                self._get_loglikelihood(inputs=[inputs[idx]],
                                        conts=[conts[idx]])
                for idx in range(len(inputs))
            ])

    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))


@MODELS.register_module()
class HuggingFaceCausalLM(HuggingFace):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False


class HuggingFaceChatGLM3(HuggingFace):
    """Model wrapper around HuggingFace's ChatGLM3. Details available in
    `https://huggingface.co/THUDM/chatglm3-6b`.

    model.chat() is used for inference.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 num_extra_tokens: int = 50):
        super().__init__(path=path,
                         hf_cache_dir=hf_cache_dir,
                         max_seq_len=max_seq_len,
                         tokenizer_path=tokenizer_path,
                         tokenizer_kwargs=tokenizer_kwargs,
                         peft_path=peft_path,
                         tokenizer_only=tokenizer_only,
                         generation_kwargs=generation_kwargs,
                         model_kwargs=model_kwargs,
                         meta_template=meta_template,
                         extract_pred_after_decode=extract_pred_after_decode,
                         batch_padding=batch_padding,
                         pad_token_id=pad_token_id,
                         mode=mode)
        self.template_parser = APITemplateParser(meta_template)
        # used to compensate for #tokens occupied by sth like system prompt
        self.num_extra_tokens = num_extra_tokens

    def generate(self,
                 inputs: List[str or PromptList],
                 max_out_len: int = 512,
                 skip_overlength=False,
                 **kwargs) -> str:
        """Generate response from input prompt.

        Args:
            inputs (list): input prompt
            max_out_len (int): max output length
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)

        responses = []
        for _input in inputs:
            assert isinstance(_input, (str, PromptList))
            if isinstance(_input, str):
                history = [{'role': 'user', 'content': _input}]
            else:
                history = []
                for item in _input:
                    msg = {
                        'content': item['prompt'],
                        'role': {
                            'HUMAN': 'user',
                            'BOT': 'assistant',
                            'SYSTEM': 'system',
                        }[item['role'].upper()]
                    }
                    history.append(msg)
            user_content = history[-1]['content']
            history = history[:-1]

            if skip_overlength:
                # The model will report the following error
                # if the sequence length is greater than the maximum length:
                # "Input length of input_ids is {INPUT_IDS},
                # but `max_length` is set to 8192.
                # This can lead to unexpected behavior.
                # You should consider increasing `max_new_tokens`."
                # The following hardcode can fix this exception.
                len_user_content = len(self.tokenizer.encode(user_content))
                if len_user_content > 8192:
                    responses.append('')
                    continue

            response, history = self.model.chat(self.tokenizer,
                                                user_content,
                                                history=history,
                                                max_new_tokens=max_out_len,
                                                **generation_kwargs)
            # response will be dict sometime
            if isinstance(response, dict):
                response = response.get('content', '')
            responses.append(response)
        return responses

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt)) + self.num_extra_tokens
