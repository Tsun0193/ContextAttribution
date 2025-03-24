import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import torch as ch
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .context_partitioner import *
from .solver import *
from .utils import *

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"
SOLVERS = {
    "lasso": LassoRegression, 
    "polynomial": PolynomialLassoRegression,
    "mlp": MLPRegression,
    "random_forest": RandomForestRegression,
    "grad_boosting": GradientBoostingRegression,
    "svr": SVRRegression,
    "elastic_net": ElasticNetRegression
}

class ContextCiter:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        solver: Optional[BaseSolver] = None,
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        partitioner: Optional[BaseContextPartitioner] = None,
        quantized: bool = False  # new optional flag
    ) -> None:
        """
        Initializes a new instance of the ContextCiter class, which automates the process of:
        1) splitting a context into multiple sources,
        2) generating a response from an LLM using the query and context,
        3) attributing which sources contributed the most to the response.
        If quantized is True, the model is assumed to be loaded in 8-bit quantized mode.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.query = query
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.num_ablations = num_ablations
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver
        if self.solver is None:
            self.solver = LassoRegression()
        else:
            self.solver = SOLVERS[solver]()
        self.prompt_template = prompt_template
        self.quantized = quantized  # store quantized flag

        # Initialize the partitioner
        if partitioner is None:
            self.partitioner = SentencePeriodPartitioner(self.context)
        else:
            self.partitioner = partitioner()
            if self.partitioner.context != self.context:
                raise ValueError("Partitioner context does not match provided context.")

        # Preprocess the context and split it into sources
        self.partitioner.split_context()

        self._cache: Dict[str, Any] = {}

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        context: str,
        query: str,
        solver: str = "lasso",
        device: str = "cuda",
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        quantized: bool = False,  # new optional argument
        **kwargs: Dict[str, Any],
    ) -> "ContextCiter":
        """
        Load a ContextCiter instance from a pretrained model.
        If quantized is True, load the model in 8-bit mode.
        """
        if quantized:
            # Load the model in 8-bit quantized mode (requires bitsandbytes)
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                load_in_8bit=True,
                device_map="auto",
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, **model_kwargs
            )
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        tokenizer.padding_side = "left"
        return cls(model, tokenizer, context, query, solver, **kwargs, quantized=quantized)

    def _get_prompt_ids(
        self,
        mask: Optional[NDArray] = None,
        return_prompt: bool = False,
    ):
        # Cache the unmasked prompt tokens so that repeated calls are faster.
        if mask is None and not return_prompt and "prompt_ids" in self._cache:
            return self._cache["prompt_ids"]

        context = self.partitioner.get_context(mask)
        prompt = self.prompt_template.format(context=context, query=self.query)
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ems = self.tokenizer(chat_prompt, add_special_tokens=False, return_attention_mask=True)

        chat_prompt_ids = ems["input_ids"]
        attention_mask = ems["attention_mask"]

        if mask is None and not return_prompt:
            self._cache["prompt_ids"] = chat_prompt_ids
            return chat_prompt_ids
        if return_prompt:
            return chat_prompt_ids, attention_mask, chat_prompt
        return chat_prompt_ids

    @property
    def _response_start(self) -> int:
        prompt_ids = self._get_prompt_ids()
        return len(prompt_ids)

    @property
    def _output(self) -> str:
        if self._cache.get("output") is None:
            prompt_ids, attention_mask, prompt = self._get_prompt_ids(return_prompt=True)
            input_ids = ch.tensor([prompt_ids], device=self.model.device)
            attention_mask = ch.tensor([attention_mask], device=self.model.device)

            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generate_kwargs,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]
            # We take the original prompt because sometimes encoding and decoding changes it.
            raw_output = self.tokenizer.decode(output_ids)
            prompt_length = len(self.tokenizer.decode(prompt_ids))
            self._cache["output"] = prompt + raw_output[prompt_length:]
        return self._cache["output"]

    @property
    def _output_tokens(self) -> Dict[str, Any]:
        # Cache the tokenized output so that it is computed only once.
        if "output_tokens" not in self._cache:
            self._cache["output_tokens"] = self.tokenizer(self._output, add_special_tokens=False)
        return self._cache["output_tokens"]

    @property
    def _response_ids(self) -> List[int]:
        return self._output_tokens["input_ids"][self._response_start :]

    @property
    def response(self) -> str:
        """
        The response generated by the model (excluding the prompt). This property is cached.
        """
        output_tokens = self._output_tokens
        char_response_start = output_tokens.token_to_chars(self._response_start).start
        response = self._output[char_response_start:]
        eos_token = self.tokenizer.eos_token
        if response.endswith(eos_token):
            response = response[: -len(eos_token)]
        return response

    @property
    def response_with_indices(self, split_by="word", color=True) -> Union[str, pd.DataFrame]:
        """
        The response generated by the model, annotated with the starting index of each part.
        """
        parts, separators, start_indices = split_text(self.response, split_by)
        separated_str = highlight_word_indices(parts, start_indices, separators, color)
        return separated_str

    @property
    def num_sources(self) -> int:
        """
        For a filtered context, re-partition the context using the SentencePeriodPartitioner
        on the filtered version (generated from the full context and the response).
        """
        if not self._cache.get("filtered_partition"):
            # Increase parameters to produce a longer filtered context.
            filtered_ctx = filter_context(self.context, self.response, bm25_top_k=20, semantic_top_k=16)
            partitioner = SentencePeriodPartitioner(filtered_ctx)
            partitioner.split_context()
            self._cache["filtered_partition"] = partitioner
        return self._cache["filtered_partition"].num_sources

    @property
    def sources(self) -> List[str]:
        """
        Return the sources from the filtered context.
        """
        if not self._cache.get("filtered_partition"):
            filtered_ctx = filter_context(self.context, self.response, bm25_top_k=20, semantic_top_k=16)
            partitioner = SentencePeriodPartitioner(filtered_ctx)
            partitioner.split_context()
            self._cache["filtered_partition"] = partitioner
        return self._cache["filtered_partition"].sources

    def _char_range_to_token_range(self, start_index: int, end_index: int) -> Tuple[int, int]:
        output_tokens = self._output_tokens
        response_start = self._response_start
        offset = output_tokens.token_to_chars(response_start).start
        ids_start_index = char_to_token(output_tokens, start_index + offset)
        ids_end_index = char_to_token(output_tokens, end_index + offset - 1) + 1
        return ids_start_index - response_start, ids_end_index - response_start

    def _indices_to_token_indices(self, start_index: Optional[int] = None, end_index: Optional[int] = None) -> Tuple[int, int]:
        if start_index is None or end_index is None:
            start_index = 0
            end_index = len(self.response)
        if not (0 <= start_index < end_index <= len(self.response)):
            raise ValueError(
                f"Invalid selection range ({start_index}, {end_index}). "
                f"Please select any range within (0, {len(self.response)})."
            )
        return self._char_range_to_token_range(start_index, end_index)

    def _compute_masks_and_logit_probs(self) -> None:
        total = self.num_ablations
        masks_list = []
        logit_probs_list = []
        
        for start in tqdm(range(0, total, self.batch_size)):
            current_batch = min(self.batch_size, total - start)
            with ch.no_grad():
                batch_masks, batch_logit_probs = get_masks_and_logit_probs(
                    self.model,
                    self.tokenizer,
                    current_batch,
                    self.num_sources,
                    self._get_prompt_ids,
                    self._response_ids,
                    self.ablation_keep_prob,
                    batch_size=current_batch,
                    base_seed=start
                )
            masks_list.append(batch_masks)
            logit_probs_list.append(batch_logit_probs)
            ch.cuda.empty_cache()  # Free up GPU memory after processing each batch.
            del batch_masks, batch_logit_probs
        # Concatenate all batches into a single array.
        all_masks = np.concatenate(masks_list, axis=0)
        all_logit_probs = np.concatenate(logit_probs_list, axis=0)
        
        self._cache["reg_masks"] = all_masks
        self._cache["reg_logit_probs"] = all_logit_probs

    @property
    def _masks(self) -> NDArray:
        if self._cache.get("reg_masks") is None:
            self._compute_masks_and_logit_probs()
        return self._cache["reg_masks"]

    @property
    def _logit_probs(self) -> NDArray:
        if self._cache.get("reg_logit_probs") is None:
            self._compute_masks_and_logit_probs()
        return self._cache["reg_logit_probs"]

    def _get_attributions_for_ids_range(self, ids_start_idx: int, ids_end_idx: int) -> Tuple[NDArray, float]:
        outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
        self._cache["actual_logit_probs"] = outputs
        num_output_tokens = ids_end_idx - ids_start_idx
        weight, bias = self.solver.fit(self._masks, outputs, self.num_sources)
        return weight, bias

    @property
    def _actual_logit_probs(self) -> NDArray:
        return self._cache["actual_logit_probs"]

    def get_attributions(
            self,
            start_idx: Optional[int] = None,
            end_idx: Optional[int] = None,
            as_dataframe: bool = False,
            top_k: Optional[int] = None,
            verbose: bool = True,
        ) -> Union[NDArray, Any]:
        """
        Get the attributions for (part of) the response.
        Before computing attributions, re-filter the context using the full context and the generated response.
        """
        # After the response is generated, update the partitioner to use a filtered context.
        if not self._cache.get("filtered_partition"):
            filtered_ctx = filter_context(
                self.context,
                self.response,
                bm25_top_k=20,
                semantic_top_k=16
            )
            partitioner = SentencePeriodPartitioner(filtered_ctx)
            partitioner.split_context()
            self._cache["filtered_partition"] = partitioner
            self.partitioner = partitioner
    
        if self.num_sources == 0:
            print("[Warning] No sources to attribute to!")
            return np.array([])
    
        if not as_dataframe and top_k is not None:
            print("[Warning] top_k is ignored when not using dataframes.")
    
        ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
        selected_text = self.response[start_idx:end_idx]
        selected_tokens = self._response_ids[ids_start_idx:ids_end_idx]
        decoded_text = self.tokenizer.decode(selected_tokens)
        if selected_text.strip() not in decoded_text.strip():
            print(
                "[Warning] Decoded selected tokens do not match selected text.\n"
                "If the following look close enough, you can ignore this.\n"
                f"What you selected: {selected_text.strip()}\n"
                f"What is being attributed: {decoded_text.strip()}"
            )
    
        if verbose:
            print(f"Attributed: {decoded_text.strip()}")
    
        attributions, _bias = self._get_attributions_for_ids_range(ids_start_idx, ids_end_idx)
        if as_dataframe:
            return get_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions

    def get_pred_logit_probs(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> NDArray:
        """
        Compute the predicted logit probabilities for each ablation vector.
        
        For each ablation vector v (from self._masks), the predicted logit probability
        is computed as:
            fτ(v) = ⟨weight, v⟩ + bias,
        where (weight, bias) are obtained by regressing the actual aggregated logit 
        probabilities (from the selected part of the response) against the ablation masks.
        
        Parameters:
            start_idx (Optional[int]): The starting character index of the response for attribution.
            end_idx (Optional[int]): The ending character index of the response for attribution.
        
        Returns:
            NDArray: An array of predicted logit probabilities (one per ablation mask).
        """
        # Convert character indices into token indices.
        ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
        # Fit the regression model to obtain weights and bias.
        weight, bias = self._get_attributions_for_ids_range(ids_start_idx, ids_end_idx)
        # Compute predicted logit probabilities for each ablation vector:
        # fτ(v) = dot(weight, v) + bias.
        pred_logit_probs = self._masks @ weight + bias
        self._cache["pred_logit_probs"] = pred_logit_probs
        return pred_logit_probs

    @property
    def _pred_logit_probs(self) -> NDArray:
        if self._cache.get("pred_logit_probs") is None:
            self.get_pred_logit_probs()
        return self._cache["pred_logit_probs"]
