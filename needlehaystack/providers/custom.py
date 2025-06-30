import os
from operator import itemgetter
from typing import Optional

import torch

from .model import ModelProvider
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from .stopping_criteria import MyStoppingCriteria

class Custom(ModelProvider):
    """
    A wrapper class for interacting with Custom model (e.g Huggingface), providing methods to encode text, generate prompts,
    evaluate models.

    Attributes:
        model_name (str): The name of the Huggingface (decoder only model) to use for evaluations and interactions.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """

    def __init__(self, model_name: str):
        """
        load model from huggingface.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
        
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          device_map='cuda', 
                                                          torch_dtype=torch.bfloat16)
                                                          #attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    async def evaluate_model(self, prompt: str, chat_template=False, max_new_tokens=128) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.
            chat_template (bool): to overlap chat template to the prompt
        Returns:
            str: The content of the model's response to the prompt.
        """
        # pre-train시
        if chat_template:
            prompt_str = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True,  # 이건 모델 스타일에 따라 True/False 조절,
            enable_thinking = False # False 시 non-think mode
            )
            _stopping_criteria = None
        else:
            # stopping criteria : \n, eos, . 
            newline_token_id = self.tokenizer.convert_tokens_to_ids('\n')
            custom_eos_id = self.tokenizer.eos_token_id  
            period_id = self.tokenizer.convert_tokens_to_ids('.')
            stop_ids = [newline_token_id, custom_eos_id, period_id]
            _stopping_criteria = StoppingCriteriaList([MyStoppingCriteria(self.tokenizer, stop_ids)])

        tokenized_prompts = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized_prompts.input_ids.to(self.model.device)
        streamer = TextStreamer(self.tokenizer)
        if not chat_template:
            output = self.model.generate(
                    input_ids,
                    stopping_criteria=_stopping_criteria,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=streamer
                    )
        else:
            output = self.model.generate(
                    input_ids,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer
                    use_cache=True)
        decoded_output = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens = True)
        return decoded_output, prompt_str
    
    def generate_prompt(self, context: str, retrieval_question: str, chat_template: bool = False) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.
            chat_template (bool) : True for sft model / False for plm.
        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        if chat_template:
            return [{
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
                }]
        else:
            # plm
            return [
            {
                "role": "user",
                "content": f'Document: {context}'
            },
            {
                "role": "user",
                "content": f"Question: {retrieval_question}\nAnswer: The best thing to do in San Francisco is "
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    

