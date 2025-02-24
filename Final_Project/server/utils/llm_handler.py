import torch
from transformers import pipeline
from dataclasses import dataclass
from typing import List, Optional, Tuple
from . import logging_utils
from ..config import LLM_MODEL_ID    # Changed: .config

logger = logging_utils.get_logger(__name__)

@dataclass
class ConversationState:
    context: List[dict]
    last_response: Optional[str] = None

class LLMHandler:
    def __init__(self, model_id: str = LLM_MODEL_ID):
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            self.tokenizer = self.pipe.tokenizer
            self.conversation = ConversationState(context=[])

            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id == self.tokenizer.unk_token_id:
                self.terminators = [self.tokenizer.eos_token_id]
            else:
                self.terminators = [self.tokenizer.eos_token_id, eot_id]

            logger.info("LLM handler initialized.")
        except Exception as e:
            logger.exception(f"Failed to initialize LLM: {e}")
            raise  # Re-raise the exception to halt execution if LLM fails

    def process_input(self, text: str) -> Tuple[str, float]:
        try:
            self.conversation.context.append({"role": "user", "content": text})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant on a home speaker. "
                        "Respond concisely without repeating entire logs. "
                        "You'll hear multiple people with names if available."
                    )
                }
            ] + self.conversation.context

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = ""
                for msg in messages:
                    prompt += f"<{msg['role']}>: {msg['content']}\n"

            logger.debug(f"LLM Prompt:\n{prompt}")

            with torch.no_grad():  
                output = self.pipe(
                    prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    eos_token_id=self.terminators,
                    return_full_text=False  # only return generated text
                )
            generated_text = output[0]["generated_text"].strip()

            self.conversation.context.append({"role": "assistant", "content": generated_text})

            probability = 0.9 if any(
                kw in text.lower() for kw in ["bye", "goodbye", "stop", "end"]
            ) else 0.0
            return generated_text, probability

        except Exception as e:
            logger.exception(f"Error during LLM processing: {e}")
            return "Sorry, I encountered an error.", 0.0  # eeturn a default response