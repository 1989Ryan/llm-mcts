import os
import sys
import pdb
import numpy as np
import pickle
from tqdm import tqdm
import random

# parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(parentdir)
# from common_variables import transformer_path
# sys.path.append(transformer_path)


from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BartModel,
    BartForConditionalGeneration,
    BartTokenizer,
)

MODEL_CLASSES = {
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
            "gpt2-medium": (GPT2LMHeadModel, GPT2Tokenizer),
            "gpt2-large": (GPT2LMHeadModel, GPT2Tokenizer),
            "gpt2-xl": (GPT2LMHeadModel, GPT2Tokenizer),
            "bart-base": (BartForConditionalGeneration, BartTokenizer),
            "bart-large": (BartForConditionalGeneration, BartTokenizer),
            "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMWithLMHeadModel, XLMTokenizer),
        }


LLM_HIDDEN_SIZE = {
    'gpt2': 768,
    'gpt2-medium': 1024,
    'gpt2-large': 1280,
    'gpt2-xl': 1600,
    'bart-base': 768,
    'bart-large': 768
}


def get_pretrained_tokenizer(model_type, model_name_or_path):
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    print('loading tokenizer %s' % model_type)
    return tokenizer


