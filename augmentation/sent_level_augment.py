import os

import torch
import transformers
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline


def data_iter(data):
    for d in data:
        yield d


def back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size):
    bt_args = aug_ops.split("-")
    temp = float(bt_args[1])
    torch.cuda.empty_cache()
    en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", cache_dir='./augmentation/HFCache/')
    en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr",
                                                cache_dir='./augmentation/HFCache/').cuda()
    fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", cache_dir='./augmentation/HFCache/')
    fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en",
                                                cache_dir='./augmentation/HFCache/').cuda()
    print("Translating origin data to French...")
    model_args = {"do_sample": True,
                  "top_k": 10,
                  "top_p": 0.95,
                  "temperature": temp,
                  "num_return_sequences": aug_copy_num}
    en_fr_translator = pipeline("translation", model=en_fr_model, tokenizer=en_fr_tokenizer, model_kwargs=model_args,
                                device=0)
    fr_lines = []
    for out in tqdm(en_fr_translator(data_iter(ori_lines)), total=len(ori_lines)):
        fr_lines.append(out["translation_text"])
    print("Translating French data back to English...")
    fr_en_translator = pipeline("translation", model=fr_en_model, tokenizer=fr_en_tokenizer, model_kwargs=model_args,
                                device=0)
    aug_lines = []
    for out in tqdm(fr_en_translator(data_iter(fr_lines)), total=len(fr_lines)):
        aug_lines.append(out["translation_text"])
    return aug_lines


def run_augment(ori_lines, aug_ops, aug_copy_num, aug_batch_size):
    if aug_ops:
        if aug_ops.startswith("bt"):
            aug_lines = back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size)
        else:
            pass
    return aug_lines
