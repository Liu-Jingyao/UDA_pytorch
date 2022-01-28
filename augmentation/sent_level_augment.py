import os

import torch
import transformers
from transformers import MarianMTModel, MarianTokenizer


def back_translation(ori_lines, aug_ops, aug_copy_num):
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
    translated_tokens = en_fr_model.generate(
        **{k: v.cuda() for k, v in
           en_fr_tokenizer(ori_lines, return_tensors="pt", padding=True, max_length=512).items()},
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=temp,
        num_return_sequences=aug_copy_num
    )
    in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    print("Translating French data back to English...")
    bt_tokens = fr_en_model.generate(
        **{k: v.cuda() for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, max_length=512).items()},
        do_sample=True,
        top_k=10,
        top_p=0.95,
        temperature=temp,
        num_return_sequences=aug_copy_num
    )
    in_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]
    return in_en


def run_augment(ori_lines, aug_ops, aug_copy_num):
    if aug_ops:
        if aug_ops.startswith("bt"):
            aug_lines = back_translation(ori_lines, aug_ops, aug_copy_num)
        else:
            pass
    return aug_lines
