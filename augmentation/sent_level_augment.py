import os
from math import ceil

import torch
import transformers
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline

def back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len):
    bt_args = aug_ops.split("-")
    temp = float(bt_args[1])
    torch.cuda.empty_cache()
    en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", cache_dir='./augmentation/HFCache/',
                                                      model_max_length=max_len)
    en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr",
                                                cache_dir='./augmentation/HFCache/').cuda()
    fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", cache_dir='./augmentation/HFCache/',
                                                      model_max_length=max_len)
    fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en",
                                                cache_dir='./augmentation/HFCache/').cuda()
    batch_num = ceil(len(ori_lines) / aug_batch_size)
    print("Translating origin data to French...")
    fr_lines = []
    for i in tqdm(range(batch_num)):
        start = i * aug_batch_size
        end = min(start + aug_batch_size, len(ori_lines))
        translated_tokens = en_fr_model.generate(
            **{k: v.cuda() for k, v in
               en_fr_tokenizer(ori_lines[start:end], return_tensors="pt", padding=True, truncation=True, max_length=max_len).items()},
            do_sample=True,
            top_k=10,
            temperature=temp,
        )
        in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        fr_lines.extend(in_fr)
    print("Translating French data back to English...")
    aug_lines = []
    for i in tqdm(range(batch_num)):
        start = i * aug_batch_size
        end = min(start + aug_batch_size, len(fr_lines))
        bt_tokens = fr_en_model.generate(
            **{k: v.cuda() for k, v in
               fr_en_tokenizer(ori_lines[start:end], return_tensors="pt", padding=True, truncation=True, max_length=max_len).items()},
            do_sample=True,
            top_k=10,
            temperature=temp,
        )
        in_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]
        aug_lines.extend(in_en)
    return aug_lines


def run_augment(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len):
    if aug_ops:
        if aug_ops.startswith("bt"):
            aug_lines = back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len)
        else:
            aug_lines = ori_lines
    return aug_lines



if __name__ == '__main__':
    en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", cache_dir='./HFCache/',
                                                      model_max_length=512)
    en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr",
                                                cache_dir='./HFCache/').cuda()
    model_args = {"do_sample": True,
                  "top_k": 10,
                  "top_p": 0.95,
                  "temperature": 0.9,
                  "num_return_sequences": 1}
    en_fr_translator = pipeline("translation", model=en_fr_model, tokenizer=en_fr_tokenizer, model_kwargs=model_args,
                                device=0)
    a = en_fr_translator(["How old are you?" * 512], truncation=True, max_length=400)
    for out in en_fr_translator([""] ,truncation=True, max_length=400):
        print(out)
