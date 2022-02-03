import os
from math import ceil

import torch
import transformers
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline

def clean_web_text(st):
  """clean text."""
  st = st.replace("<br />", " ")
  st = st.replace("&quot;", "\"")
  st = st.replace("<p>", " ")
  if "<a href=" in st:
    # print("before:\n", st)
    while "<a href=" in st:
      start_pos = st.find("<a href=")
      end_pos = st.find(">", start_pos)
      if end_pos != -1:
        st = st[:start_pos] + st[end_pos + 1:]
      else:
        print("incomplete href")
        print("before", st)
        st = st[:start_pos] + st[start_pos + len("<a href=")]
        print("after", st)

    st = st.replace("</a>", "")
    # print("after\n", st)
    # print("")
  st = st.replace("\\n", " ")
  st = st.replace("\\", " ")
  # while "  " in st:
  #   st = st.replace("  ", " ")
  return st

def back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len):
    ori_lines = [clean_web_text(d) for d in ori_lines]
    bt_args = aug_ops.split("-")
    temp = float(bt_args[1])
    torch.cuda.empty_cache()
    en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", cache_dir='./augmentation/HFCache/',
                                                      model_max_length=max_len)
    en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr",
                                                cache_dir='./augmentation/HFCache/').cuda()
    en_fr_translator = pipeline("translation", model=en_fr_model, tokenizer=en_fr_tokenizer, device=0, truncation=True,
                                max_length=512)
    fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", cache_dir='./augmentation/HFCache/',
                                                      model_max_length=max_len)
    fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en",
                                                cache_dir='./augmentation/HFCache/').cuda()
    fr_en_translator = pipeline("translation", model=fr_en_model, tokenizer=fr_en_tokenizer, device=0, truncation=True,
                                max_length=512)
    batch_num = ceil(len(ori_lines) / aug_batch_size)
    print("Translating origin data to French...")
    fr_lines = []
    for i in tqdm(range(batch_num)):
        start = i * aug_batch_size
        end = min(start + aug_batch_size, len(ori_lines))
        in_fr = en_fr_translator(ori_lines[start:end],
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95,
                                 temperature=temp,
                                 num_return_sequences=aug_copy_num)
        fr_lines.extend([d["translation_text"] for d in in_fr])
    print("Translating French data back to English...")
    aug_lines = []
    for i in tqdm(range(batch_num)):
        start = i * aug_batch_size
        end = min(start + aug_batch_size, len(fr_lines))
        in_en = fr_en_translator(ori_lines[start:end],
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95,
                                 temperature=temp,
                                 num_return_sequences=aug_copy_num)
        aug_lines.extend([d["translation_text"] for d in in_en])
    return aug_lines


def run_augment(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len):
    if aug_ops:
        if aug_ops.startswith("bt"):
            aug_lines = back_translation(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len)
        else:
            pass
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
