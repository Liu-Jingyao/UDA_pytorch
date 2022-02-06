import collections
import copy
import json
import math
import string
import numpy as np
import re
from string import punctuation

from tqdm import tqdm

printable = set(string.printable)


def filter_unicode(st):
    return "".join([c for c in st if c in printable])


def build_vocab(data):
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(data)):
        add_to_vocab(data[i])
    return vocab


def get_data_stats(data):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(data)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(data[i])
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(data) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(data)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(data[i])
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return {
        "idf": idf,
        "tf_idf": tf_idf,
    }


class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token


class TfIdfWordRep(EfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob, data_stats):
        super(TfIdfWordRep, self).__init__()
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = list(data_stats["tf_idf"].items())
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max()
                                  - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf
                                  / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (replace_prob / replace_prob.sum() *
                        self.token_prob * len(all_words))
        return replace_prob

    def __call__(self, data):
        if self.get_random_prob() < 0.001:
            show_data = True
        else:
            show_data = False
        all_words = copy.deepcopy(data)

        if show_data:
            # print("before tf_idf_unif aug: {:s}".format(
            #     filter_unicode(" ".join(all_words))))

        replace_prob = self.get_replace_prob(all_words)
        data = self.replace_tokens(
            data,
            replace_prob[:len(data)]
        )

        if show_data:
            all_words = copy.deepcopy(data)
            # print("after tf_idf_unif aug: {:s}".format(
            #     filter_unicode(" ".join(all_words))))
        return data

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if word_list[i] in punctuation:
                continue
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        # print("sampled token list: {:s}".format(
        #     filter_unicode(" ".join(self.token_list))))


def run_augment(ori_lines, aug_ops, tokenizer, aug_copy_num):
    print("word level augmentation using %s..." % aug_ops.split("-")[0])
    if aug_ops:
        if aug_ops.startswith("tf_idf"):
            ori_lines = [tokenizer(d) for d in copy.deepcopy(ori_lines)]
            # data = [get_only_chars(sup).strip().split(" ") for sup in ori_lines]
            data_stats = get_data_stats(ori_lines)

            print("\n>>Using augmentation {}".format(aug_ops))
            token_prob = float(aug_ops.split("-")[1])
            op = TfIdfWordRep(token_prob, data_stats)

            aug_lines = []
            for i in tqdm(range(len(ori_lines) * aug_copy_num)):
                aug_lines.append(" ".join(op(ori_lines[i // aug_copy_num])))
            return aug_lines
    return ori_lines
