import os
import re

os.environ["MODEL_DIR"] = '../model'

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action


def diy_aug(aug_class):
    class DiyAug(aug_class):
        def __init__(self, **params):
            params_needed = super(DiyAug, self).__init__.__code__.co_varnames
            params = {k: v for k, v in params.items() if k in params_needed}
            super(DiyAug, self).__init__(**params)

    return DiyAug


aug_dict = {
    "spelling": lambda params: diy_aug(naw.SpellingAug)(**params),
    "random_swap": lambda params: diy_aug(naw.RandomWordAug)(action="swap", **params),
    "model_substitute": lambda params: diy_aug(naw.ContextualWordEmbsAug)(model_path='distilbert-base-uncased',
                                                                          action="substitute", **params),
    "bt": lambda params: diy_aug(naw.BackTranslationAug)(from_model_name='facebook/wmt19-en-de',
                                                to_model_name='facebook/wmt19-de-en', **params),
    "sentence_insert": lambda params: diy_aug(nas.ContextualWordEmbsForSentenceAug)(model_path='distilgpt2', **params),
}


def run_augment(ori_lines, aug_ops, aug_copy_num, aug_batch_size, max_len):
    print("\ntool augmentation using %s..." % aug_ops.split("-")[0])
    aug_ops = aug_ops.split("-")
    aug_type = aug_ops[0]
    p = float(aug_ops[1]) if len(aug_ops) > 1 else None
    aug_params = {
        "aug_p": p,
        "device": "cuda",
        "batch_size": aug_batch_size,
        "max_length": max_len
    }
    aug = aug_dict[aug_type](aug_params)
    aug_lines = aug.augment(ori_lines, n=aug_copy_num)
    return aug_lines


if __name__ == '__main__':
    # print(run_augment(["The quick brown fox jumps over the lazy dog ."], "spelling-0.3", 1, 1, 512))
    # print(run_augment(["The quick brown fox jumps over the lazy dog ."], "random_swap-0.3", 1, 1, 512))
    # print(run_augment(["The quick brown fox jumps over the lazy dog ."], "tf_idf_insert", 1, 1, 512))
    print(run_augment(["The quick brown fox jumps over the lazy dog ."], "model_substitute-0.3", 1, 1, 512))
