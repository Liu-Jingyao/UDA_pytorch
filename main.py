# Copyright 2019 SanghunYun, Korea University.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import re
from math import ceil

import fire
import nltk

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
from augmentation import sent_level_augment, word_level_augment
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one, clean_web_text
from utils import optim, configuration


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def unsup_data_augmentation(cfg):
    with open(cfg.unsup_data_dir, "r") as f:
        ori_lines = f.readlines()
        def tokenizer(sen): return re.findall(r"[\w']+|[.,!?;]", sen)
        ori_lines = [clean_web_text(d).strip() for d in ori_lines]
        ori_lines = [" ".join(tokenizer(d)[-cfg.max_seq_length:]) for d in ori_lines]
        data_per_worker = ceil(len(ori_lines) / cfg.replicas)
        start = cfg.worker_id * data_per_worker
        end = min(start + data_per_worker, len(ori_lines))
        ori_lines = ori_lines[start:end]
        print("processing data from %d to %d" % (start, end - 1))
    aug_lines = copy.deepcopy(ori_lines)
    if cfg.aug_ops.startswith("bt"):
        aug_lines = sent_level_augment.run_augment(aug_lines, cfg.aug_ops, cfg.aug_copy_num, cfg.aug_batch_size,
                                                   cfg.max_seq_length)
    else:
        aug_lines = word_level_augment.run_augment(aug_lines, cfg.aug_ops, tokenizer, cfg.aug_copy_num)
    ori_aug_lines = [(ori_lines[i // cfg.aug_copy_num].rstrip(), aug_lines[i]) for i in range(len(aug_lines))]
    return ori_aug_lines


def main(cfg, model_cfg):
    # Load Configuration
    cfg = configuration.params.from_json(cfg)  # Train or Eval cfg
    if cfg.mode == "augmentation":
        ori_aug_lines = unsup_data_augmentation(cfg)
        with open("data/imdb_unsup_train_aug.txt", "w") as f:
            f.writelines(["%s\t%s\n" % (ori, aug) for ori, aug in ori_aug_lines])
        return
    if cfg.mode == "tokenize":
        data = load_data(cfg)
        sup_dataset, unsup_dataset, eval_dataset = data.get_all_dataset()
        with open("data/imdb_sup_train_tokenized.txt", "w") as f1, open("data/imdb_unsup_train_tokenized.txt",
                                                                        "w") as f2, open(
            "data/imdb_sup_test_tokenized.txt", "w") as f3:
            f1.write("input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n")
            f3.write("input_ids\tinput_mask\tinput_type_ids\tlabel_ids\n")
            f2.write(
                "ori_input_ids\tori_input_mask\tori_input_type_ids\taug_input_ids\taug_input_mask\taug_input_type_ids\n")
            for d in sup_dataset.data:
                f1.write("%s\t%s\t%s\t%s\n" % (d[0], d[2], d[1], d[3]))
            for d in eval_dataset.data:
                f3.write("%s\t%s\t%s\t%s\n" % (d[0], d[2], d[1], d[3]))
            for i in range(len(unsup_dataset.data['ori'])):
                ori = unsup_dataset.data['ori'][i]
                aug = unsup_dataset.data['aug'][i]
                f2.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (ori[0], ori[2], ori[1], aug[0], aug[2], aug[1]))
        return
    model_cfg = configuration.model.from_json(model_cfg)  # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)
    data_iter = dict()
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        if cfg.mode == 'train':
            data_iter['sup_iter'] = data.sup_data_iter()
            data_iter['unsup_iter'] = data.unsup_data_iter()
        elif cfg.mode == 'train_eval':
            data_iter['sup_iter'] = data.sup_data_iter()
            data_iter['unsup_iter'] = data.unsup_data_iter()
            data_iter['eval_iter'] = data.eval_data_iter()
        else:
            data_iter['eval_iter'] = data.eval_data_iter()
    else:
        if cfg.mode == 'train':
            data_iter['sup_iter'] = data.sup_data_iter()
        elif cfg.mode == 'train_eval':
            data_iter['sup_iter'] = data.sup_data_iter()
            data_iter['eval_iter'] = data.eval_data_iter()
        else:
            data_iter['eval_iter'] = data.eval_data_iter()
    sup_criterion = nn.CrossEntropyLoss(reduction='none')

    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))

    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device())

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):

        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)

        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1. / logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (
                    1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                           torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob = F.softmax(ori_logits, dim=-1)  # KLdiv target
                # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())

            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html

                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                     torch_device_one())
            final_loss = sup_loss + cfg.uda_coeff * unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)

        result = (label_pred == label_id).float()
        accuracy = result.mean()
        # output_dump.logs(sentence, label_pred, label_id)    # output dump

        return accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
        trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
    # main('config/uda.json', 'config/bert_base.json')
