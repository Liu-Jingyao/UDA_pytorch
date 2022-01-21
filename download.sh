# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
#!/bin/bash

# **** download pretrained models ****
#wget storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#unzip uncased_L-12_H-768_A-12.zip && rm uncased_L-12_H-768_A-12.zip
#mv uncased_L-12_H-768_A-12 BERT_Base_Uncased
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip

unzip uncased_L-8_H-512_A-8.zip && rm uncased_L-8_H-512_A-8.zip
mv uncased_L-8_H-512_A-8 BERT_Medium_Uncased

# **** unzip data ****
unzip data.zip && rm data.zip