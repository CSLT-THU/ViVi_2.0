# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Utilities for data processing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle


# Special vocabulary symbols.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 4776
GO_ID = 0
EOS_ID = 1
UNK_ID = 4


# Get Word2vec dictionaries.
def Word2vec():
    with open('../resource/train_resource/word2vec_poem_58k.txt', 'rb') as f:
        wordMisc = pickle.load(f)
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        P_Emb = wordMisc['P_Emb']
        P_sig = wordMisc['P_sig']
    return word2id, id2word, P_Emb, P_sig
