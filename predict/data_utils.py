# coding:utf-8
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

"""Utilities for Yunlv, files for prediction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle as pickle
import sys

PAD_ID = 4776
GO_ID = 0
EOS_ID = 1
UNK_ID = 4


def get_options(poem_type, dataset):
    options = {}

    options['dataset'] = dataset
    options['poem_type'] = poem_type

    options['song_gen'] = 1
    options['75_gen'] = 0

    # model params
    options['predict_prob_seq_margin'] = []
    options['memory_weight'] = float(sys.argv[3])  # memory的权重
    options['use_lstm'] = 1
    options['emb_dim'] = 200

    # files
    options['test_in_file'] = '../resource/predict_resource/' + 'test_' + options['dataset'] + '.txt'  # testset
    # options['predict_file'] = '../results/result_' + options['dataset'] + '.txt'  # default prediction results
    options['predict_file'] = 'results/' + sys.argv[1] + '_' + sys.argv[2] + '_' \
                              + str(options['memory_weight']) + '_' + str(options['poem_type']) + '.txt'
    options['test_head_file'] = '../resource/predict_resource/cangtou/head_song.txt'  # cangtou
    options['ci_list'] = '../resource/predict_resource/term_set.txt'

    # train restriction
    options['train_percentage'] = 1  # use x/1 percent of the dataset to train
    options['use_little_predict'] = 0  # use part of the dataset to predict
    options['cut_out_sort'] = 5000  # stop finding word if search more times than this
    options['cut_predict'] = 5000  # stop predicting if dataset is longer than this

    # for YunLv class
    options['yunmu_file'] = '../resource/predict_resource/YunLv/yun_utf8.txt'
    options['word_l_file'] = '../resource/predict_resource/YunLv/yunmu.txt'

    # pingshuiyun
    options['pingshui_file'] = ['../resource/predict_resource/YunLv/ze_format.txt',
                                '../resource/predict_resource/YunLv/ping_format.txt']  # old
    options['pingshui_file_new'] = ['../resource/predict_resource/YunLv/yun_z_new.txt',
                                    '../resource/predict_resource/YunLv/yun_p_new.txt']  # new
    options['hanzipinyin_file'] = '../resource/predict_resource/YunLv/hzpy.txt'
    options['use_pingshuiyun'] = 1
    options['use_pingshuiyun_morden'] = 1

    # Lv(pingze)
    if options['poem_type'] == 'poem7':
        options['yun_list'] = [[1, 2, 4]]
        lv_l = [['0', 'p', '0', 'z', '0', 'p', '0'], ['0', 'z', '0', 'p', '0', 'z', 'p'],
                ['0', 'z', '0', 'p', '0', 'z', '0'],
                ['0', 'p', '0', 'z', '0', 'p', 'p']]
    elif options['poem_type'] == 'poem5':
        options['yun_list'] = [[1, 2, 4]]
        lv_l = [['0', 'z', '0', 'p', '0'], ['0', 'p', '0', 'z', 'p'], ['0', 'p', '0', 'z', '0'],
                ['0', 'z', '0', 'p', 'p']]
    elif options['poem_type'] == 'jzmlh':  # 减字木兰花
        options['yun_list'] = [[1, 2], [3, 4], [5, 6]]  # ,[7,8]
        lv_l = [['0', 'p', '0', 'z'], ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'z', 'p', 'p'],
                ['0', 'z', 'p', 'p', '0', 'z', 'p'],
                ['0', 'p', '0', 'z'], ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'z', 'p', 'p'],
                ['0', 'z', 'p', 'p', '0', 'z', 'p']]
    elif options['poem_type'] == 'djc':  # 点绛唇
        options['yun_list'] = [[2, 3, 4], [6, 7, 8, 9]]
        lv_l = [['z', 'z', 'p', 'p'], ['z', 'p', 'z', 'z', 'p', 'p', 'z'], ['z', 'p', 'p', 'z'],
                ['p', 'z', 'p', 'p', 'z'],
                ['z', 'p', 'p', 'z'], ['p', 'z', 'p', 'p', 'z'], ['p', 'p', 'z'], ['z', 'p', 'p', 'z'],
                ['z', 'z', 'p', 'p', 'z']]
    elif options['poem_type'] == 'ymr':  # 虞美人
        options['yun_list'] = [[1, 2], [3, 5], [6, 7], [8, 10]]
        lv_l = [['0', 'p', '0', 'z', 'p', 'p', 'z'], ['0', 'z', 'p', 'p', 'z'], ['0', 'p', '0', 'z', 'z', 'p', 'p'],
                ['0', 'z', '0', 'p', '0', 'z'], ['z', 'p', 'p'],
                ['0', 'p', '0', 'z', 'p', 'p', 'z'], ['0', 'z', 'p', 'p', 'z'], ['0', 'p', '0', 'z', 'z', 'p', 'p'],
                ['0', 'z', '0', 'p', '0', 'z'], ['z', 'p', 'p']]
    elif options['poem_type'] == 'dlh':  # 蝶恋花
        options['yun_list'] = [[1, 3, 4, 5, 6, 8, 9, 10]]
        lv_l = [['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'z', 'p', 'p'], ['0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'z', 'p', 'p'], ['0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z']]
    elif options['poem_type'] == 'zgt':  # 鹧鸪天
        options['yun_list'] = [[1, 2, 4, 6, 7, 9]]  # [1,2],[4,6],[7,9]
        lv_l = [['0', 'z', 'p', 'p', 'z', 'z', 'p'], ['0', 'p', '0', 'z', 'z', 'p', 'p'],
                ['0', 'p', '0', 'z', 'p', 'p', 'z'], ['0', 'z', 'p', 'p', '0', 'z', 'p'], ['p', 'z', 'z'],
                ['z', 'p', 'p'], ['0', 'p', '0', 'z', 'z', 'p', 'p'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', 'p', 'p', 'z', 'z', 'p']]
    elif options['poem_type'] == 'psm':  # 菩萨蛮
        options['yun_list'] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        lv_l = [['0', 'p', '0', 'z', 'p', 'p', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p'], ['0', 'p', 'p', 'z', 'p'],
                ['0', 'p', 'p', 'z', 'z'], ['0', 'z', '0', 'p', 'z'],
                ['0', 'z', 'z', 'p', 'p'], ['0', 'p', '0', 'z', 'p']]
    elif options['poem_type'] == 'yja':  # 渔家傲
        options['yun_list'] = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]
        lv_l = [['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['p', '0', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z'],
                ['0', 'z', '0', 'p', 'p', 'z', 'z'], ['p', '0', 'z'], ['0', 'p', '0', 'z', 'p', 'p', 'z']]
    else:
        options['yun_list'] = [[]]
        lv_l = [[]]
        options['hard_pz'] = 0

    options['lv_list'] = lv_l
    options['use_all_lv'] = 1

    # related to YunLv
    options['use_correspond'] = 0
    options['use_correspond_finetune'] = 0
    options['use_pz_finetune'] = 0
    options['use_connect_word'] = 0

    # sentence length format
    # （虞美人 蝶恋花 减字木兰花 点绛唇 鹧鸪天 菩萨蛮 渔家傲）
    options['top_k'] = 1 # only can deal with 1
    if options['poem_type'] == 'ymr':
        options['predict_seq_len'] = [7, 5, 7, 6, 3, 7, 5, 7, 6, 3]
    elif options['poem_type'] == 'dlh':
        options['predict_seq_len'] = [7, 4, 5, 7, 7, 7, 4, 5, 7, 7]
    elif options['poem_type'] == 'jzmlh':
        options['predict_seq_len'] = [4, 7, 4, 7, 4, 7, 4, 7]
    elif options['poem_type'] == 'djc':
        options['predict_seq_len'] = [4, 7, 4, 5, 4, 5, 3, 4, 5]
    elif options['poem_type'] == 'zgt':
        options['predict_seq_len'] = [7, 7, 7, 7, 3, 3, 7, 7, 7]
    elif options['poem_type'] == 'psm':
        options['predict_seq_len'] = [7, 7, 5, 5, 5, 5, 5, 5]
    elif options['poem_type'] == 'yja':
        options['predict_seq_len'] = [7, 7, 7, 3, 7, 7, 7, 7, 3, 7]
    elif options['poem_type'] == 'poem5':
        options['predict_seq_len'] = [5, 5, 5, 5]
    elif options['poem_type'] == 'poem7':
        options['predict_seq_len'] = [7, 7, 7, 7]
    elif options['poem_type'] == 'test':
        options['predict_seq_len'] = [7, 7, 7, 9]
    else:
        options['predict_seq_len'] = [1000]
        options['end_break'] = 1
        options['top_k'] = 0
        options['count'] = []

    options['force_type'] = ''
    options['type_len_format'] = ['4-5-4-7-6-4-4-5-4-6-6-4-5-7-6-4-4-5-4-6', '4-7-7-4-5-7-7-7-4-5',
                                  '4-4-6-7-3-4-4-4-6-7-3-4',
                                  '6-6-7-6-6-6-7-6', '4-7-4-5-4-5-3-4-5', '3-6-7-5-6-6-7-5',
                                  '5-5-6-5-6-6-5-5-5-3-3-3-4-7-6-6-5-5-5', '5-4-7-4-6-5-3-4-4-7-4-6-5-3',
                                  '5-5-6-5-6-6-5-5-5-3-3-3-6-5-6-6-5-5-5', '7-7-7-7-7-7', '7-7-7-3-7-7-7-7-3-7',
                                  '7-5-6-5-3-3-3-4-4-4',
                                  '3-3-5-4-5-6-4-3-4-3-6-5-4-5-6-4-3-4', '4-5-5-3-4-4-5-3-3-5-4-5-5-3-4-4-5-3-3-5',
                                  '3-5-7-7-5-3-5-7-7-5', '6-6-5-6-2-2-6',
                                  '3-7-3-4-4-7-7-3-4-4', '7-5-7-6-3-7-5-7-6-3',
                                  '5-3-4-4-7-6-3-4-7-3-5-3-3-7-3-4-4-7-6-3-4-7-3-5-3-3', '5-5-7-5-5-5-7-5',
                                  '7-7-7-7-3-3-7-7-7',
                                  '4-5-7-6-6-6-6-6', '4-4-4-5-4-4-4-4-4-7-3-5-4-6-8-5-4-4-4-4-4-7-3-5-4',
                                  '7-5-7-5-3-3-5-7-5', '5-6-6-5-7-5-6-5', '5-4-7-7-4-5-4-7-7-4', '7-5-4-4-4-7-5-4-4-4',
                                  '4-4-6-4-5-6-3-4-3-4-5-2-3-4-4-5-4-6-3-4-3-4-5',
                                  '6-7-4-4-4-4-4-4-5-4-3-3-6-3-4-4-4-4-4-4-4-5-4-4', '7-7-7-7-7-7-7-7',
                                  '7-3-3-7-4-4-5-7-7-7-4-4-5', '7-6-4-4-6-4-5-4-4-7-6-5-4-4-4-6-4-5-4-4-5',
                                  '7-5-3-7-3-5-5-5-3-7-3-5',
                                  '7-5-6-6-4-4-4-6-6', '4-3-4-3-4-4-7-7-3-5-3-3-3-3-3-5-4-7-7-3-5-3',
                                  '7-6-7-5-5-7-6-7-5-5', '4-7-4-7-4-7-4-7', '7-7-5-5-5-5-5-5',
                                  '5-5-5-5-5-5-5-5', '4-4-4-4-4-4-6-3-4-4-4-4', '5-5-4-7-6-6-5-5-5-3-3-3-4-7-6-6-5-5-5',
                                  '7-4-5-7-7-7-4-5-7-7', '4-4-7-7-7-4-4-7-7-7', '3-3-7-5-3-3-7-5',
                                  '5-7-7-2-7-5-7-7-2-7',
                                  '7-3-3-4-5-7-3-3-7-3-3-4-5-7-3-3', '7-4-4-7-7-4-4-7',
                                  '4-3-6-7-6-4-4-5-4-6-6-4-5-7-6-4-4-5-4-6', '5-5-7-6-3-5-5-7-6-3', '7-7-7-7',
                                  '5-5-5-5']

    # rules
    options['first_w_kill'] = []  # ['不', '一', '天', '梦', '我', '月', '此', '水', '今', '夜', '风', '雨',
    #  '何', '惟', '谁', '万', '只', '故' , '人', '孤', '客', '山', '春', '满', '白', '若', '莫']
    options['number_forbidden'] = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '千', '百', '万']
    options['stop_words'] = ['的']
    options['allow_diff_sentence_repeat'] = 0

    # settings
    options['not_first'] = 0
    options['not_first_yun'] = 1
    options['lfirst'] = 1  # last first
    options['dfirst'] = 1  # use both first sentence and last sentence
    options['use_fgen'] = 1  # use the first generated sentence as final first sentence
    options['use_dgen'] = 0
    options['use_two_start'] = 1  # add START1 and END1 to word list
    options['use_se'] = 1
    options['reverse_training'] = 0  # options['top_k']!=0  # do not use 藏头
    options['topic_gen'] = 1
    options['cut_to_one'] = 0  # only can be use in acrostic poem 藏头诗

    # output
    options['print_param'] = 1
    options['write_file'] = 1

    # -------------控制规则强弱---------------
    options['hard_pz'] = 1  # 平仄
    options['hard_yun'] = 1  # 押韵，设为0可以生成宋词
    options['allow_repeat_word'] = 0  # 重复字
    options['allow_continue_repeat_word'] = 0
    options['use_noConnect'] = 1
    # --------------------------------------

    return options


def get_memory_options():
    options = {}
    options['other_file'] = ''  # add new style for memory. file name.
    options['other_type'] = 0  # check the following files for the format 0 or 1 refers to
    options['biansai_file'] = 'biansaishi-poetry.txt'
    options['biansai_type'] = 1
    options['tianyuan_file'] = 'tianyuan_poetry.txt'
    options['tianyuan_type'] = 0
    options['yanqing_file'] = 'yanqing-poetry.txt'
    options['yanqing_type'] = 1
    options['general_file'] = 'memory_poem.txt'
    options['general_type'] = 0
    return options


def Word2vec():
    with open('../resource/train_resource/word2vec_poem_58k.txt', 'rb') as f:
        wordMisc = pickle.load(f)
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        P_Emb = wordMisc['P_Emb']
        P_sig = wordMisc['P_sig']
    return word2id, id2word, P_Emb, P_sig
