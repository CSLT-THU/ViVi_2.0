# coding:utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys

import numpy as np
import tensorflow as tf
import seq2seq_model
import data_utils
import memoryModule_decoder

np.set_printoptions(threshold='nan')

word2id, id2word, P_Emb, P_sig = data_utils.Word2vec()

# get frequency of words in training set
dic_fre = {}
for p in open('../resource/train_resource/poem_58k_theme.txt', 'r').readlines():
    _, target = p.strip().split('==')
    lines = target.decode('utf-8').split('\t')
    for l in lines:
        words = l.split(' ')
    for w in words:
        temp_w = w.encode('utf-8')
        if dic_fre.has_key(temp_w):
            dic_fre[temp_w] += 1
        else:
            dic_fre[temp_w] = 1

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 500, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", len(word2id), "Vocabulary size.")
FLAGS = tf.app.flags.FLAGS

_buckets = [(20, 2)]


def get_next_id(startWords, idSet, predictls_temp, candidate, _candidates, \
                sen_candidates, yunmuModel, count_accum, i_count, j, i, len_c, word2id, id2word, \
                lv_list_l, charac_num, predict_word_id_list, options, l_sentence, full_words, ran, ci_list, hid_dict):
    """Get next word in predict process.

  Args:
    startWords: prepared the start word.
    idSet: banned word sets.
    candidate: candidate word set.
    _candidates: temporary word set.
    sen_candidates: sentence candidates.
    yunmuModel: yunmu model.
    buckets: A list of pairs of (input size, output size) for each bucket.
    keep_prob: the dropout rate.
    is_feed: if True, only the first of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    memory_weight: the weight of the memory model.
    count_accum: the number of generated words.
    i_count: the number of generated sentences.
    j: the jth sentence of the poem.
    i:the ith word of the sentence.
    len_c: the length of generated words.
    word2id: the word2id dictionary.
    id2word: the id2word dictionary.
    lv_list_l: rythm list.
    charac_num: the number of words in dictionary.
    predict_word_id_list: the order of the predicted words.
    options: configs.
    l_sentence: the theme words.
    full_words:  To be deleted
    ran: the length format of the poem.
    ci_list: the words in the dictionary.
    hid_dict: to be deleted

  Returns:
     The maximum probability word.
  """

    if options['cut_to_one'] == 1:
        count_accum /= i_count  # count_accum 单词数；i_count 句子数

    lv_list = lv_list_l  # lv_lsit_l: rythm list
    if options['not_first'] != 0:
        if options['lfirst'] != 1:
            lv_list = lv_list[1:]
        else:
            lv_list = lv_list[1:] + [lv_list[0]]
    else:
        if options['lfirst'] == 1:
            if options['dfirst'] != 1:
                lv_list = lv_list[1:] + [lv_list[0]]
            else:
                lv_list = lv_list + [lv_list[0]]

    if options['reverse_training']:
        lv_list = lv_list[::-1]
        for iil in range(len(lv_list)):
            temp = lv_list[iil][::-1]
            lv_list[iil] = temp

    def get_former_dui_word(i_count, j, i, candidate):
        # j: the jth sentence of the poem.
        # i:the ith word of the sentence.
        # count_accum: the number of generated words.
        # i_count: the number of generated sentences.
        # candidate: candidate word set.
        l = candidate['word'].replace('START', '')
        count = 0
        t = l.split()
        for temps in t:
            for w in temps.decode('utf-8'):
                count += 1
                if count == (i_count - 1) * i + j + 1:
                    return w.encode('utf-8')
        return ''

    yunD0 = {}

    top_count = -1
    k_count = 0
    k_count_sen = 0
    do_not_w = 0
    do_not_sen = 0

    id_s = set([])
    if options['cut_to_one'] == 0:
        wset = candidate['word'].split()
        if options['allow_diff_sentence_repeat']:
            wset = candidate['word'].split(' /')[-1].split()
        for w in wset:
            id_s.add(word2id.get(w, charac_num - 1))
    else:
        # print full_words    full_words:  To be deleted
        wset = full_words.replace(' /', '').split()
        if options['allow_diff_sentence_repeat']:
            wset = full_words.split(' /')[-1].split()
        for w in wset:
            id_s.add(word2id.get(w, charac_num - 1))

    last_repeat = 0
    temp = full_words.replace(' /', '').split()
    if len(temp) == 0:
        last_id = word2id.get(l_sentence.split()[-1], charac_num - 1)
    else:
        last_id = word2id.get(temp[-1], charac_num - 1)
        if len(temp) > 1:
            if temp[-1] == temp[-2]:
                last_repeat = 1

    finded = 0
    if len(startWords) <= i_count - 1:
        finded = 1
    else:
        w_temp = startWords[i_count - 1].encode('utf-8')

        if j == 0:
            for w in word2id.keys():
                if w.find(w_temp) == 0:
                    finded = 1
            if finded == 0:
                temp_can_dict = {'word': candidate['word'] + ' ' + w_temp, 'prob': candidate['prob'] * 1}
                temp_can_dict['S_H'] = hid_dict['S_H']
                if options['use_lstm']:
                    temp_can_dict['C_H'] = hid_dict['C_H']  # hid_dict: to be deleted
                _candidates.append(temp_can_dict)
                # break
    # skip
    if options['use_correspond_finetune'] != 0 and j != 0:  # default=0
        if i_count % 2 == 0:
            word_former = get_former_dui_word(i_count - 1, j, i, candidate)
            # it is not strong finetune
            if word_former in yunmuModel.dui_zhang_dict.keys():
                w_l = yunmuModel.dui_zhang_dict[word_former]
                for w in w_l:
                    idt = word2id.get(w, charac_num - 1)
                    predict_word_id_list[idt] = predict_word_id_list[idt] * options['use_correspond_finetune']  # *avg
                predictls_temp = predict_word_id_list.argsort(axis=0)
        else:
            for idt in range(len(predict_word_id_list)):
                if id2word[idt] in yunmuModel.dui_zhang_dict.keys():
                    predict_word_id_list[idt] = predict_word_id_list[idt] * options['use_correspond_finetune']  # *avg
            predictls_temp = predict_word_id_list.argsort(axis=0)

    # skip
    if options['use_pz_finetune'] != 0 and j != 0:  # default=0
        former_ws = []
        l = candidate['word'].replace('START', '')
        if options['allow_diff_sentence_repeat'] == 0:
            l = l.replace(' /', '')
        else:
            l = l.split(' /')[-1].split()
        temp_str = ''
        count = 0
        t = l.split()
        for temps in t:
            for w in temps.decode('utf-8'):
                count += 1
                if count % i != 0:
                    former_ws.append(w.encode('utf-8'))
        jc = count % i
        if len(lv_list) > i_count - 1:
            if len(lv_list[i_count - 1]) > jc:
                lv = lv_list[i_count - 1][jc]
                if lv != '0':
                    for idt in range(len(predict_word_id_list)):
                        if yunmuModel.getYunDiao(id2word[idt])['p'] == lv:
                            predict_word_id_list[idt] = predict_word_id_list[idt] * options['use_pz_finetune']  # *avg
                    predictls_temp = predict_word_id_list.argsort(axis=0)

    if finded == 1 or j != 0:
        for id_t in predictls_temp[::-1]:
            top_count += 1
            word = id2word[id_t]

            if len(options['predict_prob_seq_margin']):
                if predict_word_id_list[id_t] > options['predict_prob_seq_margin'][1] or predict_word_id_list[id_t] < \
                        options['predict_prob_seq_margin'][0]:
                    continue

            if options['use_noConnect']:
                if options['poem_type'] == 'poem5' and j == 2:
                    if id2word[last_id] + id2word[id_t] in ci_list:
                        continue
                if options['poem_type'] == 'poem7' and j == 4:
                    if id2word[last_id] + id2word[id_t] in ci_list:
                        continue

            if len(options['number_forbidden']):
                if options['poem_type'] == 'poem5' and (j == 1 or j == 4):
                    if id2word[id_t] in options['number_forbidden']:
                        continue
                if options['poem_type'] == 'poem7' and (j == 3 or j == 6):
                    if id2word[id_t] in options['number_forbidden']:
                        continue

            yun0 = yunmuModel.getYunDiao(word)

            if len(startWords) == 0 and (id_t in idSet):
                continue
            if len(startWords) == 0 and options['allow_repeat_word'] == 0 and options[
                'allow_continue_repeat_word'] == 0 and (id_t in id_s):
                continue
            if len(startWords) == 0 and options['allow_repeat_word'] == 1 and options[
                'allow_continue_repeat_word'] == 0 and id_t == last_id:
                continue
            if len(startWords) == 0 and options['allow_repeat_word'] == 1 and options[
                'allow_continue_repeat_word'] == 1 and id_t == last_id and last_repeat == 1:
                continue

            if options['allow_repeat_word'] == 0:
                if (id_t in idSet) or (j != 0 and id_t in id_s) or (
                        j == i - 1 and len(startWords) and word.decode('utf-8') == startWords[i_count - 1]):
                    continue

            str_t = word.decode('utf-8')
            if j == 0 and len(startWords) >= i_count and startWords[i_count - 1].find(str_t) != 0:
                continue

            if j == 0 and word in options['first_w_kill']:
                continue

            former_ws = []
            l = candidate['word'].replace('START', '')

            if options['allow_diff_sentence_repeat']:
                l = l.split(' /')[-1]
            else:
                l = l.replace(' /', '')

            temp_str = ''
            count = 0
            t = l.split()

            for w in t:
                if w != ' ' and w != '':
                    count += 1
                    if count % i != 1 or len(startWords) == 0:
                        former_ws.append(w)

            if options['allow_repeat_word'] == 0:
                if j != 0 and word in former_ws:
                    continue
                if j == 0 and len(startWords) == 0 and word in former_ws:
                    continue

                if options['not_first'] and word in l_sentence.split():
                    continue

                if len(startWords) and (word.decode('utf-8') in startWords) and j != 0:
                    continue

            # 必须使用词性表 ？ 活用型词语 ？
            # 两句对仗
            word_former = ''

            if options['use_correspond'] == 1 and j != 0:
                if i_count % 2 == 1:
                    if word not in yunmuModel.dui_zhang_dict.keys():
                        continue
                else:
                    word_former = get_former_dui_word(i_count - 1, j, i, candidate)

                    if word == word_former:
                        continue
                    else:
                        if options['use_correspond_finetune'] == 0:
                            w_l = yunmuModel.dui_zhang_dict[word_former]
                            wi = random.randint(0, len(w_l) - 1)
                            word = w_l[wi]

            def fit_lv(word_l, lv_list, i_count, j):
                return 1

            if options['use_connect_word'] == 1:
                if j == 0:
                    word_l = yunmuModel.find_start(word)
                    if word_l != '' and fit_lv(word_l, lv_list, i_count, j):
                        word = word_l
                    else:
                        continue

            if not dic_fre.has_key(word) or dic_fre[word] < 200:  # important, cannot be modifid
                continue

            if options['reverse_training'] == 0:
                if options['hard_pz']:  # 不藏头 @@@@??
                    if len(lv_list) > i_count - 1:
                        if len(lv_list[i_count - 1]) > j:
                            lv = lv_list[i_count - 1][j]
                            if options['use_pingshuiyun'] == 0:
                                if lv != '0' and yun0['p'] != '-1':
                                    if yun0['p'] != lv:  # if word not in list, continue
                                        continue
                                    else:
                                        if options['use_pingshuiyun_morden']:
                                            if lv != '0' and word not in yunmuModel.pzlist[lv]:
                                                print('00000' * 80)  # -yb
                                                continue
                                if not (options['yun_list'][0][0] == i_count and j == i - 1):
                                    if (lv != '0' and word not in yunmuModel.pzlist[lv]) or (
                                            lv != '0' and yun0['p'] != lv):
                                        continue
                                else:
                                    if lv != '0' and word not in yunmuModel.yun_pzlist[lv]:
                                        continue
            else:
                if options['hard_pz']:
                    if len(lv_list) > i_count - 1:
                        if len(lv_list[i_count - 1]) > j:
                            lv = lv_list[i_count - 1][j]
                            if options['use_pingshuiyun'] == 0:
                                if lv != '0' and yun0['p'] != '-1':
                                    if yun0['p'] != lv:  # if word not in list, continue
                                        continue
                                    else:
                                        if options['use_pingshuiyun_morden']:
                                            if lv != '0' and word not in yunmuModel.pzlist[lv]:
                                                continue
                            else:
                                if (lv != '0' and word not in yunmuModel.pzlist[lv]) or (lv != '0' and lv != yun0['p']):
                                    continue

            if options['hard_yun'] and options['use_pingshuiyun'] == 1 \
                    and len(word.decode('utf-8')) + len_c == count_accum:
                i_count_f = i_count
                if options['not_first'] != 0:
                    i_count_f += 1
                else:
                    if options['lfirst'] == 1:
                        if options['dfirst'] != 1:
                            i_count_f += 1

                if (options['not_first'] == 0 and options['lfirst'] == 1) and i_count_f == len(
                        options['predict_seq_len']) + 1:
                    i_count_f = 1

                yun_lis = []
                for yu_li in options['yun_list']:
                    if i_count_f in yu_li[1:] or (i_count_f in yu_li and 1 in yu_li):
                        yun_lis = yu_li
                        break
                all_pz_list = set([])
                for yl in yun_lis:
                    if yl - 1 < len(lv_list):
                        pz = lv_list[yl - 1][-1]
                        if pz != '0':
                            all_pz_list.add(pz)

                # print '皓' in yunmuModel.pzlist['p']
                to_continue = 0
                for pz in all_pz_list:
                    if word not in yunmuModel.pzlist[pz]:
                        to_continue = 1
                        break

                if options['use_pingshuiyun_morden']:
                    if len(options['yun_list']) > yunmuModel.getYunLineLen(word) - 5:
                        to_continue = 1
                else:
                    if len(options['yun_list']) > yunmuModel.getYunLineLen(word):
                        to_continue = 1

                if to_continue:
                    continue

            if options['reverse_training'] == 0:
                if (len(word.decode(
                        'utf-8')) + len_c < count_accum) and not do_not_w:  # and len(word.decode('utf-8'))<i-1:
                    temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                     'prob': candidate['prob'] * predict_word_id_list[id_t]}
                    temp_can_dict['S_H'] = hid_dict['S_H']
                    if options['use_lstm']:
                        temp_can_dict['C_H'] = hid_dict['C_H']
                    _candidates.append(temp_can_dict)
                    k_count += 1

                if (len(word.decode(
                        'utf-8')) + len_c == count_accum) and not do_not_sen:  # and len(word.decode('utf-8'))<i-1:
                    i_count_f = i_count
                    if options['not_first'] != 0:
                        i_count_f += 1
                    else:
                        if options['lfirst'] == 1:
                            if options['dfirst'] != 1:
                                i_count_f += 1

                    if (options['not_first'] == 0 and options['lfirst'] == 1) and i_count_f == len(
                            options['predict_seq_len']) + 1:
                        i_count_f = 1

                    yun_lis = []
                    for yu_li in options['yun_list']:
                        if i_count_f in yu_li[1:] or (i_count_f in yu_li and 1 in yu_li):
                            yun_lis = yu_li
                            break

                    if options['hard_yun'] != 1:
                        yun_lis = []

                    if len(yun_lis) != 0:
                        if (not (options['not_first_yun'] == 1 and i_count_f == 1)) and (
                                i_count_f in yun_lis[1:] or (i_count_f in yun_lis and 1 in yun_lis)) and j > 1 and \
                                options['use_correspond'] != 1:
                            word1 = ''
                            if len(yun_lis) and yun_lis[0] == 1 and options['not_first_yun'] == 0:
                                word1 = l_sentence.split()[-1]
                                yunD0 = yunmuModel.getYunDiao(word1)
                            else:
                                lentt = 0
                                wordss = []
                                if options['cut_to_one']:
                                    wordss = full_words.replace(' /', '').split()
                                else:
                                    wordss = candidate['word'].replace(' /', ' ').split()

                                position = yun_lis.index(i_count_f)

                                len_before = 0

                                for x in range(yun_lis[position - 1] + 1, yun_lis[position] + 1):
                                    len_before += options['predict_seq_len'][x - 1]

                                if options['not_first'] == 0:
                                    if options['lfirst'] == 0:
                                        w_Y_count = count_accum - len_before
                                    else:
                                        if options['dfirst']:
                                            w_Y_count = count_accum - len_before
                                        else:
                                            if options['song_gen'] == 1:
                                                w_Y_count = count_accum - len_before
                                            else:
                                                w_Y_count = (count_accum - i) - len_before
                                else:
                                    if options['song_gen'] == 1:
                                        w_Y_count = count_accum - len_before
                                    else:
                                        w_Y_count = (count_accum - i) - len_before

                                for w in wordss:

                                    if w != ' ' and w != 'START' and w != '':
                                        lentt += len(w.decode('utf-8'))

                                    if lentt == w_Y_count:
                                        word1 = w
                                        break

                                yunD0 = yunmuModel.getYunDiao(word1)

                            if options['use_correspond'] == 1:

                                temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                                 'prob': candidate['prob'] * predict_word_id_list[id_t]}
                                temp_can_dict['S_H'] = hid_dict['S_H']
                                if options['use_lstm']:
                                    temp_can_dict['C_H'] = hid_dict['C_H']
                                sen_candidates.append(temp_can_dict)
                                k_count_sen += 1
                            else:
                                if options['use_pingshuiyun'] == 0:
                                    if (yunD0['y'] == yun0['y']):
                                        temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                                         'prob': candidate['prob'] * predict_word_id_list[id_t]}
                                        temp_can_dict['S_H'] = hid_dict['S_H']
                                        if options['use_lstm']:
                                            temp_can_dict['C_H'] = hid_dict['C_H']
                                        sen_candidates.append(temp_can_dict)
                                        k_count_sen += 1
                                else:
                                    if yunmuModel.yapingshui(word1, word):
                                        if options['use_pingshuiyun_morden'] == 0:
                                            temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                                             'prob': candidate['prob'] * predict_word_id_list[id_t]}
                                            temp_can_dict['S_H'] = hid_dict['S_H']
                                            if options['use_lstm']:
                                                temp_can_dict['C_H'] = hid_dict['C_H']
                                            sen_candidates.append(temp_can_dict)

                                            k_count_sen += 1
                                        else:
                                            if (yunD0['y'] == yun0['y']):
                                                temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                                                 'prob': candidate['prob'] * predict_word_id_list[id_t]}
                                                temp_can_dict['S_H'] = hid_dict['S_H']
                                                if options['use_lstm']:
                                                    temp_can_dict['C_H'] = hid_dict['C_H']
                                                sen_candidates.append(temp_can_dict)

                                                k_count_sen += 1

                        else:
                            temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                             'prob': candidate['prob'] * predict_word_id_list[id_t]}
                            temp_can_dict['S_H'] = hid_dict['S_H']
                            if options['use_lstm']:
                                temp_can_dict['C_H'] = hid_dict['C_H']
                            sen_candidates.append(temp_can_dict)

                            k_count_sen += 1
                    else:
                        temp_can_dict = {'word': candidate['word'] + ' ' + word,
                                         'prob': candidate['prob'] * predict_word_id_list[id_t]}
                        temp_can_dict['S_H'] = hid_dict['S_H']
                        if options['use_lstm']:
                            temp_can_dict['C_H'] = hid_dict['C_H']
                        sen_candidates.append(temp_can_dict)
                        k_count_sen += 1

            else:
                print('no top k allowed')
                if (len(word.decode(
                        'utf-8')) + len_c == count_accum - i + 1) and not do_not_w:  # and len(word.decode('utf-8'))<i-1:
                    if j != 0:
                        _candidates.append({'word': candidate['word'] + ' ' + word,
                                            'prob': candidate['prob'] * predict_word_id_list[id_t]})
                        k_count += 1
                    else:
                        i_count_f = i_count
                        i_count_f -= 1

                        yun_lis = []
                        for yu_li in options['yun_list']:
                            if len(options['predict_seq_len']) - i_count_f in yu_li[::-1][1:] or (
                                    len(options['predict_seq_len']) - i_count_f in yu_li and 1 in yu_li):
                                yun_lis = yu_li
                                break

                        if options['hard_yun'] != 1:
                            yun_lis = []

                        if len(yun_lis) != 0 and (
                                not (options['not_first_yun'] == 1 and len(
                                    options['predict_seq_len']) - i_count_f == 1)) and (
                                len(options['predict_seq_len']) - i_count_f in yun_lis[::-1][1:] or (len(
                            options['predict_seq_len']) - i_count_f in yun_lis and 1 in yun_lis)) and j < 1 and \
                                options['use_correspond'] != 1:
                            if len(yun_lis) and yun_lis[0] == 1 and options['not_first_yun'] == 0:
                                yunD0 = yunmuModel.getYunDiao(l_sentence.split()[-1])
                            else:
                                word1 = ''
                                lentt = 0
                                wordss = []
                                if options['cut_to_one']:
                                    wordss = full_words.replace(' /', '').split()
                                else:
                                    wordss = candidate['word'].replace(' /', ' ').split()

                                position = yun_lis.index(len(options['predict_seq_len']) - i_count_f)

                                len_before = 0

                                for x in range(yun_lis[position] + 1, yun_lis[position + 1] + 1):
                                    len_before += options['predict_seq_len'][x - 1]
                                if options['not_first'] == 0:
                                    if options['lfirst'] == 0:
                                        w_Y_count = count_accum - i + 1 - len_before
                                    else:
                                        if options['song_gen'] == 1:
                                            w_Y_count = count_accum - i + 1 - len_before
                                        else:
                                            w_Y_count = (count_accum - i) - len_before
                                else:
                                    if options['song_gen'] == 1:
                                        w_Y_count = count_accum - i + 1 - len_before
                                    else:
                                        w_Y_count = (count_accum - i) - len_before

                                for w in wordss:

                                    if w != ' ' and w != 'START' and w != '':
                                        lentt += len(w.decode('utf-8'))

                                    if lentt == w_Y_count:
                                        word1 = w
                                        break

                                yunD0 = yunmuModel.getYunDiao(word1)

                        NOT_IN = 1
                        for yu_li in options['yun_list']:
                            if len(options['predict_seq_len']) - i_count_f in yu_li[::-1][1:] or (
                                    len(options['predict_seq_len']) - i_count_f in yu_li and 1 in yu_li and (
                                    not (options['not_first_yun'] == 1 and len(
                                        options['predict_seq_len']) - i_count_f == 1))):
                                NOT_IN = 0

                        if options['hard_yun'] == 0:
                            NOT_IN = 1

                        if NOT_IN or options['use_correspond'] == 1 or (
                                not NOT_IN and (yunD0['y'] == yun0['y'] or yunD0['y'] == 'aaaa')):
                            _candidates.append({'word': candidate['word'] + ' ' + word,
                                                'prob': candidate['prob'] * predict_word_id_list[id_t]})
                            k_count += 1

                elif (len(word.decode(
                        'utf-8')) + len_c < count_accum) and not do_not_w:
                    _candidates.append({'word': candidate['word'] + ' ' + word,
                                        'prob': candidate['prob'] * predict_word_id_list[id_t]})
                    k_count += 1

                if (len(word.decode(
                        'utf-8')) + len_c == count_accum) and not do_not_sen:
                    sen_candidates.append({'word': candidate['word'] + ' ' + word,
                                           'prob': candidate['prob'] * predict_word_id_list[id_t]})
                    k_count_sen += 1

            if k_count >= options['top_k']:
                do_not_w = 1
            if k_count_sen >= options['top_k']:
                do_not_sen = 1

            if do_not_sen or do_not_w:
                break

            if k_count_sen + k_count > options['cut_out_sort']:
                break

            if k_count >= options['top_k'] or k_count_sen >= options['top_k']:
                break

    predictls_temp = predictls_temp[::-1]
    prob_idt = predict_word_id_list[predictls_temp[top_count]]
    prob_max = predict_word_id_list[predictls_temp[0]]
    return sen_candidates, _candidates, yunD0, top_count, prob_idt, prob_max


class yunLv(object):
    """docstring for yunMu"""

    # sheng = ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','r','zh','ch','sh']
    def __init__(self, options):

        file_path = options['yunmu_file']
        pingshui_yun_file_ping = options['pingshui_file'][1]
        pingshui_yun_file_ze = options['pingshui_file'][0]
        pingshui_yun_file_ping_new = options['pingshui_file_new'][1]
        pingshui_yun_file_ze_new = options['pingshui_file_new'][0]
        hanzipinyin_file_path = options['hanzipinyin_file']

        # hanzipinyin
        yun_constrain = [['zhi', 'chi', 'shi'], ['zi', 'ci', 'si']]
        self.hanzipinyin = [set([]) for i in range(len(yun_constrain))]

        with open(hanzipinyin_file_path) as f:
            lines = f.readlines()
            for l in lines:
                l_temp = l.strip()
                l_split = l_temp.split(',')
                for i in range(len(yun_constrain)):
                    if l_split[1] in yun_constrain[i]:
                        self.hanzipinyin[i].add(l_split[0])

        # yundiao
        if 1:
            vocSet = set([])  # all words
            yunDict = {}  # all yun

            # get pingze, yun for each word (11729)
            f = open(file_path).readlines()
            for l in f:
                ls = l.split()
                if len(ls) == 3:
                    word = ls[0]
                    pingZ = ls[1]
                    yun = ls[2]

                    vocSet.add(word)
                    yunDict[word] = {'p': pingZ, 'y': yun}
                else:
                    pass

            self.vocSet = vocSet
            self.yunDict = yunDict

        if 1:
            # vocab set in pingshuiyun
            self.vocSet_ping = set([])
            for f in options['pingshui_file']:
                for l in open(f).readlines():
                    for w in l.decode('utf-8'):
                        self.vocSet_ping.add(w.encode('utf-8'))

            # yunlist, used in function getYunLineLen
            self.pzlist = {}
            self.pzlist['p'] = set([])
            self.pzlist['z'] = set([])
            self.pzlist['0'] = set([])

            self.yunlist = []

            for l in open(pingshui_yun_file_ze).readlines():
                temp_list = set([])
                for w in l.decode('utf-8'):
                    self.pzlist['z'].add(w.encode('utf-8'))
                    temp_list.add(w.encode('utf-8'))
                self.yunlist.append(temp_list)

            for l in open(pingshui_yun_file_ping).readlines():
                temp_list = set([])
                for w in l.decode('utf-8'):
                    self.pzlist['p'].add(w.encode('utf-8'))
                    temp_list.add(w.encode('utf-8'))
                self.yunlist.append(temp_list)

            # yunlist_new, used in function yapingshui
            self.yunlist_new = []
        self.yun_pzlist = {}
        self.yun_pzlist['p'] = set([])
        self.yun_pzlist['z'] = set([])

        for l in open(pingshui_yun_file_ze_new).readlines():
            temp_list = set([])
            for w in l.decode('utf-8').strip():
                self.yun_pzlist['z'].add(w.encode('utf-8'))
                temp_list.add(w.encode('utf-8'))
            self.yunlist_new.append(temp_list)

            for l in open(pingshui_yun_file_ping_new).readlines():
                temp_list = set([])
                for w in l.decode('utf-8').strip():
                    self.yun_pzlist['p'].add(w.encode('utf-8'))
                    temp_list.add(w.encode('utf-8'))
                self.yunlist_new.append(temp_list)

    def getYunDiao(self, x):
        y = 'aaaa'
        p = '-1'
        if x in self.vocSet:
            p = self.yunDict[x]['p']
            y = self.yunDict[x]['y']
        return {'y': y, 'p': p}

    def yapingshui(self, word1, word):
        sig_bool = False
        for temp_set in self.yunlist_new:
            if (word1 in temp_set) and (word in temp_set):
                sig_bool = True
                for yun_cons_set in self.hanzipinyin:
                    if (word1 in yun_cons_set) and (word not in yun_cons_set):
                        sig_bool = False
                    if (word1 not in yun_cons_set) and (word in yun_cons_set):
                        sig_bool = False
        return sig_bool

    def getYunLineLen(self, word):
        for temp_set in self.yunlist:
            if word in temp_set:
                return len(temp_set)
        return 0


def create_model(session, is_predict):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        is_predict=is_predict, cell_initializer=tf.constant_initializer(np.array(P_Emb)))
    # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    return model


def prepare_data_for_prediction(options, test_file, theme_file):
    '''
    :param options:
    :param test_file: test_poem_58k.txt
    :param theme_file: head_song.txt for cangtou
    :return: test: list of testset, target, theme_list, type_list
    '''
    print('Predict: loading data')
    test = []
    target = []
    theme_list = []
    type_list = []

    if 1:
        for l in test_file: # one line
            if l != '\n':
                l_temp = l.split('\t')[0]
                if options['reverse_training']:
                    str_temp = ' '.join(l_temp.rstrip().split(' ')[::-1])
                    test.append(str_temp)
                else:
                    test.append(l_temp)
                target.append(l.replace('\t', '    '))

    if len(options['type_len_format']):  # totally 51 types (0-50). 49: '7-7-7-7', 50: '5-5-5-5'
        for l in test_file:
            # print(options['force_type'])
            if options['force_type'] == '':
            # e.g. 测试数据行 '溪 居 \t 御风 \t 遗世独立 \t 羽化登仙' 得到temp_str='2-2-4-4'
            # 但是因为不在type_len_format中，默认为'5-5-5-5'
            # 问题：必须要测试数据是需要生成的格式。或者用force_type指定。
                l = l.rstrip().lstrip()
                temp_str = ''
                for ll in l.split('\t'):
                    lenn = str(len(ll.replace(' ', '').decode('utf-8')))
                    temp_str += lenn
                    temp_str += '-'
                temp_str = temp_str[:-1]
                if temp_str != '' and temp_str in options['type_len_format']:
                    type_list.append(options['type_len_format'].index(temp_str))  # index() if contain temp_str
                elif temp_str == '':
                    print(l)
                    return
                else:
                    type_list.append(len(options['type_len_format']) - 1)
            else:
                type_list.append(options['type_len_format'].index(options['force_type']))
                # add the index of force_type to type_list
                # if poem7, then type_list = [49, 49, 49, 49, ...]

    if len(type_list) != len(test):
        print('type len err ' * 80)
        return

    for l in theme_file:
        theme_list.append(l.split('\t')[0])

    print('Predict: loading parameters')

    if options['use_little_predict']:
        test = test[int(options['train_percentage'] * len(test)):]
        target = target[int(options['train_percentage'] * len(target)):]
        type_list = type_list[int(options['train_percentage'] * len(type_list)):]

    # print('data process:')
    # print(test)
    # print(len(test))
    # print(type_list)

    return test, target, theme_list, type_list


def predict_process(options, data_set, sess, model, bucket_id=0, is_feed=True):
    print('Predict results:')
    test, target, theme_list, type_list = data_set
    # test: testset, including keyword inputs.
    # type_list: type of each sentence expected in testset. usually, the types in one testset are the same.
    # e.g. [49, 49, 49] (testset contains 3 sentences, all expecting 49th type, which is type='7-7-7-7')

    return_list = []  # all poems
    return_str = ''  # one poem
    global yunmuModel
    yunmuModel = yunLv(options)
    ci_list = open(options['ci_list']).readlines()
    predict_file = options['predict_file']  # dir of prediction results

    # write prediction results into a file
    if os.path.exists(predict_file):
        os.remove(predict_file)
    pfile = open(predict_file, 'w')

    lentest = len(test)
    for t in range(lentest):
        isFirst = True
        theme = ''  # remain a problem, no use
        l_sentence = test[t]
        type_sig = type_list[t]
        # print('type_sig')
        # print(type_sig)
        charac_num = FLAGS.vocab_size
        attention_state = []
        state = []
        mem_state = []

        if options['use_se']:  # default=1
            if options['use_two_start']:  # default=1
                sen_w = ['START1'] + l_sentence.split() + ['END1']
        else:
            sen_w = l_sentence.split()
        in_sentences_ids = [word2id.get(w, charac_num - 1) for w in sen_w]

        pre_words_e = []

        aij_max = []

        top_chose_list = []
        prob_idt_list = []
        prob_max_list = []

        x_in = []

        for q in in_sentences_ids:
            x_in.append(q)

        first_w = 'START'
        pre_words_e.append(word2id.get(first_w, charac_num - 1))

        return_str = l_sentence.lstrip().rstrip()

        # write number in result file
        if options['write_file']:
            pfile.write(str(t + 1))
            pfile.write('\t')

        # encoder process, get Atinput,h_left,h_right
        sen_candidates = []
        temp_dict_candidate = {'word': 'START', 'prob': 1.0, 'S_H': [0.0] * 500}
        temp_dict_candidate['C_H'] = [0.0] * 500
        sen_candidates.append(temp_dict_candidate)

        # Empty list, but cannot delete, StartWords is used, remain a problem with theme
        startWords = []  # theme.split()
        for s in theme.split():
            s = s.decode('utf-8')
            startWords.append(s)

        # PRINT NUMBER 
        print(str(t+1) + '/' + str(lentest))

        # maximum predict_2.0, default=5000
        if t > options['cut_predict']:
            break

        # idSet = set([0, 1, 2, 65, 2098, len(word2id.keys()) - 1, len(word2id.keys()) - 2])
        idSet = {0, 1, 2, len(word2id.keys()), len(word2id.keys()) - 1}
        for stop_word in options['stop_words']:  # default=['的']
            idSet.add(word2id.get(stop_word, 1))

        if options['use_two_start']:
            idSet.add(3)
            idSet.add(4)

        count_accum = 0
        i_count = 0

        # the length format of the poem. 
        ran = []
        if len(startWords):  # default=0
            ran = range(len(startWords))
        else:
            if options['not_first'] == 0:  # default=0
                if options['lfirst'] == 1:  # default=1
                    if options['dfirst'] == 0:  # default=1
                        ran = options['predict_seq_len'][1:] + [options['predict_seq_len'][-1]]
                    else:
                        ran = options['predict_seq_len'] + [options['predict_seq_len'][-1]]  #
                else:
                    ran = options['predict_seq_len']
            else:
                ran = options['predict_seq_len'][1:]

        if options['75_gen'] != 0:  # default=0
            lll = len(l_sentence.split())
            if len(options['predict_seq_len']):
                lll = options['predict_seq_len'][0]
            ran = []
            if len(startWords):
                for i in range(len(startWords)):
                    ran.append(lll)
            else:
                for i in range(options['75_gen']):
                    ran.append(lll)

        if options['song_gen'] == 1:  # default=1
            if options['not_first'] == 1:
                ran = options['predict_seq_len'][1:]
            else:
                if options['lfirst'] == 1:  # default=1
                    if options['dfirst'] == 0:  # default=1
                        ran = options['predict_seq_len'][1:] + [options['predict_seq_len'][-1]]
                    else:
                        ran = options['predict_seq_len'] + [options['predict_seq_len'][-1]]
                else:
                    ran = options['predict_seq_len']

        # predict a sentence
        full_candidates = ''
        for s_len in ran:  # s_len: length of each sentence in ran
            if len(startWords):
                if len(options['predict_seq_len']):
                    i = options['predict_seq_len'][0]
                else:
                    i = len(l_sentence.split())
            else:
                i = s_len  # i: length of the sentence

            candidates = sen_candidates
            sen_candidates = []
            count_accum += i
            i_count += 1

            # iteratively predict one word in a sentence
            for j in range(0, i):  # i: length of the sentence
                _candidates = []
                for c in candidates:
                    len_c = 0
                    for w in c['word'].split():
                        if w != 'START' and w != '/':
                            len_c += len(w.decode('utf-8'))

                    pre_words_e = []
                    for w in c['word'].split():
                        if (w != 'START'):
                            pre_words_e.append(word2id.get(w, charac_num - 1))

                    hid_dict = {}

                    # Get a 1-element batch to feed the sentence to the model.
                    encoder_inputs, reverse_encoder_inputs, decoder_inputs, target_weights, sequence_length, batch_encoder_weights, sig_list = model.get_batch(
                        {bucket_id: [(x_in, pre_words_e if len(pre_words_e) == 0 else [pre_words_e[-1]])]}, bucket_id,
                        0, P_sig)

                    # Get output logits for the sentence.
                    # for the first word
                    if isFirst == True:
                        encoder_outputs = model.step(sess, encoder_inputs, reverse_encoder_inputs, decoder_inputs,
                                                     batch_encoder_weights, target_weights, sequence_length, bucket_id,
                                                     1.0, options['memory_weight'], True, False,
                                                     sig_weight=np.array([P_sig[type_sig]]), isFirst=isFirst)
                        attention_state = encoder_outputs[0]
                        state = encoder_outputs[1]
                        mem_state = np.zeros(encoder_outputs[1].shape, np.float32)
                        isFirst = False

                    # for the following words
                    if len(pre_words_e) != 0 and pre_words_e[-1] == 2:
                        temp_decoder_inputs = [d.copy() for d in decoder_inputs]
                        temp_decoder_inputs[0][0] = pre_words_e[-2]

                        temp_decoder_outputs = model.step(sess, encoder_inputs, reverse_encoder_inputs,
                                                          temp_decoder_inputs,
                                                          batch_encoder_weights, target_weights, sequence_length,
                                                          bucket_id,
                                                          1.0, options['memory_weight'], True, False,
                                                          sig_weight=np.array([P_sig[type_sig]]), isFirst=isFirst,
                                                          attention_state=attention_state, state=state,
                                                          mem_state=mem_state)
                        state = temp_decoder_outputs[1]
                        mem_state = temp_decoder_outputs[2]

                    decoder_outputs = model.step(sess, encoder_inputs, reverse_encoder_inputs, decoder_inputs,
                                                 batch_encoder_weights, target_weights, sequence_length, bucket_id,
                                                 1.0, options['memory_weight'], True, False,
                                                 sig_weight=np.array([P_sig[type_sig]]), isFirst=isFirst,
                                                 attention_state=attention_state, state=state, mem_state=mem_state)
                    state = decoder_outputs[1]
                    mem_state = decoder_outputs[2]

                    predict_word_id_list = decoder_outputs[0][0]

                    S_H_temp = state[0][0]
                    C_H_temp = state[1][0]

                    c['S_H'], c['C_H'] = S_H_temp, C_H_temp
                    hid_dict['S_H'] = S_H_temp
                    hid_dict['C_H'] = C_H_temp

                    aij_max.append([0])

                    predictls_temp = predict_word_id_list.argsort(axis=0)

                    full_words = full_candidates + c['word'].replace('START', '')

                    # get_next_id
                    sen_candidates, _candidates, yunD1, id_top, prob_idt, prob_max = \
                        get_next_id(startWords, idSet, predictls_temp, c, _candidates, sen_candidates,
                                    yunmuModel, count_accum, i_count, j, i, len_c, word2id, id2word,
                                    options['lv_list'], charac_num, predict_word_id_list, options,
                                    l_sentence, full_words, ran, ci_list, hid_dict)

                    top_chose_list.append(str(id_top))
                    prob_idt_list.append(round(prob_idt, 2))
                    prob_max_list.append(round(prob_max, 2))

                    if yunD1 != {}:
                        yunD0 = yunD1

                # print('len of _candidates:')
                # print(len(_candidates))  # depends on top_k. if top_k=2, len=4

                '''
                10/10
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                0
                sen_candidates:
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                0
                sen_candidates:
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                0
                sen_candidates:
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                0
                sen_candidates:
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                1
                len of _candidates:
                0
                sen_candidates:
                红尘妃子笑倾城====红尘怜子笑	便欲出江城	绿玉共青山	双眼碧云横
                '''

                candidates = sorted(_candidates, key=lambda k: k['prob'])[-1 * options['top_k']:]
                # top_k = 1

                # e.g. candidates: list of dic{'S_H':array[dim=500], 'word':'START 红 尘 怜 子 笑 / 便 欲 出 江 城 / 绿 玉 共 青',
                # 'prob': 9.249938479933126e+23, 'C_H':array[dim=500]}

                # print('candidates:')
                # print(type(candidates))
                # for can in candidates:
                    # print(can['word'])

            sen_candidates = sorted(sen_candidates, key=lambda k: k['prob'])[-1 * options['top_k']:]
            # top_k = 1

            # print('sen_candidates:')
            # print(type(sen_candidates)) # list
            # print(len(sen_candidates)) # 1 or 0(for the case: Cannot generate poem meeting all requirements)
            # for can in sen_candidates:
            #     print(can['word'])

            '''  
            sen_candidates:
            START 红 尘 怜 子 笑
            sen_candidates:
            START 红 尘 怜 子 笑 / 便 欲 出 江 城
            sen_candidates:
            START 红 尘 怜 子 笑 / 便 欲 出 江 城 / 绿 玉 共 青 山
            sen_candidates:
            START 红 尘 怜 子 笑 / 便 欲 出 江 城 / 绿 玉 共 青 山 / 双 眼 碧 云 横
            sen_candidates:
            START 红 尘 怜 子 笑 / 便 欲 出 江 城 / 绿 玉 共 青 山 / 双 眼 碧 云 横 / 色 胜 春 光 满
            '''

            try:
                sen_candidates[0]['word'] = sen_candidates[0]['word'] + ' /'
            except:
                print('Cannot generate poem meeting all restrictions')
                pfile.write('\n')
                break

            top_chose_list.append('-')
            prob_idt_list.append(0)
            prob_max_list.append(0)

        # for each sentence candidates
        for c in sen_candidates:
            str_1 = c['word'].replace('START', '').rstrip() + '\n'
            str_1 = str_1.replace('/\n', '')
            str_1 = str_1.replace('\n', '')

            if options['not_first'] == 0:
                if options['lfirst']:
                    if options['dfirst']:
                        if options['use_fgen']:
                            str_1_temp = str_1.lstrip().rstrip().split(' / ')[-1].lstrip().rstrip()
                            str_1 = str_1.replace(' / ' + str_1_temp, '')
                        elif options['use_dgen']:
                            str_1 = str_1
                        else:
                            str_2_temp = str_1.lstrip().rstrip().split(' / ')[0].lstrip().rstrip()
                            str_1_temp = str_1.lstrip().rstrip().split(' / ')[-1].lstrip().rstrip()
                            str_1 = str_1_temp + ' / ' + str_1.replace(str_2_temp + ' / ', '').replace(
                                ' / ' + str_1_temp, '')
                    else:
                        str_1_temp = str_1.lstrip().rstrip().split(' / ')[-1].lstrip().rstrip()
                        str_1 = str_1_temp + ' / ' + str_1.replace(' / ' + str_1_temp, '')

            if options['reverse_training']:  # default=0
                str_1 = ' '.join(str_1.lstrip().rstrip().split()[::-1])

                # temp_ss: temp sentence candidate
            temp_ss = str_1.lstrip().rstrip()
            if len(temp_ss.split(' / ')) == 4 and options['topic_gen'] == 0:  # default=1
                tt = temp_ss.split(' / ')[0] + ' / '
                temp_ss = temp_ss.replace(tt, '').lstrip().rstrip()

            # add sentence to poem
            if options['print_param']:
                return_str += '====' + temp_ss
            else:
                return_str = '====' + temp_ss

            # add poem to all poem list
            return_list.append(return_str.replace(' / ', '\t').replace(' ', '') + '\n')
            # print(len(return_list))

            # PRINT THE POEM 
            if options['print_param']:
                print(
                    return_str.replace(' / ', '\t').replace(' - ', '\t').replace(' ', '') + '\n')  # PRINT THE POEM !!!!

            # write in result file
            if options['write_file']:
                pfile.write(return_str.replace(' - ', '\t').replace(' ', ''))
            sys.stdout.flush()

            # clear for next poem
            return_str = ''

            if options['write_file']:
                pfile.write('\n')

    return return_list


def checkMemory():
    if not os.path.exists('../resource/memory_resource/npy/' + sys.argv[1] + '_' +
                          sys.argv[2] + '_memory.npy'):
        memoryModule_decoder.createMemory()
        tf.reset_default_graph()


def predict():
    print('Predict begin')
    all_type = ['ymr', 'dlh', 'jzmlh', 'djc', 'zgt', 'psm', 'yja', 'poem5', 'poem7', 'test']
    # all_type = ['poem5', 'poem7']

    checkMemory()
    # check if memory is prepared and create memory if not exist

    if (sys.argv[5] in all_type):
        poem_type = sys.argv[5]
        options = data_utils.get_options(poem_type, 'poem_58k')
    else:
        print('Poem type input not available. Default poem_type=poem7')
        poem_type = 'poem7'
        options = data_utils.get_options(poem_type, 'poem_58k')

    with tf.Session() as sess:
        print("Predict: building the model")
        model = create_model(sess, True)
        path = os.getcwd() + '/model'

        list_file = [sys.argv[1]]
        for f in list_file:
            print("Predict: reading model parameters from %s" % f)
            sess.run(tf.initialize_all_variables())
            model.saver.restore(sess, path + '/' + f)

            tmp_str = ''
            for i in options['predict_seq_len']:
                tmp_str += str(i)
                tmp_str += '-'
            tmp_str = tmp_str[:-1]
            options['force_type'] = tmp_str
            options['type_len_format'].append(tmp_str)

            if (options['poem_type'] == 'poem5'):
                options['yun_list'] = [[2, 4]]  # default=[1,2,4]
                if options['use_all_lv']:  # default=1 强平仄，每个字都有要求 （两种）
                    # options['lv_list'] = [['p', 'p', 'z', 'z', 'p'], ['z', 'z', 'z', 'p', 'p'],
                    # ['z', 'z', 'p', 'p', 'z'], ['p', 'p', 'z', 'z', 'p']]
                    options['lv_list'] = [['z', 'z', 'p', 'p', 'z'], ['p', 'p', 'z', 'z', 'p'],
                                          ['p', 'p', 'p', 'z', 'z'], ['z', 'z', 'z', 'p', 'p']]

            elif (options['poem_type'] == 'poem7'):
                options['yun_list'] = [[2, 4]]
                if options['use_all_lv']:  # 平仄强规则 （两种）
                    # options['lv_list'] = [['z', 'z', 'p', 'p', 'z', 'z', 'p'], ['z', 'p', 'p', 'z', 'z', 'p', 'p'],
                    # ['p', 'p', 'z', 'z', 'p', 'p', 'z'], ['z', 'z', 'p', 'p', 'z', 'z', 'p']]
                    options['lv_list'] = [['p', 'p', 'z', 'z', 'p', 'p', 'z'], ['z', 'z', 'p', 'p', 'z', 'z', 'p'],
                                          ['z', 'z', 'p', 'p', 'p', 'z', 'z'], ['p', 'p', 'z', 'z', 'z', 'p', 'p']]

            else:
                options['hard_yun'] = 0  # 非绝句、律诗，取消押韵限制

                # for memory_weight_index in range(0, int(sys.argv[4])):
                #     data_set = prepare_data_for_prediction(options, open(options['test_in_file']).readlines(),
                #                                            open(options['test_head_file']).readlines())  #
                #     predict_process(options, data_set, sess, model, is_feed=True)

            data_set = prepare_data_for_prediction(options, open(options['test_in_file']).readlines(),
                                                   open(options['test_head_file']).readlines())
            predict_process(options, data_set, sess, model, is_feed=True)


def main(_):
    predict()


if __name__ == "__main__":
    tf.app.run()
