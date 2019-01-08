# -*- coding: utf-8 -*-

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

"""
Binary for training poem generation models and decoding from them.
Running this program without --decode will load data from --data_dir
and start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint generates Chinese poems with different styles.
Our Chinese poem generation model is based on the following paper.

 * http://arxiv.org/abs/1705.03773v1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model

word2id, id2word, P_Emb, P_sig = data_utils.Word2vec()


# Hyper parameters.
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 80,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 500, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 4777, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 4777, "Target vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                                    os.path.split(os.path.dirname(__file__))[1],
                                                    'tmp'), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                                     os.path.split(os.path.dirname(__file__))[1],
                                                     'tmp'), "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 725,
                            "How many training steps to do per checkpoint.")
# tf.app.flags.DEFINE_boolean("decode", False,
#                             "Set to True for interactive decoding.")
# tf.app.flags.DEFINE_boolean("self_test", False,
#                             "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS

# We use buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# In this case, only one bucket is needed.
# _buckets = [(32, 41)]
_buckets = [(16, 40)]


def read_data(source_path):
    """Read data from source and target files and put into buckets.
    Args:
      source_path: path to the files with token-ids for the source language.
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        counter = 0
        charac_num = FLAGS.source_vocab_size
        for line in source_file.readlines():
            source, target = line.split('==')
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_words = []
            target_words = []

            source_words += ('START1 ' + source + ' END1').split(' ')

            target = target[:-2] if target.find('\r\n') > -1 else target[:-1]

            # for targets, repeat the first sentence at the end of the last sentence
            target_words += target.replace('\t', ' / ').split(' ') + ['/'] + target.split('\t')[0].split(' ')

            source_ids = [word2id.get(x, charac_num - 1) for x in source_words]
            target_ids = [word2id.get(x, charac_num - 1) for x in target_words]
            target_ids.append(data_utils.EOS_ID)

            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids) <= source_size and len(target_ids) <= target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break

    return data_set


def create_model(session, is_predict):
    """Create generation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.source_vocab_size, FLAGS.target_vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        is_predict=is_predict, cell_initializer=tf.constant_initializer(np.array(P_Emb, dtype=np.float32)))
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Training: reading model parameters from %s" % ckpt.model_checkpoint_path)
        session.run(tf.initialize_all_variables())
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Training: created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train a poem generation model using loaded training data."""

    with tf.Session() as sess:
        # Create model.

        print("Training: creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)
        # Read data into buckets and compute their sizes.
        print("Training: reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)

        sys.stdout.flush()

        train_set = read_data('../resource/train_resource/poem_58k_theme.txt')
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            # random_number_01 = 0.5
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            if (model.global_step.eval() % FLAGS.steps_per_checkpoint == 0):
                for i_th in range(len(train_buckets_scale)):
                    np.random.shuffle(train_set[0])

            batch_start_id = current_step % FLAGS.steps_per_checkpoint

            encoder_inputs, reverse_encoder_inputs, decoder_inputs, target_weights, sequence_length, batch_encoder_weights, type_list = \
                model.get_batch(train_set, bucket_id, batch_start_id, P_sig)
            step_loss, step_loss_1, step_loss_2 = model.step(sess, encoder_inputs, reverse_encoder_inputs,
                                                             decoder_inputs,
                                                             batch_encoder_weights, target_weights, sequence_length,
                                                             bucket_id, 1.0, False, False, sig_weight=type_list)
            step_time += (time.time() - start_time)
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            print(str(current_step) + 'th')
            print('loss: ' + "%.30f" % step_loss)
            # print(step_loss_1)
            # print(step_loss_2)

            sys.stdout.flush()

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % (FLAGS.steps_per_checkpoint) == 0:
                # Print statistics for the previous epoch.
                print("global step %d  step-time %.2f loss "
                      "%.2f" % (current_step / FLAGS.steps_per_checkpoint,
                                step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #   sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, str(
                    int(current_step / FLAGS.steps_per_checkpoint)) + "th_model.ckpt+cost=" + str(loss))
                # print (checkpoint_path)
                model.saver.save(sess, checkpoint_path)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()


def main(_):

    train()


if __name__ == "__main__":
    tf.app.run()
