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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import seq2seq
import data_utils
import third_party.rnn_cell as rnn_cell
import GlobalParams
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=True,
                 num_samples=4779, is_predict=False, cell_initializer=None):
        """Create the model.
        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          is_predict: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                  self.target_vocab_size)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        cell = single_cell
        if num_layers > 1:
            cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, reverse_encoder_inputs, decoder_inputs, sequence_length, encoder_weights,
                      decoder_weights, keep_prob, sig_weight, do_decode):
            return seq2seq.embedding_attention_seq2seq(
                encoder_inputs, reverse_encoder_inputs, decoder_inputs, cell, encoder_weights, decoder_weights,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=200,
                sig_weight=sig_weight,
                output_projection=output_projection,
                feed_previous=do_decode,
                sequence_length=sequence_length,
                output_keep_prob=keep_prob,
                cell_initializer=cell_initializer)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.reverse_encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.encoder_weights = []
        self.sequence_length = None
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_feed = tf.placeholder(tf.bool, shape=[], name='is_feed')

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                      name="encoder{0}".format(i)))
            self.reverse_encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                              name="reverse_encoder{0}".format(i)))
            self.encoder_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                                       name="encoder_weight{0}".format(i)))

        self.sequence_length = tf.placeholder(tf.int32, shape=[batch_size],
                                              name="sequence_length")
        self.sig_weight = tf.placeholder(tf.float32, shape=[batch_size, 200])
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses, self.attens, self.state_outputs, self.atten_inputs, self.hiddens, self.embed_inputs, self.state_fw, self.state_bw, self.reverse_embed_encoder_inputs, self.tmp \
            = seq2seq.model_with_buckets(
            self.encoder_inputs, self.reverse_encoder_inputs, self.decoder_inputs, targets,
            self.encoder_weights, self.target_weights, buckets, self.keep_prob, self.is_feed,
            lambda x, reverse_x, y, seq_len, encoder_weights, decoder_weights,
                   keep_prob, sig_weight, is_feed: seq2seq_f(x, reverse_x, y, seq_len, encoder_weights, decoder_weights,
                                                             keep_prob, sig_weight, is_feed),
            sequence_length=self.sequence_length, sig_weight=self.sig_weight,
            softmax_loss_function=softmax_loss_function, batch_size=self.batch_size, )

        def adadelta(params, grads, lr=np.float32(1.0), epsilon=np.float32(1e-6)):
            # rho = tf.Variable([0.95],trainable=False)
            # rho = tf.constant(0.95, dtype=tf.float32)
            # rho_r = tf.constant(0.05, dtype=tf.float32)
            # epsilon = tf.constant(epsilon,dtype=tf.float32)
            accums = []
            update_accums = []
            updates = []
            accums_upAction = []
            update_accums_upAction = []
            updates_upAction = []
            params_upAction = []
            grads_copy = []
            grads_copy_upAction = []
            grads_copy_cast = []
            params_copy = []
            params_copy_upAction = []
            params_copy_cast = []
            with vs.variable_scope('adadelta'):
                for index, item in enumerate(grads):
                    item_shape = item.get_shape()
                    init_value = init_ops.constant_initializer(np.zeros(item_shape, dtype='float32'))
                    accums.append(
                        vs.get_variable("accum_{0}".format(index), item_shape, initializer=init_value, trainable=False))
                    update_accums.append(
                        vs.get_variable("accum_update_{0}".format(index), item_shape, initializer=init_value,
                                        trainable=False))
                    updates.append(vs.get_variable("update_{0}".format(index), item_shape, initializer=init_value,
                                                   trainable=False))
                    grads_copy.append(
                        vs.get_variable("grads_{0}".format(index), item_shape, initializer=init_value, trainable=False))
                    params_copy.append(vs.get_variable("params_{0}".format(index), item_shape, initializer=init_value,
                                                       trainable=False))
                for i in range(len(grads)):
                    epsilon_tensor = tf.constant(1e-6, shape=grads[i].get_shape(), dtype=tf.float32)
                    rho = tf.constant(0.95, shape=grads[i].get_shape(), dtype=tf.float32)
                    rho_r = tf.constant(0.05, shape=grads[i].get_shape(), dtype=tf.float32)
                    # accums_upAction.append(tf.assign(accums[i],tf.add(rho*accums[i], rho_r*tf.square(grads[i]))))
                    grads_copy_upAction.append(tf.assign(grads_copy[i], grads[i]))
                    params_copy_upAction.append(tf.assign(params_copy[i], params[i]))
                    accums_upAction.append(
                        tf.assign(accums[i], tf.add(rho * accums[i], rho_r * tf.square(grads_copy[i]))))
                    updates_upAction.append(tf.assign(updates[i],
                                                      tf.sqrt(tf.add(update_accums[i], epsilon_tensor)) / tf.sqrt(
                                                          tf.add(accums[i], epsilon_tensor)) * grads_copy[i]))
                    update_accums_upAction.append(
                        tf.assign(update_accums[i], tf.add(rho * update_accums[i], rho_r * tf.square(updates[i]))))
                    params_upAction.append(
                        tf.assign(params[i], tf.sub(params_copy[i], tf.clip_by_value(updates[i], -1.0, 1.0))))
                    # accums[i] = tf.assign(accums[i],tf.add(rho*accums[i], rho_r*tf.square(grads[i])))
                    # updates[i] = tf.assign(updates[i],tf.truediv(tf.sqrt(tf.add(update_accums[i], epsilon_tensor)) , tf.sqrt(tf.add(accums[i], epsilon_tensor))) * grads[i])
                    # update_accums[i] = tf.assign(update_accums[i],tf.add(rho*update_accums[i], rho_r*tf.square(updates[i])))
                    # params[i] = tf.assign(params[i],tf.sub(params[i], tf.clip_by_value(updates[i],tf.constant(-1.0,dtype=tf.float32),tf.constant(1.0,dtype=tf.float32))))

            return accums_upAction, updates_upAction, update_accums_upAction, params_upAction, [epsilon_tensor, rho,
                                                                                                rho_r], grads_copy_upAction, params_copy_upAction
            # return accums, updates, update_accums, params

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not is_predict:
            self.gradient_norms = []
            self.updates = []
            self.ouput_gradients = []
            # self.accums = []
            # self.accum_updates = []
            # self.update_params = []
            # opt = tf.train.AdamOptimizer(1e-4)
            # opt = tf.train.AdadeltaOptimizer(0.001,epsilon=1e-8)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                self.accums, self.updates, self.update_accums, self.update_params, self.hyperparams, self.grads_copy_upAction, self.params_copy_upAction = adadelta(
                    params, gradients)

                # self.ouput_gradients.append(params)
                # gradients = tf.gradients(self.losses[b], params)
                # clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                # max_gradient_norm)
                # self.gradient_norms.append(norm)
                # self.updates.append(opt.apply_gradients(
                #    zip(gradients, params), global_step=self.global_step))
                # self.accums.append(opt.get_slot(params[0],'accum'))
                # self.accum_updates.append(opt.get_slot(params[0],'accum_update'))
                # self.update_params.append(opt.get_slot(params[0],'update'))

            # If we use output projection, we need to project outputs for decoding.
            # if output_projection is not None:
            # for b in xrange(len(buckets)):
            # self.outputs[b] = [
            # tf.matmul(output, output_projection[0]) + output_projection[1]
            # for output in self.outputs[b]
            # ]

        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=200)

    def step(self, session, encoder_inputs, reverse_encoder_inputs, decoder_inputs, encoder_weights, target_weights,
             sequence_length,
             bucket_id, keep_prob, is_predict, is_feed, sig_weight=None):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          is_predict: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.reverse_encoder_inputs[l].name] = reverse_encoder_inputs[l]
            input_feed[self.encoder_weights[l].name] = encoder_weights[l]
        input_feed[self.sequence_length.name] = sequence_length
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        input_feed[self.is_feed] = is_feed
        input_feed[self.sig_weight] = sig_weight
        # Output feed: depends on whether we do a backward step or not.
        if not is_predict:
            output_feed = [  # self.updates[bucket_id],
                # self.update_params[bucket_id],
                # self.accums[bucket_id],
                # self.accum_updates[bucket_id],
                # self.losses[bucket_id]
            ]  # Loss for this batch.
            output_feed_1 = self.accums
            output_feed_2 = self.updates
            output_feed_3 = self.update_accums
            output_feed_4 = self.update_params
            output_feed_5 = self.hyperparams
            output_feed_6 = self.grads_copy_upAction
            output_feed_7 = self.params_copy_upAction
            # output_feed += self.accums
            # output_feed += self.updates
            # output_feed += self.update_accums
            # output_feed += self.update_params
            # output_feed += [item_1, item_2, item_3, item_4 for item_1,item_2,item_3,item_4 in zip(self.accums,self.updates,self.update_accums,self.update_params)]
            # for i in range(len(self.accums)):
            #  output_feed.append(self.accums[i])
            #  output_feed.append(self.updates[i])
            #	output_feed.append(self.update_accums[i])
            #	output_feed.append(self.update_params[i])
            output_feed.append(self.losses[bucket_id])
            # for gradient_index in range(len(self.ouput_gradients[bucket_id])):
            #  output_feed.append(self.ouput_gradients[bucket_id][gradient_index])

            input_feed[self.keep_prob.name] = keep_prob
            # output_feed.append(self.embed_inputs)
        else:
            input_feed[self.keep_prob.name] = keep_prob
            output_feed = []
            output_feed.append(self.attens[bucket_id])
            output_feed.append(self.state_outputs[bucket_id])
            output_feed.append(self.losses[bucket_id])  # Loss for this batch.
            for l in xrange(decoder_size - 1):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            output_feed.append(self.atten_inputs[bucket_id])
            output_feed.append(self.hiddens[bucket_id])
            output_feed.append(self.embed_inputs[bucket_id])
            output_feed.append(self.state_fw[bucket_id])
            output_feed.append(self.state_bw[bucket_id])
            output_feed.append(self.reverse_embed_encoder_inputs[bucket_id])
            output_feed.append(self.tmp[bucket_id])
        if not is_predict:
            outputs = session.run(output_feed + output_feed_6 + output_feed_7, input_feed)
            # for item in outputs[1:]:
            #  print (item.flatten()[0:50])
            outputs_1 = session.run(output_feed_1, input_feed)
            outputs_2 = session.run(output_feed_2, input_feed)
            # outputs_3 = session.run(output_feed_3, input_feed)
            outputs_4 = session.run(output_feed_4, input_feed)
            outputs_5 = session.run(output_feed_3, input_feed)
            # for item in outputs[1:] + outputs_4 + outputs_5:
            #  print (item.flatten()[0:20])
        else:
            outputs = session.run(output_feed, input_feed)
        if not is_predict:
            return outputs[0], None, None  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2], outputs[3:-7], outputs[-7], outputs[-6], outputs[-5], outputs[
                -4], outputs[-3], outputs[-2], outputs[-1]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id, batch_start_id, p_sig):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, reverse_encoder_inputs, decoder_inputs, sequence_length, sig_list = [], [], [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i_relative_pos in xrange(self.batch_size):
            # print (batch_start_id*self.batch_size + i_relative_pos)
            # print (data[bucket_id][batch_start_id*self.batch_size + i_relative_pos])
            encoder_input, decoder_input = data[bucket_id][batch_start_id * self.batch_size + i_relative_pos]
            # sig_list.append(p_sig[49])

            if len(decoder_input) == 40:
                sig_list.append(p_sig[49])
            else:
                sig_list.append(p_sig[50])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            sequence_length.append(len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))

            reverse_encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)

            # batch_encoder_weights.append([1]*len(encoder_input) + [0]*len(encoder_size - len(encoder_input)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_reverse_encoder_inputs, batch_decoder_inputs, batch_weights, batch_encoder_weights = [], [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_reverse_encoder_inputs.append(
                np.array([reverse_encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # if(encoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID):
                if length_idx + 1 > sequence_length[batch_idx]:
                    batch_weight[batch_idx] = 0.0
            batch_encoder_weights.append(batch_weight)

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_reverse_encoder_inputs, batch_decoder_inputs, \
               batch_weights, np.array(sequence_length, dtype=np.int32), batch_encoder_weights, np.array(sig_list,
                                                                                                         dtype=np.float32)
