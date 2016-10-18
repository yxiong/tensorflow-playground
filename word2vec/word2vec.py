#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Sep 23, 2016.

from __future__ import division

import numpy as np
import os.path
import threading
import time

import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec

from xy_python_utils.os_utils import mkdir_p

class Word2Vec:
    # NOTE: member variables prefixed with `tf_` indicate they are tensorflow nodes.
    def __init__(self, session, for_training, *args, **kw):
        self.session = session
        if for_training:
            self.init_for_training(*args, **kw)


    def init_for_training(
            self,
            training_data_filename,
            training_results_path,
            analogy_questions_filename,
            emb_dim = 200,                       # Embedding dimension
            batch_size = 16,                     # Batch size
            num_neg_samples = 100,               # Number of negative samples
            epochs_to_train = 15,                # Number of epochs to train
            learning_rate = 0.2,                 # Initial learning rate.
    ):
        mkdir_p(training_results_path)
        self.training_results_path = training_results_path
        # Create a skip-gram node.
        tf_words, tf_counts, tf_words_per_epoch, self.tf_epoch, self.tf_total_words_processed, tf_examples, tf_labels = gen_word2vec.skipgram(
            filename = training_data_filename,
            batch_size = batch_size,
            window_size = 5,
            min_count = 5,
            subsample = 1e-3)
        # Execute the first three nodes to get words, word frequency and words per epoch.
        words, word_freq, words_per_epoch = self.session.run(
            [tf_words, tf_counts, tf_words_per_epoch])
        self.setup_with_word_frequency(words, word_freq)
        Word2Vec.save_word_frequency(
            os.path.join(training_results_path, "vocab.txt"), self.words, self.word_freq)
        # Setup computation graph.
        self.setup_main_graph(emb_dim)
        # Build forward model for training.
        tf_true_logits, tf_sampled_logits = Word2Vec.forward_model_for_training(
            self.tf_embeddings, self.tf_softmax_weights_t, self.tf_softmax_bias,
            tf_examples, tf_labels,
            self.word_freq, batch_size, num_neg_samples)
        # Noise-contrastive estimation (NCE) loss
        self.tf_nce_loss = Word2Vec.nce_loss(tf_true_logits, tf_sampled_logits, batch_size)
        self.tf_nce_loss_summary = tf.scalar_summary("NCE loss", self.tf_nce_loss)
        # Optimizer.
        words_to_train = float(words_per_epoch * epochs_to_train)
        self.tf_learning_rate = learning_rate * tf.maximum(
            0.0001, 1.0 - tf.cast(self.tf_total_words_processed, tf.float32) / words_to_train)
        tf_optimizer = tf.train.GradientDescentOptimizer(self.tf_learning_rate)
        # Training step.
        self.tf_global_step = tf.Variable(0, name = "global_step")
        self.tf_train_step = tf_optimizer.minimize(
            self.tf_nce_loss,
            global_step = self.tf_global_step,
            gate_gradients = tf_optimizer.GATE_NONE)
        # Initialize all variables.
        tf.initialize_all_variables().run()
        # Summary writer and saver.
        self.tf_summary_writer = tf.train.SummaryWriter(training_results_path, self.session.graph)
        self.tf_saver = tf.train.Saver()
        # Build analogy evaluation graph and read analogy questions (used as validation).
        self.read_analogy_questions(analogy_questions_filename)
        self.build_analogy_evaluation_graph()
        # Analogy accuracy summary.
        self.tf_analogy_accuracy = tf.Variable(0.0)
        self.tf_analogy_accuracy_summary = tf.scalar_summary(
            "Analogy accuracy", self.tf_analogy_accuracy)


    def setup_with_word_frequency(self, words, word_freq):
        assert len(words) == len(word_freq)
        # Given the words and word frequency, do some common setup.
        self.words = words
        self.word_freq = word_freq
        # Create a map from word to id.
        self.word2id = {}
        for i, w in enumerate(self.words):
            self.word2id[w] = i


    @staticmethod
    def save_word_frequency(filename, words, word_freq):
        assert len(words) == len(word_freq)
        with open(filename, 'w') as fp:
            for w, freq in zip(words, word_freq):
                fp.write("%s:%d\n" % (w, freq))


    def setup_main_graph(self, emb_dim):
        # Create nodes for embedding and softmax.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / emb_dim
        self.tf_embeddings = tf.get_variable(
            "embedding", [len(self.words), emb_dim],
            initializer = tf.random_uniform_initializer(-init_width, init_width))
        # Softmax weight transposed: [vocab_size, emb_dim]
        self.tf_softmax_weights_t = tf.get_variable(
            "softmax_weights_transposed", [len(self.words), emb_dim],
            initializer = tf.constant_initializer(0))
        # Softmax bias: [emb_dim]
        self.tf_softmax_bias = tf.get_variable(
            "softmax_bias", [len(self.words)],
            initializer = tf.constant_initializer(0))


    @staticmethod
    def forward_model_for_training(embeddings, softmax_weight_t, softmax_bias,
                                   examples, true_labels,
                                   word_freq, batch_size, num_neg_samples):
        def softmax_weights_and_bias_lookup(labels):
            weights = tf.nn.embedding_lookup(softmax_weight_t, labels)
            bias = tf.nn.embedding_lookup(softmax_bias, labels)
            return weights, bias
        # NOTE: most variables in this method are tensorflow nodes, others are usually constants.
        # Lable matrix in mini-batch.
        labels_matrix = tf.reshape(tf.cast(true_labels, dtype = tf.int64), [batch_size, 1])
        # Negative sampling.
        neg_sample_labels, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = labels_matrix,
            num_true = 1,
            num_sampled = num_neg_samples,
            unique = True,
            range_max = len(word_freq),
            distortion = 0.75,
            unigrams = word_freq.tolist())
        # Embeddings for examples: [batch_size, emb_dim]
        example_embedding = tf.nn.embedding_lookup(embeddings, examples)
        # Lookup softmax weights and bias for true and negative labels: [N, embed_dim], [N].
        true_weights, true_bias = softmax_weights_and_bias_lookup(true_labels)
        neg_weights, neg_bias = softmax_weights_and_bias_lookup(neg_sample_labels)
        # True logits: [batch_size, 1]
        true_logits = tf.add(tf.reduce_sum(tf.mul(example_embedding, true_weights), 1),
                             true_bias,
                             name = "true_logits")
        # Sampled logits: [batch_size, num_neg_sampled]
        neg_logits = tf.add(tf.matmul(example_embedding,
                                      neg_weights,
                                      transpose_b=True),
                            tf.reshape(neg_bias, [num_neg_samples]),
                            name = "neg_logits")
        return true_logits, neg_logits


    @staticmethod
    def nce_loss(true_logits, sampled_logits, batch_size):
        # Cross-entropy(logits, labels)
        true_x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        sampled_x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))
        return (tf.reduce_sum(true_x_entropy) + tf.reduce_sum(sampled_x_entropy)) / batch_size


    def build_analogy_evaluation_graph(self, k=4):
        # NOTE: we usually need to keep at least top k >= 4 words, because 3 of them could be the
        # same as input a, b or c.
        # Input placeholders: [N]
        self.tf_analogy_a = tf.placeholder(dtype=tf.int32)
        self.tf_analogy_b = tf.placeholder(dtype=tf.int32)
        self.tf_analogy_c = tf.placeholder(dtype=tf.int32)
        # Normalized word embeddings: [vocab_size, emb_dim]
        normalized_embeddings = tf.nn.l2_normalize(self.tf_embeddings, 1)
        # The embedding of input: [N, emb_dim]
        a_emb = tf.gather(normalized_embeddings, self.tf_analogy_a)
        b_emb = tf.gather(normalized_embeddings, self.tf_analogy_b)
        c_emb = tf.gather(normalized_embeddings, self.tf_analogy_c)
        # Target embedding: [N, emb_dim]
        target = c_emb + (b_emb - a_emb)
        # Cosine distance between each pair of target and word: [N, vocab_size].
        dist = tf.matmul(target, normalized_embeddings, transpose_b=True)
        # For each question, find top k words: [N, k].
        _, self.tf_analogy_predict_idx = tf.nn.top_k(dist, k)


    def read_analogy_questions(self, analogy_questions_filename):
        questions = []
        num_questions_skipped = 0
        with open(analogy_questions_filename, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self.word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    num_questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        self.analogy_questions = np.array(questions, dtype=np.int32)
        self.num_analogy_questions_skipped = num_questions_skipped


    def train(self,
              concurrent_steps = 12,               # Number of concurrent training steps
              summary_interval = 60,               # Save training summary every n seconds
              evaluate_interval = 300,             # Evaluate analogy every n seconds
              checkpoint_interval = 600,           # Save model checkpoint every n seconds
    ):
        # Get initial epoch.
        initial_epoch = self.session.run(self.tf_epoch)
        # Spawn threads to perform parallel training.
        workers = []
        for _ in xrange(concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)
        # Print statistics and save checkpoints.
        epoch = initial_epoch
        last_summary_time = 0
        last_evaluate_time = 0
        last_checkpoint_time = 0
        while epoch == initial_epoch:
            time.sleep(summary_interval)
            [epoch, step] = self.session.run([self.tf_epoch, self.tf_global_step])
            now = time.time()
            if now - last_summary_time > summary_interval:
                summary_str = self.session.run(self.tf_nce_loss_summary)
                self.tf_summary_writer.add_summary(summary_str, step)
                self.tf_summary_writer.flush()
                last_summary_time = now
            if now - last_evaluate_time > evaluate_interval:
                summary_str = self.session.run(
                    self.tf_analogy_accuracy_summary,
                    feed_dict = {
                        self.tf_analogy_accuracy: self.evaluate_analogy()
                    })
                self.tf_summary_writer.add_summary(summary_str, step)
                self.tf_summary_writer.flush()
                last_evaluate_time = now
            if now - last_checkpoint_time > checkpoint_interval:
                self.tf_saver.save(self.session,
                                   os.path.join(self.training_results_path, "model.ckpt"))
                last_checkpoint_time = now
        # Wait for all worker to finish.
        for t in workers:
            t.join()


    def _train_thread_body(self):
        initial_epoch = self.session.run(self.tf_epoch)
        epoch = initial_epoch
        while epoch == initial_epoch:
            _, epoch = self.session.run([self.tf_train_step, self.tf_epoch])


    def evaluate_analogy(self):
        def is_correct(result, true_answer):
            for j in xrange(4):
                if idx[question, j] == sub[question, 3]:
                    # Match
                    return True
                elif idx[question, j] in sub[question, :3]:
                    # Skip words already in question
                    continue
                else:
                    # Did not match
                    return False

        correct = 0
        total = self.analogy_questions.shape[0]
        start = 0
        while start < total:
            limit = start + 2500
            sub = self.analogy_questions[start:limit, :]
            idx = self.predict_analogy(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                correct += is_correct(idx[question], sub[question, 3])
        return correct / total


    def predict_analogy(self, analogy):
        return self.session.run(
            self.tf_analogy_predict_idx,
            feed_dict = {
                self.tf_analogy_a: analogy[:, 0],
                self.tf_analogy_b: analogy[:, 1],
                self.tf_analogy_c: analogy[:, 2],
            })


    def load_checkpoint(self, save_path):
        self.tf_saver.restore(self.session, os.path.join(save_path, "model.ckpt"))
