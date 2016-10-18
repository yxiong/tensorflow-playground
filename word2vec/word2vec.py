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

class Word2Vec:
    # NOTE: member variables prefixed with `tf_` indicate they are tensorflow nodes.
    def __init__(self, session, training_data_filename,
                 emb_dim = 200,                       # Embedding dimension
                 batch_size = 16,                     # Batch size
                 num_neg_samples = 100,               # Number of negative samples
                 epochs_to_train = 15,                # Number of epochs to train
                 learning_rate = 0.2,                 # Initial learning rate.
    ):
        self.session = session
        # Create a skip-gram node.
        tf_words, tf_counts, tf_words_per_epoch, self.tf_epoch, self.tf_total_words_processed, tf_examples, tf_labels = gen_word2vec.skipgram(
            filename = training_data_filename,
            batch_size = batch_size,
            window_size = 5,
            min_count = 5,
            subsample = 1e-3)
        # Execute the first three nodes to get words, word frequency and words per epoch.
        self.words, self.word_freq, self.words_per_epoch = self.session.run(
            [tf_words, tf_counts, tf_words_per_epoch])
        # Create a map from word to id.
        self.word2id = {}
        for i, w in enumerate(self.words):
            self.word2id[w] = i
        # Create nodes for embedding and softmax.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / emb_dim
        self.tf_embeddings = tf.Variable(
            tf.random_uniform([len(self.words), emb_dim], -init_width, init_width),
            name = "embedding")
        # Softmax weight transposed: [vocab_size, emb_dim]
        tf_softmax_weights_t = tf.Variable(
            tf.zeros([len(self.words), emb_dim]),
            name = "softmax_weights_transposed")
        # Softmax bias: [emb_dim]
        tf_softmax_bias = tf.Variable(
            tf.zeros([len(self.words)]),
            name = "softmax_bias")
        # Build forward model for training.
        tf_true_logits, tf_sampled_logits = self.forward(
            self.tf_embeddings, tf_softmax_weights_t, tf_softmax_bias, tf_examples, tf_labels,
            batch_size, num_neg_samples)
        # Noise-contrastive estimation (NCE) loss
        self.tf_nce_loss = self.nce_loss(tf_true_logits, tf_sampled_logits, batch_size)
        self.tf_nce_loss_summary = tf.scalar_summary("NCE loss", self.tf_nce_loss)
        # Optimizer.
        words_to_train = float(self.words_per_epoch * epochs_to_train)
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
        # Create a saver.
        self.tf_saver = tf.train.Saver()

        # Build analogy evaluation graph.
        self.build_analogy_evaluation_graph()
        # Analogy accuracy summary.
        self.tf_analogy_accuracy = tf.Variable(0.0)
        self.tf_analogy_accuracy_summary = tf.scalar_summary(
            "Analogy accuracy", self.tf_analogy_accuracy)


    def forward(self, embedding, softmax_weight_t, softmax_bias, examples, labels,
                batch_size, num_neg_samples):
        # NOTE: most variables in this methid are tensorflow nodes, others are usually constants.
        # Lable matrix in mini-batch.
        labels_matrix = tf.reshape(tf.cast(labels, dtype = tf.int64), [batch_size, 1])
        
        # Negative sampling.
        neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = labels_matrix,
            num_true = 1,
            num_sampled = num_neg_samples,
            unique = True,
            range_max = len(self.words),
            distortion = 0.75,
            unigrams = self.word_freq.tolist())
        # Embeddings for examples: [batch_size, emb_dim]
        example_embedding = tf.nn.embedding_lookup(embedding, examples)
        # Weights for labels: [batch_size, emb_dim]
        true_weights = tf.nn.embedding_lookup(softmax_weight_t, labels)
        # Biases for labels: [batch_size, 1]
        true_bias = tf.nn.embedding_lookup(softmax_bias, labels)
        # Weights for sampled ids: [num_neg_sampled, emb_dim]
        sampled_weights = tf.nn.embedding_lookup(softmax_weight_t, neg_samples)
        # Biases for sampled ids: [num_neg_sampled, 1]
        sampled_bias = tf.nn.embedding_lookup(softmax_bias, neg_samples)
        # True logits: [batch_size, 1]
        true_logits = tf.add(tf.reduce_sum(tf.mul(example_embedding, true_weights), 1),
                             true_bias,
                             name = "true_logits")
        # Sampled logits: [batch_size, num_neg_sampled]
        sampled_logits = tf.add(tf.matmul(example_embedding,
                                          sampled_weights,
                                          transpose_b=True),
                                tf.reshape(sampled_bias, [num_neg_samples]),
                                name = "sampled_logits")
        return true_logits, sampled_logits


    def nce_loss(self, true_logits, sampled_logits, batch_size):
        # Cross-entropy(logits, labels)
        true_x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        sampled_x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))
        return (tf.reduce_sum(true_x_entropy) + tf.reduce_sum(sampled_x_entropy)) / batch_size


    def build_analogy_evaluation_graph(self):
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
        # For each question, find top 4 words
        _, self.tf_analogy_predict_idx = tf.nn.top_k(dist, 4)
        # Nearby words.
        self.tf_nearby_word = tf.placeholder(dtype=tf.int32)
        self.tf_nearby_emb = tf.gather(normalized_embeddings, self.tf_nearby_word)
        self.tf_nearby_dist = tf.matmul(self.tf_nearby_emb, normalized_embeddings, transpose_b=True)
        self.tf_nearby_val, self.tf_nearby_idx = tf.nn.top_k(
            self.tf_nearby_dist, min(1000, len(self.words)))


    def read_analogy_questions(self, analogy_data_file):
        questions = []
        questions_skipped = 0
        with open(analogy_data_file, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self.word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print "Eval analogy file:", analogy_data_file
        print "Questions:", len(questions)
        print "Skipped:", questions_skipped
        self.analogy_questions = np.array(questions, dtype=np.int32)


    def train(self, summary_writer, save_path,
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
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                last_summary_time = now
            if now - last_evaluate_time > evaluate_interval:
                summary_str = self.session.run(
                    self.tf_analogy_accuracy_summary,
                    feed_dict = {
                        self.tf_analogy_accuracy: self.evaluate_analogy()
                    })
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                last_evaluate_time = now
            if now - last_checkpoint_time > checkpoint_interval:
                self.tf_saver.save(self.session,
                                   os.path.join(save_path, "model.ckpt"))
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
        correct = 0
        total = self.analogy_questions.shape[0]
        start = 0
        while start < total:
            limit = start + 2500
            sub = self.analogy_questions[start:limit, :]
            idx = self.predict_analogy(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        # Match
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        # Skip words already in question
                        continue
                    else:
                        # Did not match
                        break
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

