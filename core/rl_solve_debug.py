import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate,evaluate_captions,evaluate_for_particular_captions


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        val_captions = self.val_data['captions']
        n_iters_val = int(np.ceil(float(val_features.shape[0]) / self.batch_size))

        # build graphs for training model and sampling captions
        _ = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        alphas, betas, sampled_captions, _ = self.model.build_multinomial_sampler()

        _, _, greedy_caption = self.model.build_sampler(max_len=20)

        rewards = tf.placeholder(tf.float32, [None])
        base_line = tf.placeholder(tf.float32, [None])


        grad_mask = tf.placeholder(tf.int32, [None, 16])
        t1 = tf.expand_dims(grad_mask, 1)
        t1_mul = tf.to_float(tf.transpose(t1, [0, 2, 1]))

        # important step
        loss = self.model.build_loss()

        # train op
        with tf.name_scope('optimizer'):


            optimizer = self.optimizer(learning_rate=self.learning_rate)
            norm = tf.reduce_sum(t1_mul)
            r  =  rewards - base_line

            sum_loss = -  tf.reduce_sum(
                tf.transpose(tf.mul(tf.transpose(loss, [2, 1, 0]),r), [2, 1, 0]))/ norm

            grads_rl, _ = tf.clip_by_global_norm(tf.gradients(sum_loss, tf.trainable_variables(),aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), 5.0)

            grads_and_vars = list(zip(grads_rl,tf.trainable_variables()))

            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op


        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)



            start_t = time.time()

            with open(os.path.join(self.model_path, 'val.RandB.scores.txt'), 'w') as f:
                all_decoded_for_eval = []
                for k in range(n_iters_val):
                    captions_batch = np.array(val_captions[k * self.batch_size:(k + 1) * self.batch_size])
                    features_batch = val_features[k * self.batch_size:(k + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}

                    greedy_words = sess.run(greedy_caption,
                                            feed_dict)
                    _, greedy_decoded = decode_captions_for_blue(np.array(greedy_words), self.model.idx_to_word)
                    all_decoded_for_eval.extend(greedy_decoded)

                scores = evaluate_for_particular_captions(all_decoded_for_eval, data_path='./data', split='val',
                                                          get_scores=True)

                f.write("before train:")
                f.write('\n')
                f.write("Bleu_1:" + str(scores['Bleu_1']))
                f.write('\n')
                f.write("Bleu_2:" + str(scores['Bleu_2']))
                f.write('\n')
                f.write("Bleu_3:" + str(scores['Bleu_3']))
                f.write('\n')
                f.write("Bleu_4:" + str(scores['Bleu_4']))
                f.write('\n')
                f.write("ROUGE_L:" + str(scores['ROUGE_L']))
                f.write('\n')
                f.write("metric" + str(1*scores['Bleu_4'] + 1*scores['Bleu_3'] + 0.5*scores['Bleu_1'] + 0.5*scores['Bleu_2']))
                f.write('\n')

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = np.array(captions[rand_idxs])
                image_idxs = np.array(image_idxs[rand_idxs])
                b_for_eval = []
                greedy_decoded_for_eval = []
                for i in range(n_iters_per_epoch):
                    captions_batch = np.array(captions[i * self.batch_size:(i + 1) * self.batch_size])
                    image_idxs_batch = np.array(image_idxs[i * self.batch_size:(i + 1) * self.batch_size])
                    features_batch = np.array(features[image_idxs_batch])

                    ground_truths = [captions[image_idxs == image_idxs_batch[j]] for j in
                                     range(len(image_idxs_batch))]

                    ref_decoded = [decode_captions(ground_truths[j], self.model.idx_to_word) for j in range(len(ground_truths))]

                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}



                    # fetch captions as cache
                    samples, greedy_words = sess.run([sampled_captions, greedy_caption],
                                                             feed_dict)
                    mask, all_decoded = decode_captions_for_blue(samples, self.model.idx_to_word)
                    _, greedy_decoded = decode_captions_for_blue(greedy_words, self.model.idx_to_word)
                    greedy_decoded_for_eval.extend(greedy_decoded)

                    r = [evaluate_captions([k], [v])  for k, v in zip(ref_decoded, all_decoded)]
                    b = [evaluate_captions([k], [v]) for k, v in zip(ref_decoded, greedy_decoded)]

                    b_for_eval.extend(b)

                    feed_dict = { grad_mask: mask, self.model.sample_caption:samples ,rewards: r, base_line: b,
                                 self.model.features: features_batch, self.model.captions: captions_batch
                                 }  # write summary for tensorboard visualization
                    _ = sess.run([train_op], feed_dict)

                # print out BLEU scores and file write
                if self.print_bleu:

                    with open(os.path.join(self.model_path, 'val.RandB.scores.txt'), 'a') as f:
                        all_decoded_for_eval = []
                        for k in range(n_iters_val):
                            captions_batch = np.array(val_captions[k * self.batch_size:(k + 1) * self.batch_size])
                            # image_idxs_batch = np.arange(k * self.batch_size, (k + 1) * self.batch_size)
                            # image_idxs_batch = np.array(image_idxs[k * self.batch_size:(k+ 1) * self.batch_size])
                            # features_batch = np.array(features[image_idxs_batch])
                            features_batch = val_features[k * self.batch_size:(k + 1) * self.batch_size]
                            feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}

                            greedy_words = sess.run(greedy_caption,
                                                             feed_dict)
                            _, greedy_decoded = decode_captions_for_blue(np.array(greedy_words), self.model.idx_to_word)
                            all_decoded_for_eval.extend(greedy_decoded)



                        scores = evaluate_for_particular_captions(all_decoded_for_eval,data_path='./data', split='val', get_scores=True)

                        s = [i > j for i, j in zip(r, b)]
                        f.write('Epoch %d\n' % (e + 1))
                        f.write("b:" + str(np.mean(np.array(b_for_eval))))
                        f.write('\n')
                        f.write("count true:" + str(s.count(True)))
                        f.write('\n')
                        f.write(str(ref_decoded[0]))
                        f.write('\n')
                        f.write(str(all_decoded[0]))
                        f.write('\n')
                        f.write(str(greedy_decoded[0]))
                        f.write('\n')
                        f.write("Bleu_1:" + str(scores['Bleu_1']))
                        f.write('\n')
                        f.write("Bleu_2:" + str(scores['Bleu_2']))
                        f.write('\n')
                        f.write("Bleu_3:" + str(scores['Bleu_3']))
                        f.write('\n')
                        f.write("Bleu_4:" + str(scores['Bleu_4']))
                        f.write('\n')
                        f.write("ROUGE_L:" + str(scores['ROUGE_L']))
                        f.write('\n')
                        f.write("metric" + str(
                            1 * scores['Bleu_4'] + 1 * scores['Bleu_3'] + 0.5 * scores['Bleu_1'] + 0.5 * scores['Bleu_2']))
                        f.write('\n')
                        if (e + 1) % self.save_every == 0:
                            saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                            print "model-%s saved." % (e + 1)

    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=False):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if self.print_bleu:
                all_gen_cap = np.ndarray((features.shape[0], 20))
                for i in range(n_iters_per_epoch):
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    gen_cap = sess.run(sampled_captions, feed_dict=feed_dict)
                    all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                scores = evaluate(data_path='./data', split='val', get_scores=True)


            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" % decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.clf()
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t + 2)
                        plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n, t]), color='black', backgroundcolor='white',
                                 fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n, t, :].reshape(14, 14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.savefig(str(n) + 'test.pdf')

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))

    def inference(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/lstm/model-20')
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            print "end"
            print decoded