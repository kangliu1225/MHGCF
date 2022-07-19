# encoding:utf-8
import logging
import os
import time
import tensorflow as tf
import os
import sys
import pickle
import math
from load_data import Data
import numpy as np
from sklearn.metrics import roc_auc_score
import math
import multiprocessing
import heapq
import random as rd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'our'
data_path = '../Data/'


batch_size = 2048
dataset = 'Art'
decay = 1e-2
lambda_v = 0.2
lambda_t = 1
lambda_all = 1

lr = 0.001
embed_size = 64
epoch = 1000
data_generator = Data(path='../Data/' + dataset, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = batch_size
Ks = np.arange(1, 21)


def test_one_user(x):
    u, rating = x[1], x[0]

    training_items = data_generator.train_items[u]

    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = rd.sample(list(all_items - set(training_items) - set(user_pos_test)), 99) + user_pos_test

    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    precision, recall, ndcg, hit_ratio = [], [], [], []

    def hit_at_k(r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def ndcg_at_k(r, k):
        r = np.array(r)[:k]

        if np.sum(r) > 0:
            return math.log(2) / math.log(np.where(r == 1)[0] + 2)
        else:
            return 0.

    for K in Ks:
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def test(sess, model, users, items, batch_size, cores):
    result = {'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2

    n_test_users = len(users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size

        end = (u_batch_id + 1) * u_batch_size

        user_batch = users[start: end]

        item_batch = items

        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                    model.pos_items: item_batch})

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result


class Model(object):

    def __init__(self, data_config, img_feat, text_feat):

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entity = data_config['n_entity']

        self.n_fold = 10

        self.norm_adj = data_config['norm_adj']
        self.all_norm_adj = data_config['all_norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = data_config['lr']

        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']

        self.n_layers = data_config['n_layers']

        self.decay = data_config['decay']

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.items_all = tf.placeholder(tf.int32, shape=(None,))
        self.pos_e_all = tf.placeholder(tf.int32, shape=(None,))
        self.neg_e_all = tf.placeholder(tf.int32, shape=(None,))

        # initialization of model parameters
        self.weights = self._init_weights()

        self.im_v = tf.matmul(img_feat, self.weights['w1_v'])
        self.um_v = self.weights['user_embedding_v']

        self.im_t = tf.matmul(text_feat, self.weights['w1_t'])
        self.um_t = self.weights['user_embedding_t']

        self.all_embeddings = self._create_norm_embed()

        self.all_embeddings_all = self._create_norm_embed_all()

        self.ua_embeddings_4, self.ia_embeddings_4, self.ea_embeddings_4 = self.all_embeddings[3]
        self.u_g_embeddings_4 = tf.nn.embedding_lookup(self.ua_embeddings_4, self.users)
        self.pos_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.pos_items)
        self.neg_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.neg_items)

        self.ua_embeddings_v, self.ia_embeddings_v, _ = self._create_norm_embed_v()[0]

        self.u_g_embeddings_v = tf.nn.embedding_lookup(self.ua_embeddings_v, self.users)
        self.pos_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.pos_items)
        self.neg_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.neg_items)

        self.ua_embeddings_t, self.ia_embeddings_t, _ = self._create_norm_embed_t()[0]

        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)
        self.neg_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.neg_items)

        self.ua_embeddings_4_all, self.ia_embeddings_4_all, self.ea_embeddings_4_all = self.all_embeddings_all[1]
        self.u_g_embeddings_4_all = tf.nn.embedding_lookup(self.ua_embeddings_4_all, self.users)
        self.pos_i_g_embeddings_4_all = tf.nn.embedding_lookup(self.ia_embeddings_4_all, self.pos_items)
        self.neg_i_g_embeddings_4_all = tf.nn.embedding_lookup(self.ia_embeddings_4_all, self.neg_items)
        self.i_g_embeddings_4_all = tf.nn.embedding_lookup(self.ia_embeddings_4_all, self.items_all)
        self.pos_e_g_embeddings_4_all = tf.nn.embedding_lookup(self.ea_embeddings_4_all, self.pos_e_all)
        self.neg_e_g_embeddings_4_all = tf.nn.embedding_lookup(self.ea_embeddings_4_all, self.neg_e_all)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        self.u_g_embeddings_v_pre = tf.nn.embedding_lookup(self.um_v, self.users)
        self.pos_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.pos_items)
        self.neg_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.neg_items)

        self.u_g_embeddings_t_pre = tf.nn.embedding_lookup(self.um_t, self.users)
        self.pos_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.pos_items)
        self.neg_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.neg_items)

        self.u_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['user_embedding_all'], self.users)
        self.pos_i_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['item_embedding_all'], self.pos_items)
        self.neg_i_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['item_embedding_all'], self.neg_items)

        self.i_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['item_embedding_all'], self.items_all)
        self.pos_e_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['e_embedding'], self.pos_e_all)
        self.neg_e_g_embeddings_pre_all = tf.nn.embedding_lookup(self.weights['e_embedding'], self.neg_e_all)

        self.batch_ratings = tf.matmul(self.u_g_embeddings_4, self.pos_i_g_embeddings_4,
                                       transpose_a=False, transpose_b=True) + \
                             lambda_all * tf.matmul(self.u_g_embeddings_4_all, self.pos_i_g_embeddings_4_all,
                                                    transpose_a=False, transpose_b=True) + \
                             lambda_v * tf.matmul(self.u_g_embeddings_v, self.pos_i_g_embeddings_v,
                                                  transpose_a=False, transpose_b=True) + \
                             lambda_t * tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t,
                                                  transpose_a=False, transpose_b=True)

        self.loss, self.mf_loss, self.loss_all, self.emb_loss = self.create_bpr_loss()

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['user_embedding_v'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_t')
        all_weights['user_embedding_t'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                      name='user_embedding_t')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['user_embedding_all'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding_all')
        all_weights['item_embedding_all'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding_all')
        all_weights['e_embedding'] = tf.Variable(initializer([self.n_entity, self.emb_dim]),
                                                 name='e_embedding')
        all_weights['w1_v'] = tf.Variable(initializer([4096, self.emb_dim]), name='w1_v')
        all_weights['w1_t'] = tf.Variable(initializer([300, self.emb_dim]), name='w1_t')
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items + self.n_entity) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items + self.n_entity
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_norm_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.weights['user_embedding'], self.weights['item_embedding'], self.weights['e_embedding']], axis=0)

        all_embeddings = {}

        for k in range(0, self.n_layers + 3):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

            u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4 = tf.split(ego_embeddings,
                                                                            [self.n_users, self.n_items, self.n_entity],
                                                                            0)
            all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4]

        return all_embeddings

    def _create_norm_embed_v(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.um_v, self.im_v, self.weights['e_embedding']], axis=0)

        all_embeddings = {}
        # all_embeddings[0] = [self.um_v, self.im_v, self.weights['e_embedding']]

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

            u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4 = tf.split(ego_embeddings,
                                                                            [self.n_users, self.n_items, self.n_entity],
                                                                            0)
            all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4]

        return all_embeddings

    def _create_norm_embed_t(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.um_t, self.im_t, self.weights['e_embedding']], axis=0)

        all_embeddings = {}

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

            u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4 = tf.split(ego_embeddings,
                                                                            [self.n_users, self.n_items, self.n_entity],
                                                                            0)
            all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4]

        return all_embeddings

    def _create_norm_embed_all(self):

        A_fold_hat = self._split_A_hat(self.all_norm_adj)

        ego_embeddings = tf.concat(
            [self.weights['user_embedding_all'], self.weights['item_embedding_all'], self.weights['e_embedding']],
            axis=0)

        all_embeddings = {}

        for k in range(0, self.n_layers):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

            u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4 = tf.split(ego_embeddings,
                                                                            [self.n_users, self.n_items, self.n_entity],
                                                                            0)
            all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4]

        return all_embeddings

    def create_bpr_loss(self):

        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.pos_i_g_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.neg_i_g_embeddings_4), axis=1)

        pos_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.pos_i_g_embeddings_v), axis=1)
        neg_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.neg_i_g_embeddings_v), axis=1)

        pos_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.pos_i_g_embeddings_t), axis=1)
        neg_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.neg_i_g_embeddings_t), axis=1)

        pos_scores_4_all = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4_all, self.pos_i_g_embeddings_4_all),
                                         axis=1)
        neg_scores_4_all = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4_all, self.neg_i_g_embeddings_4_all),
                                         axis=1)

        pos_scores_kg_4_all = tf.reduce_sum(tf.multiply(self.i_g_embeddings_4_all, self.pos_e_g_embeddings_4_all),
                                            axis=1)
        neg_scores_kg_4_all = tf.reduce_sum(tf.multiply(self.i_g_embeddings_4_all, self.neg_e_g_embeddings_4_all),
                                            axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_pre)

        regularizer_all = tf.nn.l2_loss(self.u_g_embeddings_pre_all) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre_all) + \
                          tf.nn.l2_loss(self.neg_i_g_embeddings_pre_all)

        regularizer_m = tf.nn.l2_loss(self.u_g_embeddings_t_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_t_pre) + \
                        tf.nn.l2_loss(self.neg_i_g_embeddings_t_pre) + tf.nn.l2_loss(self.u_g_embeddings_v_pre) + \
                        tf.nn.l2_loss(self.pos_i_g_embeddings_v_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_v_pre)

        regularizer = (regularizer_mf + 0.1 * regularizer_all + 0.1 * regularizer_m) / self.batch_size

        mf_loss_4 = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4))) + tf.reduce_mean(
            tf.nn.softplus(-(pos_scores_t - neg_scores_t))) + tf.reduce_mean(
            tf.nn.softplus(-(pos_scores_v - neg_scores_v)))

        kg_loss_4_all = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4_all - neg_scores_4_all))) + tf.reduce_mean(
            tf.nn.softplus(-(pos_scores_kg_4_all - neg_scores_kg_4_all)))

        emb_loss = self.decay * regularizer

        loss = mf_loss_4 + emb_loss + kg_loss_4_all

        return loss, mf_loss_4, kg_loss_4_all, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    for repeat in range(2):
        for lambda_all in [1]:
            n_layers = 2
            lambda_t = 1
            decay = 1e-3

            cores = multiprocessing.cpu_count() // 2
            Ks = np.arange(1, 21)

            data_generator.print_statistics()
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items
            config['n_entity'] = data_generator.n_entity
            config['decay'] = decay
            config['n_layers'] = n_layers
            config['embed_size'] = embed_size
            config['lr'] = lr
            config['batch_size'] = batch_size

            """
            *********************************************************
            Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
            """
            left, norm_3, norm_4, norm_5, all_norm_3, all_norm_4, all_norm_5, = data_generator.get_adj_mat()

            config['norm_adj'] = norm_4
            config['all_norm_adj'] = all_norm_4

            print('shape of adjacency', norm_4.shape)

            t0 = time.time()

            model = Model(data_config=config, img_feat=data_generator.imageFeaMatrix,
                          text_feat=data_generator.textFeatMatrix)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.

            """
            *********************************************************
            Train.
            """
            loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
            stopping_step = 0
            should_stop = False
            max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
            max_epoch = 0

            best_score = 0
            best_result = {}
            all_result = {}

            for epoch in range(1000):
                t1 = time.time()
                loss, mf_loss, title_loss, review_loss, visual_loss, all_loss, emb_loss, kg_loss = 0., 0., 0., 0., 0., 0., 0., 0.
                n_batch = data_generator.n_train // batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data_generator.sample_u()
                    items_all, pos_e_all, neg_e_all = data_generator.sample_i_all()

                    _, batch_loss, batch_mf_loss, batch_all_loss, batch_emb_loss = sess.run(
                        [model.opt, model.loss, model.mf_loss, model.loss_all, model.emb_loss],
                        feed_dict={model.users: users,
                                   model.pos_items: pos_items,
                                   model.neg_items: neg_items,
                                   model.items_all: items_all,
                                   model.pos_e_all: pos_e_all,
                                   model.neg_e_all: neg_e_all
                                   })
                    loss += batch_loss
                    mf_loss += batch_mf_loss
                    all_loss += batch_all_loss
                    emb_loss += batch_emb_loss

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                if (epoch + 1) % 10 != 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time.time() - t1, loss, mf_loss, all_loss, emb_loss)
                    # print(perf_str)
                    continue

                t2 = time.time()
                users_to_test = list(data_generator.test_set.keys())

                result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
                hr = result['hit_ratio']
                ndcg = result['ndcg']

                score = hr[4] + hr[9] + ndcg[4] + ndcg[9]
                if score > best_score:
                    best_score = score
                    best_result['hr'] = [str(i) for i in hr]
                    best_result['ndcg'] = [str(i) for i in ndcg]

                    stopping_step = 0

                else:
                    stopping_step += 1

                t3 = time.time()

                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],hit@5=[%.5f],,hit@10=[%.5f],hit@20=[%.5f],ndcg@5=[%.5f],ndcg@10=[%.5f],ndcg@20=[%.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, hr[4], hr[9], hr[19], ndcg[4], ndcg[9],
                            ndcg[19])
                print(perf_str)

                all_result[epoch + 1] = result
                if stopping_step == 5:
                    print('best result: hr@5={};ndcg@5={}'.format(best_result['hr'][4], best_result['ndcg'][4]))
                    break

