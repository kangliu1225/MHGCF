import numpy as np
import random as rd
import scipy.sparse as sp
import time
import collections
import pickle
import os


class Data(object):
    def __init__(self, path, batch_size):
        title_enable = True
        if 'Taobao' in path or 'taobao' in path:
            title_enable = False

        self.path = path

        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        title_file = path + '/i2e_title.txt'
        review_file = path + '/i2e_review.txt'
        visual_file = path + '/i2e_visual.txt'
        all_file = path + '/i2e_all.txt'
        # kg_file = path + '/item2entity_.txt'
        img_feat_file = path + '/item2imgfeat.txt'
        text_feat_file = path + '/itemtitle2vec.txt'

        self.exist_items_in_entity = set()
        self.exist_items_in_title = set()
        self.exist_items_in_review = set()
        self.exist_items_in_visual = set()

        # get number of users and items
        self.n_users, self.n_items, self.n_entity = 0, 0, 0
        
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.exist_title_entity, self.exist_review_entity, self.exist_visual_entity = set(), set(), set()

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.item2title_entity = {}
        with open(title_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0],l[1:] 
                        e_list = [int(e) for e in e_list]
                        self.exist_title_entity |= set(e_list)
                        self.item2title_entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2title_entity[int(i)] = []

        self.item2review_entity = {}
        if os.path.exists(review_file):
            with open(review_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if len(l) != 1 and l[1:] != ['']:
                            i, e_list = l[0], l[1:]
                            e_list = [int(e) for e in e_list]
                            self.exist_review_entity |= set(e_list)
                            self.item2review_entity[int(i)] = e_list
                        else:
                            i = l[0]
                            self.item2review_entity[int(i)] = []
        else:
            for i in range(self.n_items):
                self.item2review_entity[i] = []

        self.item2visual_entity = {}
        with open(visual_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0], l[1:]
                        e_list = [int(e) for e in e_list]
                        self.exist_visual_entity |= set(e_list)
                        self.item2visual_entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2visual_entity[int(i)] = []

        self.exist_entity = set()
        self.item2entity = {}
        with open(all_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) != 1 and l[1:] != ['']:
                        i, e_list = l[0], l[1:]
                        e_list = [int(e) for e in e_list]
                        self.exist_entity |= set(e_list)
                        self.item2entity[int(i)] = e_list
                    else:
                        i = l[0]
                        self.item2entity[int(i)] = []

        self.n_items += 1
        self.n_users += 1

        self.n_entity = len(self.exist_entity)

        self.exist_items = list(range(self.n_items))

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.title_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.review_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.visual_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)
        self.all_R = sp.dok_matrix((self.n_items, self.n_entity), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        self.train_users = {}
        self.train_users_f = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    for i in items:
                        if i not in self.train_users_f:
                            self.train_users_f[i] = []
                        else:
                            self.train_users_f[i].append(uid)

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                        if i not in self.train_users:
                            self.train_users[i] = []
                        self.train_users[i].append(uid)

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        for i, l in self.item2title_entity.items():
            if len(l) > 1:
                self.exist_items_in_title.add(i)
                for e in l:
                    self.title_R[i, e] = 1.
                    self.all_R[i, e] = 1.


        for i, l in self.item2review_entity.items():
            if len(l) > 1:
                self.exist_items_in_review.add(i)
                for e in l:
                    self.review_R[i, e] = 1.
                    self.all_R[i, e] = 1.

                
        for i, l in self.item2visual_entity.items():
            if len(l) > 1:
                self.exist_items_in_visual.add(i)
                for e in l:
                    self.visual_R[i, e] = 1.
                    self.all_R[i, e] = 1.


        self.img_features = {}
        with open(img_feat_file, 'r') as file:
            for line in file.readlines():
                l = line.strip().split(' ')
                item_id = l[0]
                img_feat = list(map(float, l[1:]))
                self.img_features[int(item_id)] = img_feat
            self.imageFeaMatrix = [[0.] * 4096] * self.n_items
            for item in self.img_features:
                self.imageFeaMatrix[item] = self.img_features[item]
        #
        if title_enable:
            self.text_features = {}
            with open(text_feat_file, 'r') as file:
                for line in file.readlines():
                    l = line.strip().split(' ')
                    item_id = l[0]
                    text_feat = list(map(float, l[1:]))
                    self.text_features[int(item_id)] = text_feat
                self.textFeatMatrix = [[0.] * 300] * self.n_items
                for item in self.text_features:
                    self.textFeatMatrix[item] = self.text_features[item]

        self.exist_items_in_entity = self.exist_items_in_title & self.exist_items_in_review & self.exist_items_in_visual
        self.R = self.R.tocsr()

        self.title_R = self.title_R.tocsr()
        self.review_R = self.review_R.tocsr()
        self.visual_R = self.visual_R.tocsr()
        self.all_R = self.all_R.tocsr()



        self.coo_R = self.R.tocoo()
        self.coo_title_R = self.title_R.tocoo()
        self.coo_review_R = self.review_R.tocoo()
        self.coo_visual_R = self.visual_R.tocoo()
        self.coo_all_R = self.all_R.tocoo()

    def get_adj_mat(self):

        left, norm_adj_mat_3, norm_adj_mat_4, norm_adj_mat_5, all_norm_adj_mat_3,  all_norm_adj_mat_4, all_norm_adj_mat_5 = self.create_adj_mat()

        return left, norm_adj_mat_3, norm_adj_mat_4, norm_adj_mat_5, all_norm_adj_mat_3, all_norm_adj_mat_4, all_norm_adj_mat_5

    def create_adj_mat(self):
            t1 = time.time()
            adj_mat = sp.dok_matrix((self.n_users + self.n_items + self.n_entity, self.n_users + self.n_items + self.n_entity), dtype=np.float32)
            adj_mat = adj_mat.tolil()
    
            adj_mat_all = sp.dok_matrix(
                (self.n_users + self.n_items + self.n_entity, self.n_users + self.n_items + self.n_entity),
                dtype=np.float32)
            adj_mat_all = adj_mat_all.tolil()

            adj_mat_senti = sp.dok_matrix(
                (self.n_users + self.n_items + self.n_entity, self.n_users + self.n_items + self.n_entity),
                dtype=np.float32)
            adj_mat_senti = adj_mat_senti.tolil()

            adj_mat_all_senti = sp.dok_matrix(
                (self.n_users + self.n_items + self.n_entity, self.n_users + self.n_items + self.n_entity),
                dtype=np.float32)
            adj_mat_all_senti = adj_mat_all_senti.tolil()
    
            R = self.R.tolil()

            all_R = self.all_R.tolil()
    
            adj_mat[:self.n_users, self.n_users: self.n_users + self.n_items] = R
            adj_mat[self.n_users: self.n_users + self.n_items, :self.n_users] = R.T

            adj_mat_all[:self.n_users, self.n_users: self.n_users + self.n_items] = R
            adj_mat_all[self.n_users: self.n_users + self.n_items, :self.n_users] = R.T

            adj_mat_all[self.n_users: self.n_users + self.n_items, self.n_users + self.n_items:] = all_R
            adj_mat_all[self.n_users + self.n_items:, self.n_users: self.n_users + self.n_items] = all_R.T

            adj_mat = adj_mat.todok()
            adj_mat_all = adj_mat_all.todok()
            adj_mat_senti = adj_mat_senti.tocsr()
            adj_mat_all_senti = adj_mat_all_senti.tocsr()

            print('already create adjacency matrix', adj_mat.shape, time.time() - t1)
    
            t2 = time.time()
    
            def normalized_adj_symetric(adj, d1, d2):
                adj = sp.coo_matrix(adj)
                rowsum = np.array(adj.sum(1))
                d_inv_sqrt = np.power(rowsum, d1).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
                d_inv_sqrt_last = np.power(rowsum, d2).flatten()
                d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
                d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)
    
                return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

            norm_adj_mat_left = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.9, -0.0)
            norm_adj_mat_53 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.3)
            norm_adj_mat_54 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.4)
            norm_adj_mat_55 = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), - 0.5, -0.5)

            norm_adj_mat_all_53 = normalized_adj_symetric(adj_mat_all + sp.eye(adj_mat_all.shape[0]), - 0.5, -0.3)
            norm_adj_mat_all_54 = normalized_adj_symetric(adj_mat_all + sp.eye(adj_mat_all.shape[0]), - 0.5, -0.4)
            norm_adj_mat_all_55 = normalized_adj_symetric(adj_mat_all + sp.eye(adj_mat_all.shape[0]), - 0.5, -0.5)

            norm_adj_mat_left = norm_adj_mat_left.tocsr()
            norm_adj_mat_53 = norm_adj_mat_53.tocsr()
            norm_adj_mat_54 = norm_adj_mat_54.tocsr()
            norm_adj_mat_55 = norm_adj_mat_55.tocsr()

            norm_adj_mat_all_53 = norm_adj_mat_all_53.tocsr()
            norm_adj_mat_all_54 = norm_adj_mat_all_54.tocsr()
            norm_adj_mat_all_55 = norm_adj_mat_all_55.tocsr()

            print('already normalize adjacency matrix', time.time() - t2)
            return norm_adj_mat_left.tocsr(), norm_adj_mat_53.tocsr(), norm_adj_mat_54.tocsr(), norm_adj_mat_55.tocsr(), \
                   norm_adj_mat_all_53.tocsr(),  norm_adj_mat_all_54.tocsr(), norm_adj_mat_all_55.tocsr()

    def sample_u(self):
        total_users = self.exist_users 
        users = rd.sample(total_users, self.batch_size)

        def sample_pos_items_for_u(u):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            return pos_i_id

        def sample_neg_items_for_u(u):
            pos_items = self.train_items[u]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items:
                    return neg_id

        pos_items, neg_items, pos_users_for_pi, neg_users_for_pi = [], [], [], []
        for u in users:
            pos_i = sample_pos_items_for_u(u)
            neg_i = sample_neg_items_for_u(u)

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return users, pos_items, neg_items

    def sample_i_all(self):
        total_items = self.item2entity.keys()
        items = rd.sample(total_items, self.batch_size)

        def sample_pos_e_for_i(i):
            pos_entities = self.item2entity[i]
            n_pos_entities = len(pos_entities)
            pos_id = np.random.randint(low=0, high=n_pos_entities, size=1)[0]
            pos_e_id = pos_entities[pos_id]
            return pos_e_id

        def sample_neg_e_for_i(i):
            pos_entities = self.item2entity[i]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_entity, size=1)[0]
                if neg_id not in pos_entities:
                    return neg_id

        pos_e, neg_e = [], []
        for i in items:
            pos_i = sample_pos_e_for_i(i)
            neg_i = sample_neg_e_for_i(i)

            pos_e.append(pos_i)
            neg_e.append(neg_i)

        return items, pos_e, neg_e

    def sample_i_title(self):
        total_items = self.exist_items_in_title
        items = rd.sample(total_items, self.batch_size)

        def sample_pos_e_for_i(i):
            pos_entities = self.item2title_entity[i]
            n_pos_entities = len(pos_entities)
            pos_id = np.random.randint(low=0, high=n_pos_entities, size=1)[0]
            pos_e_id = pos_entities[pos_id]
            return pos_e_id

        def sample_neg_e_for_i(i):
            pos_entities = self.item2title_entity[i]
            while True:
                neg_id = rd.sample(self.exist_title_entity, 1)[0]
                if neg_id not in pos_entities:
                    return neg_id

        pos_e, neg_e = [], []
        for i in items:
            pos_i = sample_pos_e_for_i(i)
            neg_i = sample_neg_e_for_i(i)

            pos_e.append(pos_i)
            neg_e.append(neg_i)

        return items, pos_e, neg_e
    
    def sample_i_review(self):
        total_items = self.exist_items_in_review
        items = rd.sample(total_items, self.batch_size)

        def sample_pos_e_for_i(i):
            pos_entities = self.item2review_entity[i]
            n_pos_entities = len(pos_entities)
            pos_id = np.random.randint(low=0, high=n_pos_entities, size=1)[0]
            pos_e_id = pos_entities[pos_id]
            return pos_e_id

        def sample_neg_e_for_i(i):
            pos_entities = self.item2review_entity[i]
            while True:
                neg_id = rd.sample(self.exist_review_entity, 1)[0]
                if neg_id not in pos_entities:
                    return neg_id

        pos_e, neg_e = [], []
        for i in items:
            pos_i = sample_pos_e_for_i(i)
            neg_i = sample_neg_e_for_i(i)

            pos_e.append(pos_i)
            neg_e.append(neg_i)

        return items, pos_e, neg_e
    
    def sample_i_visual(self):
        total_items = self.exist_items_in_visual
        items = rd.sample(total_items, self.batch_size)

        def sample_pos_e_for_i(i):
            pos_entities = self.item2visual_entity[i]
            n_pos_entities = len(pos_entities)
            pos_id = np.random.randint(low=0, high=n_pos_entities, size=1)[0]
            pos_e_id = pos_entities[pos_id]
            return pos_e_id

        def sample_neg_e_for_i(i):
            pos_entities = self.item2visual_entity[i]
            while True:
                neg_id = rd.sample(self.exist_visual_entity, 1)[0]
                if neg_id not in pos_entities:
                    return neg_id

        pos_e, neg_e = [], []
        for i in items:
            pos_i = sample_pos_e_for_i(i)
            neg_i = sample_neg_e_for_i(i)

            pos_e.append(pos_i)
            neg_e.append(neg_i)

        return items, pos_e, neg_e

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_entity={}, n_title_entity={}, n_review_entity={}, n_visual_entity={}'.format(self.n_entity, len(self.exist_title_entity), len(self.exist_review_entity), len(self.exist_visual_entity)))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def test_data(self):
        for u, i in self.test_set.items():
            user_batch = [0] * 100
            item_batch = [0] * 100
            test_items = []
            negative_items = []
            while len(negative_items) < 99:
                h = np.random.randint(self.n_items)
                if h in self.train_items[u]:
                    continue
                negative_items.append(h)
            test_items.extend(negative_items)
            test_items.extend(i)

            for k, item in enumerate(test_items):
                user_batch[k] = u
                item_batch[k] = item

            yield user_batch, item_batch
