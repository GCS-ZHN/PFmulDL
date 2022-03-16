import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.metrics import auc
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate, TimeDistributed,
    MaxPooling1D, Dropout, RepeatVector, Layer, Reshape, SimpleRNN, LSTM, BatchNormalization, GRU
)
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import pickle
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_score, recall_score, roc_auc_score,f1_score
from collections import deque, Counter
from tqdm import tqdm
import pickle as pkl

AAINDEX = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
MAXLEN = 2000


def to_onehot(seq, start=0):
    onehot = np.zeros((MAXLEN, 21), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot


class Ontology(object):
    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set


class DFGenerator(Sequence):
    def __init__(self, df, terms_dict, nb_classes, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict

    def __len__(self):
        return np.ceil(len(self.df) / float(self.batch_size)).astype(np.int32)

    def __getitem__(self, idx):
        batch_index = np.arange(idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))
        df = self.df.iloc[batch_index]
        data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        self.start += self.batch_size
        return (data_onehot, labels)

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.int32)
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences
                onehot = to_onehot(seq)
                data_onehot[i, :, :] = onehot
                for t_id in row.prop_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()


def load_weight(model_path1, model_path2):
    model = load_model(model_path1)
    loaded_model = load_model(model_path2)
    old_weights = loaded_model.get_weights()
    now_weights = model.get_weights()
    cnt = 0
    for i in range(len(now_weights)):
        if old_weights[cnt].shape == now_weights[i].shape:
            now_weights[i] = old_weights[cnt]
            cnt = cnt + 1
    print(f'{cnt} layers weights copied, total {len(now_weights)}')
    model.set_weights(now_weights)
    model.save(model_path1)


def plot_curve(history):
    plt.figure()
    x_range = range(0, len(history.history['loss']))
    plt.plot(x_range, history.history['loss'], 'bo', label='Training loss')
    plt.plot(x_range, history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def init_evaluate(data_size, batch_size, model_file, data_path, term_path):
    with open(term_path, 'rb') as file:
        terms_df = pd.read_pickle(file)
    with open(data_path, 'rb') as file:
        data_df = pd.read_pickle(file)
    if len(data_df) > data_size:
        data_df = data_df.sample(n=data_size)
    # data_df = data_df.loc[data_df["cafa_target"] == "True"]
    # print(data_df)
    model = load_model(f'model/{model_file}.h5')
    data_file = data_path.split('/')[-1].split('.')[0]
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    labels = np.zeros((len(data_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(data_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    print('predict……')
    data_generator = DFGenerator(data_df, terms_dict, nb_classes, batch_size)
    data_steps = int(math.ceil(len(data_df) / batch_size))
    preds = model.predict(data_generator, steps=data_steps)
    return terms, labels, preds, data_file


def fmeasure(real_annots, pred_annots):
    cnt = 0
    precision = 0.0
    recall = 0.0
    p_total = 0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        cnt += 1
        recall += tpn / (1.0 * (tpn + fnn))
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision_x = tpn / (1.0 * (tpn + fpn))
            precision += precision_x
    recall /= cnt
    if p_total > 0:
        precision /= p_total
    fscore = 0.0
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall


def evaluate_annotations(labels_np, preds_np, terms):
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    labels = list(map(lambda x: set(terms[x == 1]), labels_np))
    for t in range(1, 101):
        threshold = t / 100.0
        preds = preds_np.copy()
        preds[preds >= threshold] = 1
        preds[preds != 1] = 0
        #         fscore, pr, rc = fmeasure(labels, prop_annotations(preds, terms))
        fscore, pr, rc = fmeasure(labels, list(map(lambda x: set(terms[x == 1]), preds)))
        precisions.append(pr)
        recalls.append(rc)
        if fmax < fscore:
            fmax = fscore
            tmax = t
    preds = preds_np.copy()
    preds[preds >= tmax / 100.0] = 1
    preds[preds != 1] = 0
    mcc = matthews_corrcoef(labels_np.flatten(), preds.flatten())
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    return fmax, tmax, recalls, precisions, mcc



def evaluate(model_file, data_path, data_size=8000, batch_size=16, term_path='data/terms.pkl'):
    ont = ['GO:0003674', 'GO:0008150', 'GO:0005575']
    namespace = ['molecular_function', 'biological_process', 'cellular_component', 'all']
    terms, labels, preds, data_file = init_evaluate(data_size, batch_size, model_file, data_path, term_path)


    with open("data/go.pkl", 'rb') as file:
        go = pkl.loads(file.read())
    plt.figure(1, figsize=(16, 3))
    evaluate_info = f'{model_file}, {data_file}:\n'
    for i in range(4):
        print(f'evaluate {namespace[i]}……')
        if i == 3:
            chose = np.ones(len(terms), dtype=bool)
        else:
            go_set = go.get_namespace_terms(namespace[i])
            go_set.remove(ont[i])
            chose = list(map(lambda x: x in go_set, terms))
        _terms = terms[chose]
        _labels = labels[:, chose]
        _preds = preds[:, chose]


        roc_auc = roc_auc_score(_labels.flatten(), _preds.flatten())
        # AUPR = aupr(_labels.flatten(), _preds.flatten())
        fmax, alpha, recalls, precisions, mcc = evaluate_annotations(_labels, _preds, _terms)
        AUPR = auc(recalls, precisions)
        plt.subplot(1, 4, i + 1)
        plt.plot(recalls, precisions, color='darkorange', lw=1, label=f'AUPR={AUPR:0.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'P-R curve of {namespace[i]}')
        plt.legend(loc="lower right")
        evaluate_info += f'\t{namespace[i]}, {len(_terms)}: fmax={fmax:0.3f}, mcc={mcc:0.3f}, AUPR = {AUPR:0.3f}, roc_auc={roc_auc:0.3f}, precision={precisions[alpha]:0.3f}, recall={recalls[alpha]:0.3f}, threshold={alpha}\n'
    plt.savefig('result.png')

    aucli = []
    fs = []
    with open(term_path, 'rb') as file:
        terms_df = pd.read_pickle(file)
    tags = terms_df['tag']
    for i in range(1, 10):
        tag_select = tags == i
        _terms = terms[tag_select]
        _labels = labels[:, tag_select]
        _preds = preds[:, tag_select]
        aucli.append(roc_auc_score(_labels.flatten(), _preds.flatten()))
        res = evaluate_annotations(_labels, _preds, _terms)
        fs.append(res[0])
    plt.figure()
    plt.plot(range(1, len(aucli) + 1), aucli, lw=1, label=f'STD of auc={np.std(aucli):0.5f}')
    plt.savefig("crnn.png")
    #     plt.plot(range(1,len(fs)+1), fs, lw=1, color='orange', label=f'STD of fmax={np.std(fs):0.5f}')
    plt.xlabel('Depth')
    plt.legend(loc="lower right")
    plt.ylim([0.0, 1.0])
    plt.show()
    #     evaluate_info += f'\tauc_std={np.std(aucli):0.5f}, fmax_std={np.std(fs):0.5f}\n'
    evaluate_info += f'\tauc_std={np.std(aucli):0.5f}\n'
    print(aucli)
    print(fs)
    print(evaluate_info)
    # with open("logfile.json", "a") as file:
    #     file.write(evaluate_info)