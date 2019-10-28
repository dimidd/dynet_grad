# -*- coding: utf-8 -*-
import random

import json
import codecs
import dynet as dy
import numpy as np

random.seed(2823274491)

# get the params
HIDDEN_DIM = 40
EMBED_DIM = 20
BILSTM_LAYERS = 2
sLAYERS = '2'


class Vocabulary(object):
    def __init__(self):
        self.all_items = []
        self.c2i = {}

    def add_text(self, paragraph):
        self.all_items.extend(paragraph)

    def finalize(self, fAddBOS=True):
        self.vocab = sorted(list(set(self.all_items)))
        c2i_start = 1 if fAddBOS else 0
        self.c2i = {c: i for i, c in enumerate(self.vocab, c2i_start)}
        self.i2c = self.vocab
        if fAddBOS:
            self.c2i['*BOS*'] = 0
            self.i2c = ['*BOS*'] + self.vocab
        self.all_items = None

    # debug
    def get_c2i(self):
        return self.c2i

    def size(self):
        return len(self.i2c)

    def __getitem__(self, c):
        return self.c2i.get(c, 0)

    def getItem(self, i):
        return self.i2c[i]

class WordEncoder(object):
    def __init__(self, name, dim, model, vocab):
        self.vocab = vocab
        self.enc = model.add_lookup_parameters((vocab.size(), dim))

    def __call__(self, char, DIRECT_LOOKUP=False):
        char_i = char if DIRECT_LOOKUP else self.vocab[char]
        return dy.lookup(self.enc, char_i)


class MLP:
    def __init__(self, model, name, in_dim, hidden_dim, out_dim):
        self.mw = model.add_parameters((hidden_dim, in_dim))
        self.mb = model.add_parameters((hidden_dim))
        self.mw2 = model.add_parameters((out_dim, hidden_dim))
        self.mb2 = model.add_parameters((out_dim))

    def __call__(self, x):
        W = dy.parameter(self.mw)
        b = dy.parameter(self.mb)
        W2 = dy.parameter(self.mw2)
        b2 = dy.parameter(self.mb2)
        mlp_output = W2 * (dy.tanh(W * x + b)) + b2
        return dy.softmax(mlp_output)


class BILSTMTransducer:
    def __init__(self, LSTM_LAYERS, IN_DIM, OUT_DIM, model):
        self.lstmF = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)
        self.lstmB = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)

    def __call__(self, seq):
        """
        seq is a list of vectors (either character embeddings or bilstm outputs)
        """
        fw = self.lstmF.initial_state()
        bw = self.lstmB.initial_state()
        outf = fw.transduce(seq)
        outb = list(reversed(bw.transduce(reversed(seq))))
        return [dy.concatenate([f, b]) for f, b in zip(outf, outb)]


def read_data():
    training_set = json.load(codecs.open('lstm_training.json', "rb", encoding="utf-8"))
    tags = ['aramaic','mishnaic','ambiguous']
    training_set = [{'word':w['word'],'tag':tags.index(w['tag'])} for w in training_set]
    return training_set

def train_epoch(training_dataset):
        last_loss, last_lang_prec = 0.0, 0.0
        total_loss, total_lang_prec = 0.0, 0.0
        total_lang_items = 0

        # shuffle the train data
        random.shuffle(training_dataset)

        items_seen = 0
        # iterate
        for word_obj in training_dataset:
            # calculate the loss & prec
            loss, lang_prec = CalculateLossForWord(word_obj, fValidation=False)

            # forward propagate
            total_loss += loss.value() if loss else 0.0
            # back propagate
            if loss: loss.backward()
            trainer.update()

            # increment the prec variable
            total_lang_prec += lang_prec
            total_lang_items += 1

            items_seen += 1
            # breakpoint?
            breakpoint = 10000
            if items_seen % breakpoint == 0:
                last_loss = total_loss / breakpoint
                last_lang_prec = total_lang_prec / total_lang_items * 100
                total_loss, total_lang_prec = 0.0, 0.0
                total_lang_items = 0

        return last_loss

# When fValidation is true and fRunning is false
# return (1 for true prediction; 0 for false) and a dict with word, predicted_lang, gold_lang and
# confidence
def CalculateLossForWord(word_obj, fValidation=False, fRunning=False):
    dy.renew_cg()

    if not fRunning: gold_lang = word_obj['tag']
    # add a bos before and after
    seq = ['*BOS*'] + list(word_obj['word']) + ['*BOS*']

    # get all the char encodings for the daf
    char_embeds = [let_enc(let) for let in seq]

    # run it through the bilstm
    char_bilstm_outputs = bilstm(char_embeds)
    bilistm_output = dy.concatenate([char_bilstm_outputs[0],char_bilstm_outputs[-1]])

    mlp_input = bilistm_output
    mlp_out = lang_mlp(mlp_input)
    try:
        temp_lang_array = mlp_out.npvalue()
        possible_lang_array = np.zeros(temp_lang_array.shape)
        possible_lang_indices = list(lang_hashtable[word_obj['word']])
        possible_lang_array[possible_lang_indices] = temp_lang_array[possible_lang_indices]
    except KeyError:
        possible_lang_array = mlp_out.npvalue()

    predicted_lang = lang_tags[np.argmax(possible_lang_array)]
    confidence = (mlp_out.npvalue()[:2] / np.sum(mlp_out.npvalue()[:2])).tolist() #skip ambiguous
    # if we aren't doing validation, calculate the loss
    if not fValidation and not fRunning:
        loss = -dy.log(dy.pick(mlp_out, gold_lang))
    # otherwise, set the answer to be the argmax
    elif not fRunning and fValidation:
        loss = None
    else:
        return predicted_lang,confidence

    pos_prec = 1 if predicted_lang == lang_tags[gold_lang] else 0

    tagged_word = { 'word': word_obj['word'], 'tag': predicted_lang, 'confidence':confidence, 'gold_tag':lang_tags[gold_lang]}

    if fValidation:
        return pos_prec, tagged_word

    return loss, pos_prec

# read in all the data
all_data = list(read_data())

random.shuffle(all_data)
lang_hashtable = {}  # make_word_hashtable(train_data)

# create the vocabulary
let_vocab = Vocabulary()
lang_tags = ['aramaic','mishnaic','ambiguous']

# iterate through all the dapim and put everything in the vocabulary
for word in all_data:
    let_vocab.add_text(list(word['word']))

let_vocab.finalize()
# create the model and all it's parameters
model = dy.Model()

# create the word encoders (an encoder for the chars for the bilstm, and an encoder for the prev-pos lstm)
let_enc = WordEncoder("letenc", EMBED_DIM, model, let_vocab)

# the BiLSTM for all the chars, take input of embed dim, and output of the hidden_dim minus the embed_dim because we will concatenate
# with output from a separate bilstm of just the word
bilstm = BILSTMTransducer(BILSTM_LAYERS, EMBED_DIM, HIDDEN_DIM, model)

# now the class mlp, it will take input of 2*HIDDEN_DIM (A concatenate of the before and after the word) + EMBED_DIM from the prev-pos
# output of 2, unknown\talmud
lang_mlp = MLP(model, "classmlp", 2 * HIDDEN_DIM, HIDDEN_DIM, 3)

# the trainer
trainer = dy.AdamTrainer(model)

inf_file = 'inf_grad_last_epoch.model'
model.populate(inf_file)

import unicodecsv as csv
with open('words.csv', 'rb') as inf_csv:
     train_data_inf = [{k: int(v) if k == 'tag' else v for k, v in row.items()}
        for row in csv.DictReader(inf_csv, skipinitialspace=True)]

train_epoch(train_data_inf)
