import random
import json
import codecs
import dynet as dy
import unicodecsv as csv

random.seed(2823274491)
HIDDEN_DIM = 40
EMBED_DIM = 20
BILSTM_LAYERS = 2

class Vocabulary(object):
    def __init__(self):
        self.all_items = []
        self.c2i = {}

    def add_text(self, paragraph):
        self.all_items.extend(paragraph)

    def finalize(self):
        self.vocab = sorted(list(set(self.all_items)))
        self.c2i = {c: i for i, c in enumerate(self.vocab, 1)}
        self.c2i['*BOS*'] = 0
        self.i2c = ['*BOS*'] + self.vocab
        self.all_items = None

    def size(self):
        return len(self.i2c)

    def __getitem__(self, c):
        return self.c2i.get(c, 0)

class WordEncoder(object):
    def __init__(self, name, dim, model, vocab):
        self.vocab = vocab
        self.enc = model.add_lookup_parameters((vocab.size(), dim))

    def __call__(self, char):
        return dy.lookup(self.enc, self.vocab[char])


class MLP:
    def __init__(self, model, name, in_dim, hidden_dim, out_dim):
        self.mw = model.add_parameters((hidden_dim, in_dim))
        self.mb = model.add_parameters((hidden_dim))
        self.mw2 = model.add_parameters((out_dim, hidden_dim))
        self.mb2 = model.add_parameters((out_dim))

    def __call__(self, x):
        return dy.softmax(self.mw2 * (dy.tanh(self.mw * x + self.mb)) + self.mb2)


class BILSTMTransducer:
    def __init__(self, LSTM_LAYERS, IN_DIM, OUT_DIM, model):
        self.lstmF = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)
        self.lstmB = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)

    def __call__(self, seq):
        fw = self.lstmF.initial_state()
        bw = self.lstmB.initial_state()
        outf = fw.transduce(seq)
        outb = list(reversed(bw.transduce(reversed(seq))))
        return [dy.concatenate([f, b]) for f, b in zip(outf, outb)]

def train_epoch(training_dataset):
        random.shuffle(training_dataset)
        for word_obj in training_dataset:
            loss = CalculateLossForWord(word_obj)
            if loss:
                loss.backward()
            trainer.update()

def CalculateLossForWord(word_obj):
    dy.renew_cg()
    seq = ['*BOS*'] + list(word_obj['word']) + ['*BOS*']
    char_bilstm_outputs = bilstm([let_enc(let) for let in seq])
    bilistm_output = dy.concatenate([char_bilstm_outputs[0],char_bilstm_outputs[-1]])
    return -dy.log(dy.pick(lang_mlp(bilistm_output), word_obj['tag']))

training_set = json.load(codecs.open('lstm_training.json', "rb", encoding="utf-8"))
tags = ['aramaic','mishnaic','ambiguous']
all_data = [{'word':w['word'],'tag':tags.index(w['tag'])} for w in training_set]
random.shuffle(all_data)
let_vocab = Vocabulary()
for word in all_data:
    let_vocab.add_text(list(word['word']))

let_vocab.finalize()
model = dy.Model()
let_enc = WordEncoder("letenc", EMBED_DIM, model, let_vocab)
bilstm = BILSTMTransducer(BILSTM_LAYERS, EMBED_DIM, HIDDEN_DIM, model)
lang_mlp = MLP(model, "classmlp", 2 * HIDDEN_DIM, HIDDEN_DIM, 3)
trainer = dy.AdamTrainer(model)

model.populate('inf_grad_last_epoch.model')
with open('words.csv', 'rb') as inf_csv:
     train_data_inf = [{k: int(v) if k == 'tag' else v for k, v in row.items()}
        for row in csv.DictReader(inf_csv, skipinitialspace=True)]

train_epoch(train_data_inf)
