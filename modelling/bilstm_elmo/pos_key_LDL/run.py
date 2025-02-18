from collections import Counter
import codecs
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data as data_utils
import time
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lines( filename):
    with open(filename, 'r',encoding="utf-8") as fp:
        lines = [line.strip() for line in fp]
    return lines
def read_data(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    bio_lst , bio_lsts = [], []
    freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_tag,pos_tags = [], []
    is_punct, is_puncts = [],[]

    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]
            bio= splitted[2]
            freq = splitted[3]
            e_freq = splitted[4]
            pos = splitted[5]
            punct = float(splitted[6])
            word_id_lst.append(word_id)
            post_lst.append(words)
            bio_lst.append(bio)
            freq_lst.append(freq)
            e_freq_lst.append([float(e_freq),1-float(e_freq)])
            pos_tag.append(pos)
            is_punct.append(punct)
        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            bio_lsts.append(bio_lst)
            freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_tags.append(pos_tag)
            is_puncts.append(is_punct)
            word_id_lst =[]
            post_lst =[]
            bio_lst =[]
            freq_lst =[]
            e_freq_lst =[]
            pos_tag = []
            is_punct = []
    return word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_tags, is_puncts

a,b,c,d,e,f,g = read_data('train_sci.txt')
a1,b1,c1,d1,e1,f1,g1 = read_data('valid_sci.txt')
#a2,b2,c2,d2,e2,f2,g2 = read_data('dev_slide_emph.txt')

def read_data_test(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    #bio_lst , bio_lsts = [], []
    #freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_tag,pos_tags = [], []
    is_punct, is_puncts = [],[]

    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]
            #bio= splitted[2]
            #freq = splitted[3]
            e_freq = splitted[2]
            pos = splitted[3]
            punct = int(splitted[4])
            word_id_lst.append(word_id)
            post_lst.append(words)
            #bio_lst.append(bio)
            #freq_lst.append(freq)
            e_freq_lst.append([float(e_freq),1-float(e_freq)])
            pos_tag.append(pos)
            is_punct.append(punct)
        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            #bio_lsts.append(bio_lst)
            #freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_tags.append(pos_tag)
            is_puncts.append(is_punct)
            word_id_lst =[]
            post_lst =[]
            #bio_lst =[]
            #freq_lst =[]
            e_freq_lst =[]
            pos_tag = []
            is_punct = []
    return word_id_lsts, post_lsts,e_freq_lsts, pos_tags, is_puncts

a2,b2,e2,f2,g2 = read_data_test('test_sci_2.txt')


def create_maps(words, tags, min_word_freq=5, min_char_freq=1):
    """
    Creates maps.
    """
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    for w, t in zip(words, tags):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)

    # add to maps if word freq and char freq are greater than specified min_word_freq and min_char_freq
    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}

    # add special tokens to map
    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map

word_map, char_map, tag_map = create_maps(b+b1+b2 ,f + f1 +f2,1,1)


def create_input_tensors(words, tags, probs, word_map, char_map, tag_map, is_punct):
    """
    Creates input tensors with padding using maps.
    :param words: words in the input
    :param tags: tags in the input
    :param probs: probs in the input
    :param word_map: word map
    :param char_map: character map
    :param tag_map: tag map
    :return: padded input tensors
    """
    # Encode sentences into word maps with <end> at the end
    # [['dunston', 'checks', 'in', '<end>']] -> [[4670, 4670, 185, 4669]]
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [word_map['<end>']], words))

    # Forward and backward character streams
    # [['d', 'u', 'n', 's', 't', 'o', 'n', ' ', 'c', 'h', 'e', 'c', 'k', 's', ' ', 'i', 'n', ' ']]
    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))
    # [['n', 'i', ' ', 's', 'k', 'c', 'e', 'h', 'c', ' ', 'n', 'o', 't', 's', 'n', 'u', 'd', ' ']]
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    # Encode streams into forward and backward character maps with <end> at the end
    # [[29, 2, 12, 8, 7, 14, 12, 3, 6, 18, 1, 6, 21, 8, 3, 17, 12, 3, 60]]
    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    # [[12, 17, 3, 8, 21, 6, 1, 18, 6, 3, 12, 14, 7, 8, 12, 2, 29, 3, 60]]
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))

    # Positions of spaces and <end> character
    # Words are predicted or encoded at these places in the language and tagging models respectively
    # [[7, 14, 17, 18]] are points after '...dunston', '...checks', '...in', '...<end>' respectively
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
    # Reverse the markers for the backward stream before adding <end>, so the words of the f and b markers coincide
    # i.e., [[17, 9, 2, 18]] are points after '...notsnud', '...skcehc', '...ni', '...<end>' respectively
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

    # Encode tags into tag maps with <end> at the end
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
    # Note - the actual tag indices can be recovered with tmaps % len(tag_map)

    # Pad, because need fixed length to be passed around by DataLoaders and other layers
    word_pad_len = max(list(map(lambda s: len(s), wmaps)))
    char_pad_len = max(list(map(lambda s: len(s), cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(lambda s: len(s), tmaps)))
    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    wmap_lengths = []
    cmap_lengths = []
    padded_probs = []
    padded_is_punct = []

    for word, w, cf, cb, cmf, cmb, t,p,punct in zip(words, wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps,probs,is_punct):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t)
        assert len(cmaps_f) == len(cmaps_b)

        # Pad
        # A note -  it doesn't really matter what we pad with, as long as it's a valid index
        # i.e., we'll extract output at those pad points (to extract equal lengths), but never use them

        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))
        #padded_probs.append(p + [0] * (word_pad_len - len(p)))
        for i in range((word_pad_len)-len(p)):
            p.append([float(0),float(1)])
            
        padded_probs.append(p)

        # is_punct is getting padded
        padded_is_punct.append(punct + [2] *(word_pad_len - len(punct)))


        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        #assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
        #    padded_cmarkers_b[-1]) == word_pad_len == len(padded_probs[-1])
        #assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

    #converting to tensor
    padded_wmaps = torch.LongTensor(padded_wmaps)
    padded_cmaps_f = torch.LongTensor(padded_cmaps_f)
    padded_cmaps_b = torch.LongTensor(padded_cmaps_b)
    padded_cmarkers_f = torch.LongTensor(padded_cmarkers_f)
    padded_cmarkers_b = torch.LongTensor(padded_cmarkers_b)
    padded_tmaps = torch.LongTensor(padded_tmaps)
    padded_is_punct = torch.FloatTensor(padded_is_punct)
    wmap_lengths = torch.LongTensor(wmap_lengths)
    cmap_lengths = torch.LongTensor(cmap_lengths)
    padded_probs = torch.FloatTensor(padded_probs)

    return padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct

#training data
padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs, padded_is_punct = create_input_tensors(b,f,e, word_map, char_map, tag_map,g)

t_inputs = data_utils.TensorDataset( padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct)
train_loader = torch.utils.data.DataLoader(t_inputs, batch_size = batch_size, shuffle=True, num_workers=workers, pin_memory=False)

#validation data

padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct = create_input_tensors(b1,f1,e1, word_map, char_map, tag_map,g1)

d_inputs = data_utils.TensorDataset( padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct)
val_loader = torch.utils.data.DataLoader(d_inputs, batch_size = 1, shuffle=False, num_workers=workers, pin_memory=False)

#test data

padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct = create_input_tensors(b2,f2,e2, word_map, char_map, tag_map,g2)

d_inputs = data_utils.TensorDataset( padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, wmap_lengths, cmap_lengths , padded_probs,padded_is_punct)
test_loader = torch.utils.data.DataLoader(d_inputs, batch_size = 1, shuffle=False, num_workers=workers, pin_memory=False)

def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.
    :param input_embedding: embedding tensor
    :return:
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)

def load_embeddings(emb_file, word_map, expand_vocab=True):
    """
    Load pre-trained embeddings for words in the word map.
    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """
    with open(emb_file, 'r') as f:
        emb_len = len(f.readline().split(' ')) - 1

    print("Embedding length is %d." % emb_len)

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = torch.FloatTensor(len(word_map), emb_len)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r',encoding="utf8"):
        line = line.split(' ')
        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
        embeddings = torch.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.size(0) == len(word_map)

    print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

    return embeddings, word_map, lm_vocab_size

embeddings, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,expand_vocab)
charset_size = len(char_map)
model = LM_LSTM_CRF(charset_size, char_emb_dim, char_rnn_dim, char_rnn_layers,
        lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout).to('cuda')
        
model.init_word_embeddings(embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
model.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

#loss_fn = nn.BCELoss().to(device)
loss_fn = nn.KLDivLoss(reduction='batchmean').to(device)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, loss_fn, optimizer, epoch, print_freq = print_frequency):
    
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    f1s = AverageMeter()  # f1 score

    start = time.time()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, probs,puncts) in enumerate(train_loader):
        #print(tmaps)
        #print(puncts)
        data_time.update(time.time() - start)
        max_word_len = max(wmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        probs = probs[:, :max_word_len].to(device)
        puncts = puncts[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        
        words = []
        for i in range(len(wmaps)):
            value_to_key = []
            for j in range(len(wmaps[i])):
                value_to_key.append(list(word_map.keys())[list(word_map.values()).index(wmaps.cpu().data.numpy()[i][j])])
            words.append(value_to_key)
        
        # Forward prop.
        scores, tmaps_sorted, wmaps_sorted, probs_sorted, wmap_lengths_sorted, _ = model(words, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, wmap_lengths, cmap_lengths, probs, tmaps,puncts)

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()
        
        # loss
        probs_sorted.resize_(scores.size())  
        
        # predicted scores and actual targets
        scores = pack_padded_sequence(scores, lm_lengths, batch_first=True).data
        targets = pack_padded_sequence(probs_sorted, lm_lengths, batch_first=True).data
        # calculating loss
        loss = loss_fn(scores,targets)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(lm_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        # Print training status
        if i % 2 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time, loss=losses))

def intersection(lst1, lst2):
    """
    Get intersection of two lists.
    :param lst1: first list
    :param lst2: second list
    :return: list containing intersection
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def fix_padding(scores_numpy, label_probs,  mask_numpy):
    """
    Fixes the padding
    :param scores_numpy: predicted scores
    :param label_probs: actual probs
    :param mask_numpy: mask
    :return: scores and labels with no padding
    """
    all_scores_no_padd = []
    all_labels_no_pad = []
    for i in range(len(mask_numpy)):
        all_scores_no_padd.append(scores_numpy[i][:mask_numpy[i]])
        all_labels_no_pad.append(label_probs[i][:mask_numpy[i]])
    # Sanity check
    assert len(all_scores_no_padd) == len(all_labels_no_pad)
    return all_scores_no_padd, all_labels_no_pad

def match_M(batch_scores_no_padd, batch_labels_no_pad):
    """
    Compute score.
    :param batch_scores_no_padd: predicted scores of the batch without padding
    :param batch_labels_no_padd: actual labels of the batch without padding
    :return: batch_num_m: number of words considered for m=[1,2,3,4] while evaluating score
             batch_score_m: total score of words considered for m=[1,2,3,4]
    """
    top_m = [1,5,10]
    batch_num_m=[]
    batch_score_m=[]
    for m in top_m:
        intersects_lst = []
        score_lst = []
        ############################################### computing scores:
        for s in batch_scores_no_padd:
            if len(s) <=m:
                continue
            h = m

            s = np.asarray(s.cpu())
            ind_score = sorted(range(len(s)), key = lambda sub: s[sub])[-h:]
            score_lst.append(ind_score)

        ############################################### computing labels:
        label_lst = []
        for l in batch_labels_no_pad:
            if len(l) <=m:
                continue
            # if it contains several top values with the same amount
            h = m
            l = l.cpu()
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.asarray(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label)

        ############################################### :

        for i in range(len(score_lst)):
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/(min(m, len(score_lst[i]))))
        batch_num_m.append(len(score_lst))
        batch_score_m.append(sum(intersects_lst))
    return batch_num_m, batch_score_m

import json
for epoch in range(0, epochs):
        # One epoch's training
        train(train_loader, model, loss_fn, optimizer, epoch)
        print("\n")
        #Validation
        model.train(mode=False)
        num_m = [0, 0, 0]
        score_m = [0, 0, 0]
        val_results = []
        with torch.no_grad():
            for i, ( wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, probs, puncts) in enumerate(val_loader):
                    # get maximum word len in the batch
                    max_word_len = max(wmap_lengths.tolist())

                    # Reduce batch's padded length to maximum in-batch sequence
                    # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
                    wmaps = wmaps[:, :max_word_len].to(device)
                    probs = probs[:, :max_word_len].to(device)
                    wmap_lengths = wmap_lengths.to(device)
                    cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
                    cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
                    tmaps = tmaps[:, :max_word_len].to(device)
                    puncts = puncts[:, :max_word_len].to(device)
                    
                    #get actual words from the word mappings
                    words = []
                    for i in range(len(wmaps)):
                        value_to_key = []
                        for j in range(len(wmaps[i])):
                            value_to_key.append(list(word_map.keys())[list(word_map.values()).index(wmaps.cpu().data.numpy()[i][j])])
                        words.append(value_to_key)

                    # Forward prop.
                    scores, tmaps_sorted, wmaps_sorted, probs_sorted, wmap_lengths_sorted, _ = model(words, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, wmap_lengths, cmap_lengths, probs, tmaps,puncts)
                    
                    scores = scores.tolist()
                    probs_sorted = probs_sorted.tolist()
                    for i in range(len(scores)):
                        for j in range(len(scores[i])):
                            scores[i][j].pop(1)
                            probs_sorted[i][j].pop(1)

                    scores = torch.FloatTensor(scores).to(device)
                    probs_sorted = torch.FloatTensor(probs_sorted).to(device)

                    probs_sorted = probs_sorted.reshape(probs_sorted.size(0),probs_sorted.size(1)).to(device)

                    lm_lengths = wmap_lengths_sorted - 1
                    lm_lengths = lm_lengths.tolist()
                    # fix padding and evaluate score
                    batch_scores_no_padd, batch_labels_no_pad = fix_padding(scores, probs_sorted, lm_lengths)
                    batch_num_m, batch_score_m = match_M(batch_scores_no_padd, batch_labels_no_pad)
                    num_m = [sum(i) for i in zip(num_m, batch_num_m)]
                    score_m = [sum(i) for i in zip(score_m, batch_score_m)]

                    for scores in batch_scores_no_padd:
                        val_results.append(scores.tolist())

            m_score = [i/j for i,j in zip(score_m, num_m)]
            score = sum(m_score)/len(m_score)
            print(score)
            if score>max_score:
                max_score = score
                max_m_score = m_score
                print(max_score)

                with open('val_scores.txt', 'w') as f:
                    f.write(json.dumps(val_results))

                final_results = []
                with torch.no_grad():
                    for j, ( wmaps_t, cmaps_f_t, cmaps_b_t, cmarkers_f_t, cmarkers_b_t, tmaps_t, wmap_lengths_t, cmap_lengths_t, probs_t,puncts_t) in enumerate(test_loader):
                       
                        max_word_len = max(wmap_lengths_t.tolist())
                        wmaps = wmaps_t[:, :max_word_len].to(device)
                        probs = probs_t[:, :max_word_len].to(device)
                        wmap_lengths = wmap_lengths_t.to(device)
                        cmarkers_f = cmarkers_f_t[:, :max_word_len].to(device)
                        cmarkers_b = cmarkers_b_t[:, :max_word_len].to(device)
                        tmaps = tmaps_t[:, :max_word_len].to(device)
                        puncts = puncts_t[:, :max_word_len].to(device)
                       
                        words = []
                        for i in range(len(wmaps)):
                            value_to_key = []
                            for j in range(len(wmaps[i])):
                                value_to_key.append(list(word_map.keys())[list(word_map.values()).index(wmaps.cpu().data.numpy()[i][j])])
                            words.append(value_to_key)

                        scores, tmaps_sorted, wmaps_sorted, probs_sorted, wmap_lengths_sorted, _ = model(words, cmaps_f_t, cmaps_b_t, cmarkers_f, cmarkers_b, wmaps, wmap_lengths, cmap_lengths_t, probs, tmaps,puncts)
                        scores = scores.tolist()
                        probs_sorted = probs_sorted.tolist()
                        for i in range(len(scores)):
                            for j in range(len(scores[i])):
                                scores[i][j].pop(1)
                                probs_sorted[i][j].pop(1)
	                            #pass

                        scores = torch.FloatTensor(scores).to(device)
                        probs_sorted = torch.FloatTensor(probs_sorted).to(device)

                        probs_sorted = probs_sorted.reshape(probs_sorted.size(0),probs_sorted.size(1)).to(device)

                        lm_lengths = wmap_lengths_sorted - 1

                        lm_lengths = lm_lengths.tolist()
                        #print(scores[1].shape,scores[1])
                        #print(labels[1].shape,labels[1])
                        batch_scores_no_padd, batch_labels_no_pad = fix_padding(scores, probs_sorted, lm_lengths)
                        #print(batch_scores_no_padd[1].shape,batch_scores_no_padd[1])
                        #print(batch_labels_no_pad[1].shape,batch_labels_no_pad[1])
                        #batch_scores_no_padd = batch_scores_no_padd.tolist()
                        #print(batch_scores_no_padd)
                        for scores in batch_scores_no_padd:
                            #print(len(scores.tolist()))
                            final_results.append(scores.tolist())
                            #print("Done")
                        #batch_num_m, batch_score_m = match_M(batch_scores_no_padd, batch_labels_no_pad)
                        #num_m = [sum(i) for i in zip(num_m, batch_num_m)]
                        #score_m = [sum(i) for i in zip(score_m, batch_score_m)]
                        
                with open('scores.txt', 'w') as f:
                    f.write(json.dumps(final_results))



print(max_score)


