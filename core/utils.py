import numpy as np
import cPickle as pickle
import hickle
import time
import os


def load_coco_data(data_path='./data', split='train'):

    start_t = time.time()
    data = {}
    # use validation data to debug
    if split == "debug":
        split = 'val'
        with open(os.path.join(os.path.join(data_path, 'train'), 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)
    data_path = os.path.join(data_path, split)
    data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' % split))
    with open(os.path.join(data_path, '%s.file.names.pkl' % split), 'rb') as f:
        data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.captions.pkl' % split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' % split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)

    if split == 'train':
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" % (end_t - start_t)
    return data


def load_inference_data(data_path='./data'):
    start_t = time.time()
    data = {}
    data['features'] = hickle.load(os.path.join(data_path, 'inference.features.hkl'))
    with open(os.path.join(data_path, 'inference.file.names.pkl'), 'rb') as f:
        data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, 'inference.image.idxs.pkl'), 'rb') as f:
        data['image_idxs'] = pickle.load(f)

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print
            k, type(v), v.shape, v.dtype
        else:
            print
            k, type(v), len(v)
    end_t = time.time()
    print
    "Elapse time: %.2f" % (end_t - start_t)
    return data


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def decode_captions_for_blue(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    masks = []
    for i in range(N):
        words = []
        mask = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                mask.append(1)
                break
            if word != '<NULL>':
                words.append(word)
                mask.append(1)
        decoded.append(' '.join(words))
        mask.extend([0]*(T-len(mask)))
        masks.append(mask)
    return masks, decoded

def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names


def sample_coco_minibatch_inference(data, batch_size):
    data_size = data['features'].shape[0]
    # mask = np.random.choice(data_size, batch_size)
    mask = np.array([0,1,2,3,4,5,6,7,8,9])
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names


def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' % (epoch + 1))
        f.write('Bleu_1: %f\n' % scores['Bleu_1'])
        f.write('Bleu_2: %f\n' % scores['Bleu_2'])
        f.write('Bleu_3: %f\n' % scores['Bleu_3'])
        f.write('Bleu_4: %f\n' % scores['Bleu_4'])
        f.write('METEOR: %f\n' % scores['METEOR'])
        f.write('ROUGE_L: %f\n' % scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' % scores['CIDEr'])


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' % path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)


