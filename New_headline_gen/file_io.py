from nltk import word_tokenize, sent_tokenize
import io


def load_doc(path):
    data = []
    with io.open(path,encoding='utf-8') as f:
        for line in f:
            sents = sent_tokenize(line)
            doc = [word_tokenize(sent) for sent in sents]
            data.append(doc)
    return data

def load_sent(path):
    data = []
    with io.open(path,encoding='utf-8') as f:
        for line in f:
            data.append(line.split())
    return data

def load_z(path):
    data = []
    with open(path) as f:
        for line in f:
            z_str = line.split()
            data.append([float(x) for x in z_str])

    return data

def load_desc(path,max_desc_len=-1):
    data = []
    with io.open(path,encoding='utf-8') as f:
        for line in f:
            line = line.lower()
            line_str = line.split()
            if len(line_str)>max_desc_len:
                line_str = line_str[:max_desc_len]
            data.append(line_str)
    return data

def load_vec(path):
    x = []
    with io.open(path,encoding='utf-8') as f:
        for line in f:
            p = line.split()
            p = [float(v) for v in p]
            x.append(p)
    return x

def write_doc(docs, sents, path):
    with io.open(path,'w',encoding='utf-8') as f:
        index = 0
        for doc in docs:
            for i in range(len(doc)):
                f.write(u' '.join(sents[index]).encode('utf-8'))
                f.write('\n' if i == len(doc)-1 else ' ')
                index += 1

def write_sent(sents, path):
    with io.open(path, 'w',encoding='utf-8') as f:
        for sent in sents:
            f.write(u' '.join([s for s in sent]) + '\n')

def write_vec(vecs, path):
    with io.open(path, 'w',encoding='utf-8') as f:
        for vec in vecs:
            for i, x in enumerate(vec):
                f.write('%.3f' % x)
                f.write('\n' if i == len(vec)-1 else ' ')
