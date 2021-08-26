import numpy as np
import jsonlines
import nltk
import pandas as pd
import re
import pickle
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from nltk.corpus import stopwords

def readData(instance_path, truth_path):
    train_data = dict()

    with jsonlines.open(instance_path) as reader:
        try:
            for obj in reader:
                tmp = obj['postText'][0]
                tmp1 = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(tmp))
                tmp1 = tmp1.replace('RT','')
                obj['postText'] = tmp1.lower()
                train_data[obj['id']] = obj

                tmp_para = " ".join(obj['targetParagraphs'])
                tmp_para1 = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str(tmp_para))
                obj['targetParagraphs'] = tmp_para1.lower()
                train_data[obj['id']] = obj
        except Exception as e:
            print(e)
            pass

    with jsonlines.open(truth_path) as reader:
        try:
            for obj in reader:
                id = obj['id']
                new_obj = train_data[id]
                new_obj['truthClass']=obj['truthClass']
                train_data[obj['id']] = new_obj
        except:
            pass

    return train_data.values()

def clickbaitchallengeData(type):

    train_data = readData('../original_data/clickbait17-train-170331/instances.jsonl',
                                         '../original_data/clickbait17-train-170331/truth.jsonl')
    validate_data = readData('../original_data/clickbait17-validation-170630/instances.jsonl',
                                         '../original_data/clickbait17-validation-170630/truth.jsonl')

    if type=='train':
        all_data = validate_data # has more data
    elif type=='test':
        all_data = train_data
    else:
        all_data = train_data + validate_data

    heads = np.array([" ".join(nltk.word_tokenize(ad['postText'])) for ad in all_data])
    descs = np.array([" ".join(nltk.word_tokenize(ad['targetParagraphs'])) for ad in all_data])
    truths = np.array([ad['truthClass'] for ad in all_data])

    print('Start writing files...')
    fd_clickbait = open('../data/'+type+'_descs.1','w+')
    fd_nonclickbait = open('../data/'+type+'_descs.0','w+')
    fh_clickbait = open('../data/'+type+'_head.1','w+')
    fh_nonclickbait = open('../data/'+type+'_head.0','w+')

    for i in range(len(truths)):
        t = truths[i]
        h = heads[i]
        d = descs[i]
        len_d = len(d.split(' '))
        len_h = len(h.split(' '))
        if d=='' or h=='' or len_d<20 or len_h<5:
            continue
        if t=='no-clickbait':
            fh_nonclickbait.write(h.lower() + '\n')
            fd_nonclickbait.write(d.lower()+'\n')
        elif t=='clickbait':
            fh_clickbait.write(h.lower() + '\n')
            fd_clickbait.write(d.lower() + '\n')
        else:
            print('Opps, why we have the third label')
    fd_clickbait.close()
    fd_nonclickbait.close()
    fh_clickbait.close()
    fh_nonclickbait.close()

if __name__ == '__main__':
    clickbaitchallengeData('all')