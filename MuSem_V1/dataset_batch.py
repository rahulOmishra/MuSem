'''
CREATE THE DATASET FROM THE CSV FILES
'''
print("Load csv files")

import csv
import json
import pickle

import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import numpy as np

from collections import Counter

data_directory = "../Data"
data_output_directory = "../Dataset_50_300"

train_dataset_file_path = data_output_directory+"/train_"
test_dataset_file_path = data_output_directory+"/test"

train_file_path = data_directory+"/train.csv"

embedding_file_path = data_directory+"/glove.6B.300d.txt"

batch_size = 100
num_epochs = 10
n_class = 2

embedding_dim = 300
max_sen_len = 50

X_train = []
X_train_lenght = []
y_train = []

X_test = []
X_test_lenght = []


with open(train_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X_train.append([row[i] for i in [5, 6]])
        X_train_lenght.append([len(row[i]) for i in [5, 6]])
        y_train.append(row[7])
X_train = X_train[1:]
X_train_lenght = X_train_lenght[1:]
y_train = y_train[1:]

print("data preprocessing")

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def preprocessing(sample):
    words = nltk.word_tokenize(sample)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    return words

def load_embedding(embedding_file_path, wordset, embedding_dim):
    words_dict = dict()
    word_embedding = []
    index = 1
    words_dict['$EOF$'] = 0
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r',encoding="utf-8") as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = np.array([float(s) for s in line[1:]])
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return word_embedding, words_dict

wordset = set()

i = 0

for line in X_train:
    print(i, "/", len(X_train))
    i +=1
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)

'''for line in X_test:
    print(i, "/", len(X_test))
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)'''

word_embedding, words_dict = load_embedding(embedding_file_path, wordset, embedding_dim)

no_word_vector = np.zeros(embedding_dim)

for line in X_train:

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[0]) and line[0][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[0][i]]])
        else :
            sentence.append(no_word_vector)
    line[0] = np.array(sentence)

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[1]) and line[1][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[1][i]]])
        else :
            sentence.append(no_word_vector)
    line[1] = np.array(sentence)

for i in range(len(y_train)):
    if y_train[i] == 'congruent':
        y_train[i] = np.array([1, 0])
    else :
        y_train[i] = np.array([0, 1])


test_percentage = 0.10

train_X = np.array(X_train[int(test_percentage*len(X_train)):])
train_X_lenght = np.array(X_train_lenght[int(test_percentage*len(X_train_lenght)):])
train_y = np.array(y_train[int(test_percentage*len(y_train)):])
test_X = np.array(X_train[:int(test_percentage*len(X_train))])
test_X_lenght = np.array(X_test_lenght[:int(test_percentage*len(X_test_lenght))])
test_y = np.array(y_train[:int(test_percentage*len(y_train))])

#test_dataset = [X_test, X_test_lenght]

'''
Create batches and save data
'''
print('Create batches and save data')

for epoch in range(num_epochs):
    print('epoch :', epoch, '/', num_epochs)
    indices = np.arange(len(train_X))
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_X_lenght = train_X_lenght[indices]
    train_y = train_y[indices]

    nb_batch_per_epoch = int(len(train_X)/batch_size+1)
    
    for batch in range(nb_batch_per_epoch):
        print('batch :', batch, '/', nb_batch_per_epoch)
        # normalized batch :
        idx = batch * batch_size
        x1 = np.array([train_X[idx][0]])
        x2 = np.array([train_X[idx][1]])
        l1 = np.array([train_X_lenght[idx][0]])
        l2 = np.array([train_X_lenght[idx][1]])
        y = np.array([train_y[idx]])

        class_sum = train_y[idx]
        while np.sum(class_sum) < batch_size:
            idx = (idx+1)%len(train_X)
            if class_sum[np.argmax(train_y[idx])] <= batch_size/n_class+1:
                x1 = np.append(x1, np.array([train_X[idx][0]]), axis=0)
                x2 = np.append(x2, np.array([train_X[idx][1]]), axis=0)
                l1 = np.append(l1, np.array([train_X_lenght[idx][0]]), axis=0)
                l2 = np.append(l2, np.array([train_X_lenght[idx][1]]), axis=0)
                y = np.append(y, np.array([train_y[idx]]), axis=0)
                class_sum = np.add(class_sum, train_y[idx])

        dataset = [x1, x2, l1, l2, y]
        with open(train_dataset_file_path+str(epoch)+"_"+str(batch), 'wb') as f:
            pickle.dump(dataset, f)

test_dataset = [test_X, test_X_lenght, test_y]

with open(test_dataset_file_path, 'wb') as f:
    pickle.dump(test_dataset, f, protocol='4')

