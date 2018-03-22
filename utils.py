from str2vec import Index
import numpy as np
import pickle
import os.path

def save_indices(word_index, char_index, file_name):
    with open(file_name, "wb") as f:
        pickle.dump((word_index, char_index), f)

def load_indices(file_name):
    if os.path.isfile(file_name):
        content = pickle.load(open(file_name, "rb"))
        print (content)
        return content[0], content[1]
    else:
        print ("Model Not Found")

def read_tweets(filename):
    words = set()
    chars = set()
    tweets = list()
    with open(filename) as f:
        next(f)
        for line in f:
            tweet = Tweet.from_line(line)
            for word in tweet.content.split():
                words.add(word)
                for char in word:
                    chars.add(char)
            tweets.append(tweet)
    return tweets, list(words), list(chars)

def make_vectors_train(tweets, words, chars):
    word_index = Index(words)
    char_index = Index(chars)
    for tweet in tweets:
        content = tweet.content.split()
        tweet.cont_vec = word_index.objects_to_indexes(content)
        for word in content:
            tweet.cont_char_vec.append(char_index.objects_to_indexes(word))
    return word_index, char_index

def make_vectors_test(tweets, word_index, char_index):
    for tweet in tweets:
        content = tweet.content.split()
        tweet.cont_vec = word_index.objects_to_indexes(content)
        for word in content:
            tweet.cont_char_vec.append(char_index.objects_to_indexes(word))

class Tweet:
    def __init__(self, id, content, anger, anticipation, disgust, fear,
                joy, love, optimism, pessimism, sadness, surprise, trust):
        self.id = id
        self.content = content
        self.cont_vec = None
        self.cont_char_vec = list()
        self.emotions = np.array([int(anger), int(anticipation), int(disgust),
                                int(fear), int(joy), int(love), int(optimism),
                                int(pessimism), int(sadness), int(surprise),
                                int(trust)])

    @staticmethod
    def from_line(line):
        [id, content, anger, anticipation, disgust, fear, 
        joy, love, optimism, pessimism, sadness, surprise, trust] = line.strip().split("\t")

        return Tweet(id, content, anger, anticipation, disgust, fear, 
                    joy, love, optimism, pessimism, sadness, surprise, trust)
