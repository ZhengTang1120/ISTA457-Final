from str2vec import Index
import numpy as np
import pickle
import os.path

def save_indices(word_index, char_index, hashtag_index, file_name):
    with open(file_name, "wb") as f:
        pickle.dump((word_index, char_index, hashtag_index), f)

def load_indices(file_name):
    if os.path.isfile(file_name):
        content = pickle.load(open(file_name, "rb"))
        return content[0], content[1], content[2]
    else:
        print ("Model Not Found")

def read_tweets(filename):
    words = set()
    chars = set()
    hashtags = set()
    tweets = list()
    with open(filename) as f:
        next(f)
        for line in f:
            tweet = Tweet.from_line(line)
            for word in tweet.content.split():
                if word[0] not in ["#", "@"]:
                    words.add(word)
                    for char in word:
                        chars.add(char)
                elif word[0] == "#":
                    hashtags.add(word)
            tweets.append(tweet)
    words = list(words)
    words.sort()
    chars = list(chars)
    chars.sort()
    hashtags = list(hashtags)
    hashtags.sort()
    return tweets, words, chars, hashtags 

def make_vectors_train(tweets, words, chars, hashtags):
    word_index = Index(words)
    char_index = Index(chars)
    hashtag_index = Index(hashtags)
    for tweet in tweets:
        content = [w for w in tweet.content.split() if w[0] not in ["#", "@"]]
        hashs = [w for w in tweet.content.split() if w[0] == "#"]
        tweet.cont_vec = word_index.objects_to_indexes(content)
        tweet.hashtags_vec = hashtag_index.objects_to_indexes(hashs)
        for word in content:
            tweet.cont_char_vec.append(char_index.objects_to_indexes(word))
    return word_index, char_index, hashtag_index

def make_vectors_test(tweets, word_index, char_index, hashtag_index):
    for tweet in tweets:
        content = [w for w in tweet.content.split() if w[0] not in ["#", "@"]]
        hashs = [w for w in tweet.content.split() if w[0] == "#"]
        tweet.cont_vec = word_index.objects_to_indexes(content)
        tweet.hashtags_vec = hashtag_index.objects_to_indexes(hashs)
        for word in content:
            tweet.cont_char_vec.append(char_index.objects_to_indexes(word))

class Tweet:
    def __init__(self, id, content, anger, anticipation, disgust, fear,
                joy, love, optimism, pessimism, sadness, surprise, trust):
        self.id = id
        self.content = content
        self.cont_vec = None
        self.cont_char_vec = list()
        self.hashtags_vec = None
        self.emotions = np.array([int(anger), int(anticipation), int(disgust),
                                int(fear), int(joy), int(love), int(optimism),
                                int(pessimism), int(sadness), int(surprise),
                                int(trust)])

    def __str__(self):
        return self.id + '\t' + self.content + '\t' +'\t'.join(list(map(str, self.emotions)))

    @staticmethod
    def from_line(line):
        [id, content, anger, anticipation, disgust, fear, 
        joy, love, optimism, pessimism, sadness, surprise, trust] = line.strip().split("\t")

        return Tweet(id, content, anger, anticipation, disgust, fear, 
                    joy, love, optimism, pessimism, sadness, surprise, trust)
