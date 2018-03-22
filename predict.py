import utils
import torch
import numpy as np

if __name__ == '__main__':
    tweets_test, w, c = utils.read_tweets("2018-E-c-En-dev.txt")
    model = torch.load("2.model")
    word_index, char_index = utils.load_indices("indices")
    utils.make_vectors_test(tweets_test, word_index, char_index)
    score = 0
    print ("ID  Tweet   anger   anticipation    disgust fear    joy love    optimism    pessimism   sadness surprise    trust")
    for t in tweets_test:
        predictions = ([1 if i > 0 else 0 for i in model(t).data.numpy()])
        t.emotions = predictions
        print (t)

