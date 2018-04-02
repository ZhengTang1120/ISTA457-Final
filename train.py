import utils
import lstm_tr
import numpy as np

if __name__ == '__main__':
    tweets, words, chars, hashtags = utils.read_tweets("2018-E-c-En-train.txt")
    word_index, char_index, hashtag_index = utils.make_vectors_train(tweets, words, chars, hashtags)
    utils.save_indices(word_index, char_index, hashtag_index, "indices")

    model = lstm_tr.train(tweets, words, chars, hashtags, 100)

    # for i in range(30):
    #     model = lstm_dy.Twitter(len(words)+1, len(chars)+1, 300, 50, 100, 10, 2, 50, 11)
    #     model.train(tweets)
    #     model.save(str(i))

    # model = lstm_kr.Twitter(len(words)+1, len(chars)+1, 50, 50, 300, 300, 100, 100, 100, 11)

    # X_train = np.array([t.cont_vec for t in tweets])
    # y_train = np.array([t.emotions for t in tweets])

    # model.train(X_train, y_train, model)