import utils
import lstm

if __name__ == '__main__':
    tweets, words, chars = utils.read_tweets("2018-E-c-En-train.txt")
    word_index, char_index = utils.make_vectors_train(tweets, words, chars)
    model = lstm.train(tweets, words, chars)
    tweets_test, w, c = utils.read_tweets("2018-E-c-En-dev.txt")
    utils.make_vectors_test(tweets_test, word_index, char_index)
    for t in tweets_test:
        print(model(t))