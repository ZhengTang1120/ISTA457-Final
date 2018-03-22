import utils
import lstm

if __name__ == '__main__':
    tweets, words, chars = utils.read_tweets("2018-E-c-En-train.txt")
    word_index, char_index = utils.make_vectors_train(tweets, words, chars)
    model = lstm.train(tweets, words, chars, 10)
    utils.save_indices(word_index, char_index, "indices")