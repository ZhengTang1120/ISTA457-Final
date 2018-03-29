import utils
import lstm_dy

if __name__ == '__main__':
    tweets, words, chars = utils.read_tweets("2018-E-c-En-train.txt")
    word_index, char_index = utils.make_vectors_train(tweets, words, chars)
    utils.save_indices(word_index, char_index, "indices")

    # model = lstm.train(tweets, words, chars, 100)

    for i in range(1):
        model = lstm_dy.Twitter(len(words)+1, len(chars)+1, 300, 300, 100, 100, 2, 100, 11)
        model.train(tweets)
        model.save(str(i))