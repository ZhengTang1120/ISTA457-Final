import utils
import lstm_tr

if __name__ == '__main__':
    tweets, words, chars = utils.read_tweets("2018-E-c-En-train.txt")
    word_index, char_index = utils.make_vectors_train(tweets, words, chars)
    utils.save_indices(word_index, char_index, "indices")

    model = lstm_tr.train(tweets, words, chars, 100)

    # for i in range(30):
    #     model = lstm_dy.Twitter(len(words)+1, len(chars)+1, 300, 50, 100, 10, 2, 50, 11)
    #     model.train(tweets)
    #     model.save(str(i))

    # model = lstm_kr.get_model(len(words)+1, len(chars)+1, 500, 50, 300, 300, 100, 100, 100, 11)