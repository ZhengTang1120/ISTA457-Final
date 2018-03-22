import utils

if __name__ == '__main__':
    tweets_test, w, c = utils.read_tweets("2018-E-c-En-dev.txt")
    model, word_index, char_index = utils.load_model("model")
    utils.make_vectors_test(tweets_test, word_index, char_index)
    for t in tweets_test:
        print(model(t))