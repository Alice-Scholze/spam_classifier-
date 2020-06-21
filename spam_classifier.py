from collections import Counter, defaultdict
from machine_learning import split_data
import math, random, re, glob
import numpy as np
import pandas as pd

MIN_TIMES = 50

def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)                           # remove duplicates

def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return { word: [spam, not_spam] for word, (spam, not_spam) in counts.items() if spam > MIN_TIMES and not_spam > MIN_TIMES}

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
             for w, (spam, non_spam) in counts.items()]

def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)

    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # count spam and non-spam messages
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def get_subject_data():
    cleaned_emails = pd.read_csv("cleaned_emails_without_from.csv")
    cleaned_emails.dropna(inplace=True)
    data = [tuple(x) for x in cleaned_emails.to_numpy()]

    return data

def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model():
    data = get_subject_data()
    random.seed(0)      # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > 0.5) # (actual, predicted)
                     for _, is_spam, spam_probability in classified)

    print(counts)

    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:5]

    # print("spammiest_hams", spammiest_hams)
    # print("hammiest_spams", hammiest_spams)

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-5:]
    hammiest_words = words[:5]

    # print("spammiest_words", spammiest_words)
    # print("hammiest_words", hammiest_words)

if __name__ == "__main__":
    MIN_TIMES = 12
    train_and_test_model()

# In this code, I changed the `count_words` to return just words with a minumun count.
# The emails were read in `naive_bayes.ypynb`, where I apply feature engineering and detailed the step by step

# Below, I just list the results obtained for each csv file, using the same quantities for testing and minimum word count

# 25/75 | more 50 times |  Without email from Counter({(False, False): 614, (True, True): 100, (False, True): 79, (True, False): 34}
# 25/75 | more 50 times |  With email from    Counter({(False, False): 622, (True, True): 93, (False, True): 74, (True, False): 41})
# 20/80 | more 50 times |  Without email from Counter({(False, False): 491, (True, True): 80, (False, True): 62, (True, False): 31})
# 20/80 | more 50 times |  With email from    Counter({(False, False): 495, (True, True): 80, (False, True): 61, (True, False): 31})
# 20/75 | more 100 times | Without email from Counter({(False, False): 581, (False, True): 112, (True, True): 92, (True, False): 42})
# 20/75 | more 100 times | With email from    Counter({(False, False): 587, (False, True): 109, (True, True): 91, (True, False): 43})
# 20/75 | more 75 times |  Without email from Counter({(False, False): 614, (True, True): 96, (False, True): 79, (True, False): 38})
# 20/75 | more 75 times |  With email from    Counter({(False, False): 624, (True, True): 97, (False, True): 72, (True, False): 37})
# 20/75 | more 25 times |  Without email from Counter({(False, False): 623, (True, True): 98, (False, True): 70, (True, False): 36})
# 20/75 | more 25 times |  With email from    Counter({(False, False): 629, (True, True): 97, (False, True): 67, (True, False): 37})
# 20/75 | more 15 times |  Without email from Counter({(False, False): 632, (True, True): 101, (False, True): 61, (True, False): 33})
# 20/75 | more 15 times |  With email from    Counter({(False, False): 635, (True, True): 99, (False, True): 61, (True, False): 35})
# 20/75 | more 13 times |  Without email from Counter({(False, False): 633, (True, True): 101, (False, True): 60, (True, False): 33})
# 20/75 | more 13 times |  With email from    Counter({(False, False): 636, (True, True): 99, (False, True): 60, (True, False): 35})