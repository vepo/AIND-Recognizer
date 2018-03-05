import warnings
from asl_data import SinglesData
import numpy as np
import pandas as pd
from asl_data import AslDb
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV
from asl_utils import show_errors
#import arpa

#lm_models = arpa.loadf("asl.lm")

def score_lm(word, last_words):
    try:
        return lm_models[0].log_s(' '.join(last_words + [word]))
    except:
        return float("-inf")#lm_models[0].log_s(' '.join(last_words + ['<unk>']))

def end_of_sentence(sentences_index, index):
    for _, index_list in sentences_index.items():
        if index in index_list:
            return index_list.index(index) + 1 == len(index_list)
    return False

# https://cmusphinx.github.io/wiki/arpaformat/
def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    recog_sentence = ['<s>']
    for index, (X, lengths) in test_set.get_all_Xlengths().items():
        best_score, best_guess, word_probability = float("-inf"), "", {}

        for model_word, model in models.items():
            prob = float("-inf")
            try:
                prob = model.score(X, lengths)# + score_lm(model_word, recog_sentence[-2:])
            except:
                pass
            word_probability[model_word] = prob

            if prob > best_score:
                best_score, best_guess = prob, model_word

        #if test_set.wordlist[index] != best_guess:
        #    print('WRONG WORD: {} != {} {}'.format(best_guess, test_set.wordlist[index], best_score))
        #    print(word_probability)

        guesses.append(best_guess)
        recog_sentence.append(best_guess)
        probabilities.append(word_probability)
        if end_of_sentence(test_set.sentences_index, index):
            recog_sentence = ['<s>']


    return probabilities, guesses

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

def WER(guesses, test_set):
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return float(S) / float(N)

def CORRECT(guesses, test_set):
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return N - S

if __name__ == "__main__":
    print('Initialize features...')
    asl = AslDb() # initializes the database
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
    features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
    df_means = asl.df.groupby('speaker').mean()
    asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
    df_std = asl.df.groupby('speaker').std()
    features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
    asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x'])) / asl.df['speaker'].map(df_std['right-x'])
    asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y'])) / asl.df['speaker'].map(df_std['right-y'])
    asl.df['norm-lx'] = (asl.df['left-x']  - asl.df['speaker'].map(df_means['left-x']))  / asl.df['speaker'].map(df_std['left-x'])
    asl.df['norm-ly'] = (asl.df['left-y']  - asl.df['speaker'].map(df_means['left-y']))  / asl.df['speaker'].map(df_std['left-y'])

    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

    grnd_rx, grnd_ry = asl.df['grnd-rx'], asl.df['grnd-ry']
    grnd_lx, grnd_ly = asl.df['grnd-lx'], asl.df['grnd-ly']

    asl.df['polar-rr'] = np.hypot(grnd_rx, grnd_ry)
    asl.df['polar-lr'] = np.hypot(grnd_lx, grnd_ly)
    asl.df['polar-rtheta'] = np.arctan2(grnd_rx, grnd_ry)
    asl.df['polar-ltheta'] = np.arctan2(grnd_lx, grnd_ly)

    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

    asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
    asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
    asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
    asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)

    asl.df['delta-nx'] = asl.df['nose-x'].diff().fillna(0)
    asl.df['delta-ny'] = asl.df['nose-y'].diff().fillna(0)
    asl.df['dist-norm-x'] = asl.df['norm-lx'] - asl.df['norm-rx']
    asl.df['dist-norm-y'] = asl.df['norm-ly'] - asl.df['norm-ry']

    feature_delta_nose = ['delta-nx', 'delta-ny']
    feature_dist_norm = ['dist-norm-x', 'dist-norm-y']

    features_scaled = ['scaled-rx', 'scaled-ry', 'scaled-lx', 'scaled-ly', 'scaled-grx', 'scaled-gry', 'scaled-glx', 'scaled-gly']
    for scaled, orign in zip(features_scaled, ['right-x', 'right-y', 'left-x', 'left-y', 'grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']):
        asl.df[scaled] = (asl.df[orign] - asl.df[orign].min()) / (asl.df[orign].max() - asl.df[orign].min())

    # TODO define a list named 'features_custom' for building the training set
    features_custom = feature_delta_nose + feature_dist_norm + features_scaled
    asl.df[features_custom]

    features = {
        'features_ground': features_ground,
        'features_polar': features_polar,
        'features_delta': features_delta,
        'features_norm': features_norm,
        'ALL': features_ground + features_polar + features_delta + features_norm,
        'features_custom': features_custom,
        'ALL_with_custom': features_ground + features_polar + features_delta + features_norm + features_custom
        }
    selectors = {
        'SelectorBIC': SelectorBIC,
        'SelectorDIC': SelectorDIC,
        'SelectorCV': SelectorCV
    }
    for set_name, set_value in features.items():
        for sel_name, sel in selectors.items():
            models = train_all_words(set_value, sel)
            test_set = asl.build_test(set_value)
            probabilities, guesses = recognize(models, test_set)
            print('|{}|{}|{}|{}|'.format(set_name, sel_name, CORRECT(guesses, test_set), WER(guesses, test_set)))