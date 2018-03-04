import warnings
from asl_data import SinglesData
import arpa

models = arpa.loadf("asl.lm")

def score_lm(word, last_words):
    return models[0].log_s(' '.join(last_words + [word]))

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
    for _, (X, lengths) in test_set.get_all_Xlengths().items():

      best_score, best_guess, word_probability = float("-inf"), "", {}

      for model_word, model in models.items():
        prob = float("-inf")
        try:
          prob = model.score(X, lengths) + score_lm(model_word, guesses)
        except:
            # nothing
            pass
        word_probability[model_word] = prob

        best_score, best_guess = max((prob, model_word), (best_score, best_guess))

      probabilities.append(word_probability)
      guesses.append(best_guess)

    return probabilities, guesses

