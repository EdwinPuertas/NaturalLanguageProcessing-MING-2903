import re
import nltk
from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from logic.text_processing import TextProcessing
from logic.lexical_features import lexical_es, lexical_en
from nltk.corpus import stopwords


# nltk.download('stopwords')


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_processing=None):
        try:
            if text_processing is None:
                self.tp = TextProcessing(lang=lang)
            else:
                self.tp = text_processing
            self.lexical = lexical_es if lang == 'es' else lexical_en
        except Exception as e:
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, messages: List[str]):
        try:
            result = self.get_features(messages)
            return result
        except Exception as e:
            print('Error transform: {0}'.format(e))

    def get_features(self, messages: List[str]):
        try:
            # lexical_features = self.get_features_lexical(messages)
            pos_features = self.get_pos_features(messages)
            # bow_features = self.get_bow(messages)
            # features = np.append(lexical_features, bow_features, axis=1)
            features = np.array(pos_features).reshape(-1, 1)
            return features
        except Exception as e:
            print('Error get_features: {0}'.format(e))
            return None

    @staticmethod
    def get_bow(messages: List[str]):
        try:
            stop_words = set(stopwords.words('spanish'))
            bow = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=stop_words)
            bow.fit(messages)
            return bow.transform(messages).toarray()
        except Exception as e:
            print('Error get_features: {0}'.format(e))
            return None

    def get_pos_features(self, messages: List[str]):
        index_pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                     'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        feature_vec = np.zeros(len(index_pos), dtype="float32")
        total_freq = 1
        for msg in messages:
            doc = self.tp.analysis_pipe(str(msg))
            pos = [token.pos_ for token in doc]
            for token in pos:
                if token not in index_pos:
                    index = index_pos.index(token)
                    value = feature_vec[index]
                    feature_vec[index] = value + 1
                    total_freq += feature_vec[index]
        return feature_vec / total_freq

    def get_features_lexical(self, messages: List[str]):
        try:
            lexical = self.lexical
            tags = ('mention', 'url', 'hashtag', 'emoji', 'rt')
            result = dict()
            i = 0
            for msg in messages:
                vector = dict()
                tokens_text = TextProcessing.tokenizer(msg)
                if len(tokens_text) > 0:
                    vector['weighted_position'], vector['weighted_normalized'] = self.weighted_position(tokens_text)

                    vector['label_mention'] = float(sum(1 for word in tokens_text if word == 'mention'))
                    vector['label_url'] = float(sum(1 for word in tokens_text if word == 'url'))
                    vector['label_hashtag'] = float(sum(1 for word in tokens_text if word == 'hashtag'))
                    vector['label_emoji'] = float(sum(1 for word in tokens_text if word == 'emoji'))
                    vector['label_retweets'] = float(sum(1 for word in tokens_text if word == 'rt'))

                    vector['lexical_diversity'] = self.lexical_diversity(msg)

                    label_word = vector['label_mention'] + vector['label_url'] + vector['label_hashtag']
                    label_word = label_word + vector['label_emoji'] + vector['label_retweets']
                    vector['label_word'] = float(len(tokens_text) - label_word)

                    vector['first_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['first_person_singular']))
                    vector['second_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['second_person_singular']))
                    vector['third_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['third_person_singular']))
                    vector['first_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['first_person_plurar']))
                    vector['second_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['second_person_plurar']))
                    vector['third_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['third_person_plurar']))

                    vector['avg_word'] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                    vector['avg_word'] = vector['avg_word'] if not np.isnan(vector['avg_word']) else 0.0
                    vector['avg_word'] = round(vector['avg_word'], 4)

                    vector['kur_word'] = kurtosis([len(word) for word in tokens_text if word not in tags])
                    vector['kur_word'] = vector['kur_word'] if not np.isnan(vector['kur_word']) else 0.0
                    vector['kur_word'] = round(vector['kur_word'], 4)

                    vector['skew_word'] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                    vector['skew_word'] = vector['skew_word'] if not np.isnan(vector['skew_word']) else 0.0
                    vector['skew_word'] = round(vector['skew_word'], 4)

                    # adverbios
                    vector['adverb_neg'] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                    vector['adverb_neg'] = float(vector['adverb_neg'])

                    vector['adverb_time'] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                    vector['adverb_time'] = float(vector['adverb_time'])

                    vector['adverb_place'] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                    vector['adverb_place'] = float(vector['adverb_place'])

                    vector['adverb_mode'] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                    vector['adverb_mode'] = float(vector['adverb_mode'])

                    vector['adverb_cant'] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                    vector['adverb_cant'] = float(vector['adverb_cant'])

                    vector['adverb_all'] = float(vector['adverb_neg'] + vector['adverb_time'] + vector['adverb_place'])
                    vector['adverb_all'] = float(vector['adverb_all'] + vector['adverb_mode'] + vector['adverb_cant'])

                    vector['adjetives_neg'] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                    vector['adjetives_neg'] = float(vector['adjetives_neg'])

                    vector['adjetives_pos'] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                    vector['adjetives_pos'] = float(vector['adjetives_pos'])

                    vector['who_general'] = sum(1 for word in tokens_text if word in lexical['who_general'])
                    vector['who_general'] = float(vector['who_general'])

                    vector['who_male'] = sum(1 for word in tokens_text if word in lexical['who_male'])
                    vector['who_male'] = float(vector['who_male'])

                    vector['who_female'] = sum(1 for word in tokens_text if word in lexical['who_female'])
                    vector['who_female'] = float(vector['who_female'])

                    vector['hate'] = sum(1 for word in tokens_text if word in lexical['hate'])
                    vector['hate'] = float(vector['hate'])

                result[i] = list(vector.values())
                i += 1
            features = pd.DataFrame.from_dict(result, orient='index').fillna(0.0)
            return features
        except Exception as e:
            print('Error get_lexical_features: {0}'.format(e))

    @staticmethod
    def lexical_diversity(text):
        result = None
        try:
            text_out = re.sub(r"[\U00010000-\U0010ffff]", '', text)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '', text_out)
            text_out = text_out.lower()
            result = round((len(set(text_out)) / len(text_out)), 4)
        except Exception as e:
            print('Error lexical_diversity: {0}'.format(e))
        return result

    @staticmethod
    def weighted_position(tokens_text):
        result = None
        try:
            size = len(tokens_text)
            weighted_words = 0.0
            weighted_normalized = 0.0
            for w in tokens_text:
                weighted_words += 1 / (1 + tokens_text.index(w))
                weighted_normalized += (1 + tokens_text.index(w)) / size
            result = (weighted_words, weighted_normalized)
        except Exception as e:
            print('Error weighted_position: {0}'.format(e))
        return result
