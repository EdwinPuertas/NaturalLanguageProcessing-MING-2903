import datetime
import pickle
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from logic.data_transformation import DataTransformation
from logic.classifiers import Classifiers
from logic.feature_extraction import FeatureExtraction
from logic.text_processing import TextProcessing
from root import DIR_MODELS, DIR_INPUT


class TrainingModels(object):

    def __init__(self, lang: str = 'es', iteration: int = 10, fold: int = 10):
        self.lang = lang
        self.iteration = iteration
        self.fold = fold
        self.classifiers = Classifiers.dict_classifiers
        self.fe = FeatureExtraction(lang=lang)
        self.tp = TextProcessing(lang=lang)

    def run(self):
        try:
            data = pd.read_csv(DIR_INPUT + 'TASS2018.csv', sep=';')
            date_file = datetime.datetime.now().strftime("%Y-%m-%d")
            print('***Clean data training')

            tuits = [self.tp.transformer(row) for row in tqdm(data['content'].tolist())]
            y = [row for row in data['polarity'].tolist()]

            print('***Get training features')
            x = self.fe.get_features(tuits)

            cv = StratifiedShuffleSplit(n_splits=self.fold, test_size=0.30, random_state=42)

            best = 0.0
            best_clf = None
            name_best = None
            for clf_name, clf_ in self.classifiers.items():
                classifier_name = clf_name
                clf = clf_
                start_time = time.time()
                print('**Training {0} ...'.format(classifier_name))
                scores_acc = []
                scores_recall = []
                scores_f1 = []
                for i in range(1, self.iteration + 1):
                    clf.fit(x, y)
                    accuracy = cross_val_score(clf, x, y, cv=cv, scoring='accuracy')
                    scores_acc.append(accuracy)

                # Calculated Time processing
                t_sec = round(time.time() - start_time)
                (t_min, t_sec) = divmod(t_sec, 60)
                (t_hour, t_min) = divmod(t_min, 60)
                time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)

                mean_score_acc = np.mean(scores_acc)
                std_score_acc = np.std(scores_acc)

                # Calculated statistical
                print('-' * 40)
                print("Results for {} classifier".format(classifier_name))
                print('Mean Accuracy: {0:.3f} (+/- {1: .3f})'.format(mean_score_acc, std_score_acc))
                print("Time processing: {0}".format(time_processing))
                print('-' * 40)
                if mean_score_acc > best:
                    best = mean_score_acc
                    best_clf = clf
                    name_best = classifier_name
            file_model = '{0}tass_model_{1}.pkl'.format(DIR_MODELS, self.lang)
            with open(file_model, 'wb') as file:
                pickle.dump(best_clf, file)
                print('Best classifier is {0} with Accuracy: {1}'.format(name_best, best))
        except Exception as e:
            print('Error baseline: {0}'.format(e))


if __name__ == "__main__":
    tm = TrainingModels(lang='es', iteration=10, fold=10)
    tm.run()
