import os
import pickle

import pandas as pd
from elmo_on_md.data_loaders.loader import Loader


class NERLoader(Loader):
    def __init__(self, pickle_path='data/ner/ner.pkl'):
        self.pickle_path = pickle_path

    def load_data(self):
        source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pickle_path = os.path.join(source_path, self.pickle_path)

        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Parsing the NER data. This might take a few minutes")
            ner_path = os.path.join(source_path, 'data', 'ner', 'ner.txt')
            with open(ner_path, 'r', encoding='utf8') as file:
                content = file.read()
                splat = content.split('\n\n')
                corpus = [[word.strip().split(' ') for word in sentence.split('\n')] for sentence in splat if sentence.strip()]
                corpus = filter(lambda x: len(x) > 1, corpus)
                df_list = list(map(self._sentence2df, corpus))

                with open(pickle_path, 'wb') as f:
                    pickle.dump(df_list, f)
                return df_list

    def _sentence2df(self, sentence):
        df = pd.DataFrame(sentence, columns=['word', 'name_entity'])
        df = df[df['word'] != '']
        df['label'] = 1
        df['label'][df['name_entity'] == 'O'] = 0
        return df

