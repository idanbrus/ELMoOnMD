import os
import pickle
from typing import List

import pandas as pd
from elmo_on_md.data_loaders.loader import Loader
from sklearn.preprocessing import LabelEncoder

class NERLoader(Loader):
    def __init__(self, pickle_path='data/ner/ner.pkl'):
        """
        Create a NERLoader object
        Args:
            pickle_path: path of the pickle containing the ner data
        """
        self.pickle_path = pickle_path
        self.types = ['PERS', 'MISC', 'LOC', 'TIME', 'MONEY', 'DATE', 'PERCENT', 'ORG']
        self.NOT_NAME = 'not_name'

    def load_data(self) -> List[pd.DataFrame]:
        """

        Returns:
            A list of dataframe, each represnting a sentence.
             for each dataframe there are multiple columns representing the different NER classes
        """
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

        # create single column for is named entity or not
        df[self.NOT_NAME] = 1
        df[self.NOT_NAME][df['name_entity'] != 'O'] = 0

        # create one-hot columns for type
        for type in self.types:
            df[type] = df['name_entity'].str.contains(type).astype(int)
        return df

