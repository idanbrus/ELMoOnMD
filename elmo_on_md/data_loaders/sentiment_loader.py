import os
import pickle

import pandas as pd
from elmo_on_md.data_loaders.loader import Loader


class SentimentLoader(Loader):
    def __init__(self, data_path='data/sentiment'):
        self.data_path = data_path

    def load_data(self):
        """
        loads all data
        Returns: A dictionary with 2 entries: ['train', 'test']
        Each entry is a list of tokens, and a label
        """
        source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        paths = [os.path.join(source_path, f'data\\sentiment\\token_{subset}.tsv') for subset in
                 ['train', 'test']]
        corpus = list(map(self._read_file, paths))
        corpus_dict = {'train': corpus[0], 'test': corpus[1]}
        return corpus_dict

    def _read_file(self,filename):
        file_data = {'sentences':[],'labels':[]}
        with open(filename,'r',encoding='UTF-8') as reader:
            for sentence in reader:
                tokens,label = self._read_sentence(sentence)
                file_data['sentences'].append(tokens)
                file_data['labels'].append(label)
        return file_data
    def _read_sentence(self, sentence):
        tokens,label = sentence.split('\t')
        return pd.Series(tokens.split(' ')),int(label.strip())

