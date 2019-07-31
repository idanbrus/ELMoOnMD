import os
import numpy as np
import torch

from elmo_on_md.data_loaders.loader import Loader
import conllu
import pandas as pd


class TokenLoader(Loader):
    def load_data(self) -> dict:
        """
        load the plain text, devided into tokens
        Returns: A dictionary with 3 entries: ['train', 'dev', 'test']
        each one return a list of lists with sentences devided into tokens.
        """
        source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        paths = [os.path.join(source_path, f'data\\hebrew_tree_bank\\{subset}_hebtb.tokens') for subset in
                 ['train', 'dev', 'test']]
        corpus = list(map(self._read_tokens, paths))
        corpus_dict = {'train': corpus[0], 'dev': corpus[1], 'test': corpus[2]}
        return corpus_dict

    def _read_tokens(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            splat = content.split('\n\n')
            corpus = [[word.strip() for word in sentence.split('\n')] for sentence in splat if sentence.strip()]

            return corpus


class DependencyTreesLoader(Loader):
    def __init__(self):
        self.max_sentence_length = 82

    def load_data(self) -> dict:
        """
        load the plain text, devided into tokens
        Returns: A dictionary with 3 entries: ['train', 'dev', 'test']
        each one return a list of lists with sentences devided into tokens.
        """
        source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        paths = [os.path.join(source_path, f'data\\hebrew_tree_bank\\{subset}_hebtb-gold.conll') for subset in
                 ['train', 'dev', 'test']]
        corpus = list(map(self._read_tokens, paths))
        corpus_dict = {'train': corpus[0], 'dev': corpus[1], 'test': corpus[2]}
        return corpus_dict

    def _read_tokens(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = conllu.parse(file.read())
            content_df = [pd.DataFrame(sentence) for sentence in content]
            # TODO: change the structure of the data
            return content_df


class MorphemesLoader(Loader):
    def __init__(self):
        self.pos_mapping = dict()
        self.max_pos_id = 0
        self.max_sentence_length = 82  # self measured at the moment
        self.max_morpheme_count = 49  # self measured at the moment

    def load_data(self) -> dict:
        """
        loads all morphemes to a vector-like structure
        Returns: A dictionary with 3 entries: ['train', 'dev', 'test']
        Each entry is an array of vectors, each referring to a single token
        Each token is mapped to a vector of length 51, where each entry corresponds to a single POS tag
        """
        source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        paths = [os.path.join(source_path, f'data\\hebrew_tree_bank\\{subset}_hebtb-gold.lattices') for subset in
                 ['train', 'dev', 'test']]
        corpus = list(map(self._read_morphemes, paths))
        corpus_dict = {'train': corpus[0], 'dev': corpus[1], 'test': corpus[2]}
        return corpus_dict

    def _map_pos(self, pos):
        if pos not in self.pos_mapping:
            self.pos_mapping[pos] = self.max_pos_id
            self.max_pos_id += 1
        return self.pos_mapping[pos]

    def _get_pos_and_token_id(self, morpheme_data):
        values = morpheme_data.split('\t')
        return values[-4], int(values[-1])

    def _set_to_vec(self, set):
        ans = np.zeros(self.max_morpheme_count)
        ans[list(set)] = 1
        return ans

    def _get_sentence_morpheme_map(self, sentence):
        morpheme_data = sentence.split('\n')
        pairs = [self._get_pos_and_token_id(morpheme_datum.strip()) for morpheme_datum in morpheme_data if
                 morpheme_datum.strip()]
        temp = []
        for pair in pairs:
            if len(temp) > 0 and temp[-1][1] == pair[1]:
                temp[-1][0].append(pair[0].strip())
            else:
                temp.append(([pair[0].strip()], pair[1]))
        return [(set([self._map_pos(p) for p in vals])) for (vals, pos) in temp]

    def _get_sentence_vector(self, sentence):
        mapped = self._get_sentence_morpheme_map(sentence)
        answer = np.zeros((self.max_sentence_length, self.max_morpheme_count))
        arr = np.array([self._set_to_vec(s) for s in mapped])
        answer[:arr.shape[0], :arr.shape[1]] = arr
        return answer

    def _read_morphemes(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            splat = content.split('\n\n')
            tensors = [self._get_sentence_vector(sentence.strip()) for sentence in splat if sentence.strip()]
            return torch.FloatTensor(tensors)
