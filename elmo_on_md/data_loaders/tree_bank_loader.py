import os

from elmo_on_md.data_loaders.loader import Loader


class Token_loader(Loader):
    def load_data(self) -> dict:
        """
        load the plain text, devided into tokens
        Returns: A dictonary with 3 entries: ['train', 'dev', 'test']
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
            corpus = [[word.strip() for word in sentence.split('\n')] for sentence in splat]

            return corpus

