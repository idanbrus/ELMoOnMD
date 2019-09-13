# 1. Use this script to download the hebrew elmo and the hebrew tree bank
# 2. after downloading, unzip the file hebrew.zip file
# 3. change the config.json file, with the configuration file ({rel_path}/ElmoOnMD/ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json)

import urllib.request
from itertools import product
import subprocess
import os
from os import path
from tqdm import tqdm


def download_tree_bank():
    if path.isdir('data/hebrew_tree_bank'):
        print('HebrewTreebank already exists, moving on...')
    else:
        print("Beginning Hebrew Tree Bank download...")
        root_url = 'https://raw.github.com/OnlpLab/HebrewResources/master/HebrewTreebank/hebtb/'
        os.makedirs('data/hebrew_tree_bank')
        for set, data in tqdm(
                list(product(['train', 'dev', 'BiLSTM_pos_weight_8'], ['-gold.conll', '-gold.lattices', '.lattices', '.tokens'])),
                desc='Downloading..', unit='file'):
            file = set + '_hebtb' + data
            url = root_url + file
            urllib.request.urlretrieve(url, f'data/hebrew_tree_bank/{file}')
        print("Finished downloading")


def download_hebrew_elmo():
    if path.isfile('ELMoForManyLangs/hebrew.zip'):
        print('ELMoForManyLangs was already downloaded, moving on...')
    else:
        print("Beginning Hebrew ELMo download...")
        url = 'http://vectors.nlpl.eu/repository/11/154.zip'
        urllib.request.urlretrieve(url, 'ELMoForManyLangs/hebrew.zip')
        print("Finished downloading")

def download_yap():
    if path.isdir('yapproj'):
        print('Yap already cloned, moving on...')
    else:
        os.mkdir('yapproj')
        os.mkdir('yapproj/src')
        print("Beginning to Download yap tool...")
        subprocess.run(["git", "clone", "https://github.com/OnlpLab/yap.git", "yapproj/src/yap"])
        # TODO: extract data
        # TODO: The following:
        # 1. Set environment variable GOPATH=yapproj
        # 2. go to the src directory
        # 3. Do "go get ." and then "go build ."
        # 0. Verify go installation

        print("Finished cloning")

def download_ner():
    if path.isdir('data/ner'):
        print('NER data already exists, moving on...')
    else:
        print("Beginning NER download...")
        url = 'https://www.cs.bgu.ac.il/~elhadad/nlpproj/naama/tagged_corpus.txt'
        os.makedirs('data/ner')
        urllib.request.urlretrieve(url, f'data/ner/ner.txt')
        print("Finished downloading")

def download_sentiment():
    if path.isdir('data/sentiment'):
        print('Sentiment data already exists, moving on...')
    else:
        print("Sentiment data download...")
        raw_url = 'https://raw.githubusercontent.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew/master/data/'
        os.makedirs('data/sentiment')
        for set in tqdm(
               ['train','BiLSTM_pos_weight_8'],
                desc='Downloading..', unit='file'):
            file = 'token_' + set + '.tsv'
            url = raw_url + file
            urllib.request.urlretrieve(url, f'data/sentiment/{file}')
        print("Finished downloading")


if __name__ == '__main__':
    download_hebrew_elmo()
    download_tree_bank()
    # download_yap()
    download_ner()
    download_sentiment()

    # install the ELMO package
    # subprocess.run("python ELMoForManyLangs/setup.py install")
