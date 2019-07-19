# 1. Use this script to download the hebrew elmo and the hebrew tree bank
# 2. after downloading, unzip the file hebrew.zip file
# 3. change the config.json file, with the path of the configuration file ({rel_path}/ElmoOnMD/ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json)
# 4. run the setup file to install the package

import urllib.request
from itertools import product
import subprocess

from tqdm import tqdm

if __name__ == '__main__':
    # Download Hebrew ELMo
    print("Beginning Hebrew ELMo download...")
    url = 'http://vectors.nlpl.eu/repository/11/154.zip'
    urllib.request.urlretrieve(url, 'ELMoForManyLangs/hebrew.zip')
    print("Finished downloading")

    # Download Hebrew Tree Bank
    print("Beginning Hebrew Tree Bank download...")
    root_url = 'https://raw.github.com/OnlpLab/HebrewResources/master/HebrewTreebank/hebtb/'
    for set, data in tqdm(
            list(product(['train', 'dev', 'test'], ['-gold.conll', '-gold.lattices', '.lattices', '.tokens'])),
            desc='Downloading..', unit='file'):
        file = set + '_hebtb' + data
        url = root_url + file
        urllib.request.urlretrieve(url, f'data/hebrew_tree_bank/{file}')
    print("Finished downloading")

    # install the ELMO package
    subprocess.run("python ELMoForManyLangs/setup.py install")
