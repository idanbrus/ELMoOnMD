# ElmoOnMD

ELMo On MD is an improved language representation model created on top of ELMo,
enhancing it to better cope with morphological-complex languages.
It was specifically created and tested on the Hebrew language.

This work has been created as a final project in NLP class in the Open University of Israel, as part of a masters degree.
### Installation
* Anaconda 3.7 required 
* Create an Anconda virutal environment using environment.yml
* Run "python setup_project.py" to download all the relevant data and pretrained models
* Unzip the file hebrew.zip downloaded into ELMoForManyLangs folder (just unzip in the file in it's location).
* in hebrew/config.json file, change the config_path, to "<root>/ELMoForManyLangs/hebrew/cnn_50_100_512_4096_sample.json"

### Instructions
* To train your own ELMo On MD, run elmo_on_md/model/train.py.
* Our trained model is saved under model/trained_models/elmo_on_md.pkl
* To recreate results on NER task, use notebooks/NER.ipynb
* To recreate results on Sentiment Analysis task, use notebooks/Sentiment.ipynb

