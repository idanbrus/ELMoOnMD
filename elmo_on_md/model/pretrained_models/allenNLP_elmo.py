import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def get_pretrained_allenNLP_elmo():
    elmo = Elmo(options_file, weight_file, 2, dropout=0).train()
    init_weights(elmo)
    return elmo

def init_weights(model : torch.nn.Module):
    if len(list(model.children())) == 0:
        torch.nn.init.xavier_uniform(model.weight)
    else:
        map(init_weights, model.children())

