from typing import List

import torch
from ELMoForManyLangs.elmoformanylangs.elmo import read_list, create_batches
from torch.nn import L1Loss
from torch.optim import Adam
from torch import nn
from elmo_on_md.data_loaders.tree_bank_loader import Token_loader
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo


def train():
    # create the elmo model
    embedder = get_pretrained_elmo()
    model = embedder.model

    # create training parameters
    n_epochs = 10
    total_pos_num = 30
    max_sentence_length = 50

    # create input data
    tokens = Token_loader().load_data()['test']
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = transform_input(tokens, embedder)

    # create MD data
    # TODO create MA data
    md_labels = torch.zeros([len(tokens), max_sentence_length, total_pos_num])
    md_labels = split_data(md_labels, recover_ind, embedder.batch_size)

    # create the MD module
    md_model = nn.Sequential(nn.Linear(1024, total_pos_num) ,nn.Sigmoid()) # TODO create architectures
    full_model = nn.Sequential(model, md_model)
    criterion = nn.BCELoss(reduction='sum')  # Binary cross entropy
    optimizer = Adam(full_model.parameters(), lr=0.01)

    for i in range(n_epochs):
        # mini batches
        for w, c, lens, masks, texts, labels in zip(test_w, test_c, test_lens, test_masks, test_text, md_labels):
            optimizer.zero_grad()

            # forward + pad with zeros
            output = model.forward(w, c, masks).mean(dim=0)
            output = md_model(output)
            target = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
            target[:,:output.shape[1], :] = output

            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()

    return embedder

    # embedding = embedder.sents2elmo(tokens)


# code taken from ELMO for Many LAN
def transform_input(tokens, embedder):
    test, text = read_list(tokens)

    # create test batches from the input data.
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
        test, embedder.batch_size, embedder.word_lexicon, embedder.char_lexicon, embedder.config, text=text)

    return test_w, test_c, test_lens, test_masks, test_text, recover_ind


def split_data(ma_data: torch.tensor, recover_ind: List[int], batch_size: int):
    indices = torch.tensor(recover_ind)
    indices = torch.split(indices, batch_size)
    splited_data = [ma_data[batch_indices] for batch_indices in indices]
    return splited_data


if __name__ == '__main__':
    train()
