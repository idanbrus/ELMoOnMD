import pickle
from typing import List

import torch
from ELMoForManyLangs.elmoformanylangs.elmo import read_list, create_batches
from torch.optim import Adam
from torch import nn
from elmo_on_md.data_loaders.tree_bank_loader import Token_loader, Morphemes_loader
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo


def train():
    # create the pretrained elmo model
    embedder = get_pretrained_elmo()
    model = embedder.model

    # create training parameters
    n_epochs = 2
    total_pos_num = Morphemes_loader().max_morpheme_count
    max_sentence_length = Morphemes_loader().max_sentence_length

    # create input data
    tokens = Token_loader().load_data()['test']
    train_w, train_c, train_lens, train_masks, train_text, recover_ind = transform_input(tokens, embedder)

    # create MD data
    md_labels = Morphemes_loader().load_data()['test']
    md_labels = split_data(md_labels, recover_ind, train_lens)

    # create the MD module
    md_model = nn.Sequential(nn.Linear(1024, total_pos_num), nn.Sigmoid())  # TODO decide architectures
    full_model = nn.Sequential(model, md_model)
    criterion = nn.BCELoss()  # Binary cross entropy
    optimizer = Adam(full_model.parameters(), lr=0.01)

    for i in range(n_epochs):
        # mini batches
        for w, c, lens, masks, texts, labels in zip(train_w, train_c, train_lens, train_masks, train_text, md_labels):
            optimizer.zero_grad()

            # forward + pad with zeros
            output = model.forward(w, c, masks).mean(dim=0)
            output = md_model(output)
            target = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
            target[:, :output.shape[1], :] = output

            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()

    return embedder


# code taken from ELMO for Many LAN
def transform_input(tokens, embedder):
    test, text = read_list(tokens)

    # create test batches from the input data.
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
        test, embedder.batch_size, embedder.word_lexicon, embedder.char_lexicon, embedder.config, text=text)

    return test_w, test_c, test_lens, test_masks, test_text, recover_ind


def split_data(ma_data: torch.tensor, recover_ind: List[int], batch_lens: int):
    batch_sizes = [len(batch_lens[i]) for i in range(len(batch_lens))]
    indices = torch.tensor(recover_ind)
    indices = torch.split(indices, batch_sizes)
    splited_data = [ma_data[batch_indices] for batch_indices in indices]
    return splited_data


if __name__ == '__main__':
    new_embedder = train()
    with open('trained_models/new_model.pkl', 'wb') as file:
        pickle.dump(new_embedder, file)
