import torch
from elmoformanylangs.elmo import read_list, create_batches
from torch.nn import L1Loss
from torch.optim import Adam

from elmo_on_md.data_loaders.tree_bank_loader import Token_loader
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo

def train():
    embedder = get_pretrained_elmo()
    model = embedder.model
    # create training parameters
    n_epochs = 10

    # create input data
    tokens = Token_loader().load_data()['test']
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = transform_input(tokens, embedder)

    # create MD data
    # TODO create MA data
    ma_data = None

    # create the MD module
    md_model = torch.nn.Linear(1024,4) # TODO create architectures
    criterion = L1Loss()
    optimizer = Adam(lr=0.01)

    for i in range(n_epochs):
        # mini batches
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            optimizer.zero_grad()

            # forward + backward + optimize
            embedding = model.forward(w, c, masks)
            output = md_model(embedding)

            loss = criterion(output, ma_data)
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

if __name__ == '__main__':
    train()
