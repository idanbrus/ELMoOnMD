import os
import pickle
from typing import List

import torch
import datetime

from sklearn.metrics import precision_recall_fscore_support

from ELMoForManyLangs.elmoformanylangs.elmo import read_list, create_batches
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from elmo_on_md.data_loaders.tree_bank_loader import TokenLoader, MorphemesLoader
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo


def train(tb_dir: str = 'default',
          positive_weight: float = 1,
          n_epochs: int = 3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create the pretrained elmo model
    embedder = get_pretrained_elmo()
    elmo_model = embedder.model

    # some training parameters
    total_pos_num = MorphemesLoader().max_morpheme_count
    max_sentence_length = MorphemesLoader().max_sentence_length

    # create input data
    tokens = TokenLoader().load_data()
    train_w, train_c, train_lens, train_masks, train_text, recover_ind = transform_input(tokens['train'], embedder, 64)
    val_w, val_c, val_lens, val_masks, val_text, val_recover_ind = transform_input(tokens['dev'], embedder, 8)

    # create MD data
    md_data = MorphemesLoader().load_data()
    train_md_labels = split_data(md_data['train'], recover_ind, train_lens)
    val_md_labels = split_data(md_data['dev'], val_recover_ind, val_lens)
    val_md_labels = torch.cat(val_md_labels)

    # create the MD module
    md_model = nn.Sequential(nn.Linear(1024, 512),
                             nn.ReLU(),
                             nn.Linear(512, total_pos_num))  # TODO decide architectures
    full_model = nn.Sequential(elmo_model, md_model).to(device)

    # create the tensorboard
    path = os.path.join('../../tensorboard_runs/', tb_dir)  # , str(datetime.datetime.now()))
    writer = SummaryWriter(path)
    global_step = 0

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(total_pos_num) * positive_weight)  # Binary cross entropy
    optimizer = Adam(full_model.parameters(), lr=1e-2)

    def validate():
        with torch.no_grad():
            y_pred = []
            for w, c, lens, masks, texts in zip(val_w, val_c, val_lens, val_masks, val_text):
                output = elmo_model.forward(w.to(device), c.to(device),
                                            [masks[0].to(device), masks[1].to(device), masks[2].to(device)]).mean(dim=0)
                output = md_model(output)
                # apply mask
                sentence_mask = masks[0].to(device)[:,:,None].float()
                output = output * sentence_mask

                target = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
                target[:, :output.shape[1], :] = output
                y_pred.append(target)
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = nn.Sigmoid()(y_pred) > 0.5
            precision, recall, f_score, support = precision_recall_fscore_support(val_md_labels.reshape(-1),
                                                                                  y_pred.reshape(-1))
            return precision[1], recall[1], f_score[1], support[1]

    for epoch in range(n_epochs):
        # mini batches
        for w, c, lens, masks, texts, labels in zip(train_w, train_c, train_lens, train_masks, train_text,
                                                    train_md_labels):
            optimizer.zero_grad()

            # forward + pad with zeros
            w, c, masks = w.to(device), c.to(device), [masks[0].to(device), masks[1].to(device), masks[2].to(device)]
            output = elmo_model.forward(w, c, masks).mean(dim=0)
            output = md_model(output)

            # apply mask
            sentence_mask = masks[0].to(device)[:, :, None].float()
            output = output * sentence_mask

            # pad with zeros to fit the labels
            target = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
            target[:, :output.shape[1], :] = output

            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', loss, global_step=global_step)

            # validation set
            if global_step % 10 == 0:
                output.to('cpu')
                target.to('cpu')
                loss.to('cpu')
                precision, recall, f_score, _ = validate()
                writer.add_scalar('validation/Precision', precision, global_step=global_step)
                writer.add_scalar('validation/Recall', recall, global_step=global_step)
                writer.add_scalar('validation/F_score', f_score, global_step=global_step)

            global_step += 1

    return embedder


# code taken from ELMO for Many LAN
def transform_input(tokens, embedder, batch_size=None):
    batch_size = embedder.batch_size if batch_size is None else batch_size
    test, text = read_list(tokens)

    # create test batches from the input data.
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
        test, batch_size, embedder.word_lexicon, embedder.char_lexicon, embedder.config, text=text)

    return test_w, test_c, test_lens, test_masks, test_text, recover_ind


def split_data(ma_data: torch.tensor, recover_ind: List[int], batch_lens: int):
    batch_sizes = [len(batch_lens[i]) for i in range(len(batch_lens))]
    indices = torch.tensor(recover_ind)
    indices = torch.split(indices, batch_sizes)
    splited_data = [ma_data[batch_indices] for batch_indices in indices]
    return splited_data


if __name__ == '__main__':
    new_model_name = 'test'
    new_embedder = train(tb_dir=new_model_name,n_epochs = 3, positive_weight=10)
    with open(f'trained_models/{new_model_name}.pkl', 'wb') as file:
        pickle.dump(new_embedder, file)
