import os
import pickle
from typing import List

import torch
import datetime

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from ELMoForManyLangs.elmoformanylangs.elmo import read_list, create_batches
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from elmo_on_md.data_loaders.tree_bank_loader import TokenLoader, MorphemesLoader
from elmo_on_md.model.bi_lstm import BiLSTM
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo


def train(tb_dir: str = 'default',
          positive_weight: float = 3,
          n_epochs: int = 3,
          use_power_set: bool = False,
          min_appearance_threshold: int = 0,
          combine_yy: bool = False,
          lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create the pretrained elmo model
    embedder = get_pretrained_elmo()
    elmo_model = embedder.model

    # some training parameters
    max_sentence_length = MorphemesLoader().max_sentence_length

    # create input data
    tokens = TokenLoader().load_data()
    train_w, train_c, train_lens, train_masks, train_text, recover_ind = transform_input(tokens['train'], embedder, 32)
    val_w, val_c, val_lens, val_masks, val_text, val_recover_ind = transform_input(tokens['dev'], embedder, 8)

    # create MD data
    md_loader = MorphemesLoader(use_power_set=use_power_set, min_appearance_threshold=min_appearance_threshold,
                                combine_yy=combine_yy)
    md_data = md_loader.load_data()
    train_md_labels = split_data(md_data['train'], recover_ind, train_lens, use_power_set=use_power_set)
    val_md_labels = split_data(md_data['dev'], val_recover_ind, val_lens, use_power_set=use_power_set)
    val_md_labels = torch.cat(val_md_labels)
    total_pos_num = md_loader.max_power_set_key if use_power_set else md_loader.max_pos_id

    # create the MD module
    md_model = BiLSTM(n_tags=total_pos_num, device=device, p_dropout=0.0)
    full_model = nn.Sequential(elmo_model, md_model).to(device)

    # create the tensorboard
    path = os.path.join('../../elmo_tb_runs/', tb_dir)  # , str(datetime.datetime.now()))
    writer = SummaryWriter(path)
    global_step = 0

    criterion = nn.CrossEntropyLoss() if use_power_set else \
        nn.BCEWithLogitsLoss(pos_weight=torch.ones(total_pos_num) * positive_weight)  # Binary cross entropy
    optimizer = Adam(full_model.parameters(), lr=lr)

    def validate():
        with torch.no_grad():
            y_pred = []
            for w, c, lens, masks, texts in zip(val_w, val_c, val_lens, val_masks, val_text):
                output = elmo_model.forward(w.to(device), c.to(device),
                                            [masks[0].to(device), masks[1].to(device), masks[2].to(device)])
                output = md_model(output)

                # apply mask
                sentence_mask = masks[0].to(device)[:, :, None].float()
                output = output * sentence_mask

                target = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
                target[:, :output.shape[1], :] = output
                y_pred.append(target)
            y_pred = torch.cat(y_pred, dim=0)
            if use_power_set:
                y_pred = nn.Softmax(dim=-1)(y_pred).argmax(dim=-1)
            else:
                y_pred = nn.Sigmoid()(y_pred) > 0.5
            precision, recall, f_score, support = precision_recall_fscore_support(val_md_labels.reshape(-1),
                                                                                  y_pred.reshape(-1))
            return precision[1], recall[1], f_score[1], support[1]

    for epoch in tqdm(range(n_epochs), desc='epochs', unit='epoch'):
        # mini batches
        for w, c, lens, masks, texts, labels in zip(train_w, train_c, train_lens, train_masks, train_text,
                                                    train_md_labels):
            optimizer.zero_grad()

            # forward + pad with zeros
            w, c, masks = w.to(device), c.to(device), [masks[0].to(device), masks[1].to(device), masks[2].to(device)]
            output = elmo_model.forward(w, c, masks)
            output = md_model(output)

            # apply mask
            sentence_mask = masks[0].to(device)[:, :, None].float()
            output = output * sentence_mask

            # pad with zeros to fit the labels
            full_output = torch.zeros((output.shape[0], max_sentence_length, total_pos_num))
            full_output[:, :output.shape[1], :] = output

            # change format if using power set
            full_output = full_output.transpose(-2, -1) if use_power_set else full_output

            loss = criterion(full_output, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            writer.add_scalar('train_loss', loss, global_step=global_step)

            # validation set
            if global_step % 10 == 0:
                output.to('cpu')
                full_output.to('cpu')
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

    # create batches from the input data.
    test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
        test, batch_size, embedder.word_lexicon, embedder.char_lexicon, embedder.config, text=text)

    return test_w, test_c, test_lens, test_masks, test_text, recover_ind


def split_data(ma_data: torch.tensor, recover_ind: List[int], batch_lens: int, use_power_set=False):
    ma_data = ma_data.long() if use_power_set else ma_data
    batch_sizes = [len(batch_lens[i]) for i in range(len(batch_lens))]
    indices = torch.tensor(recover_ind)
    indices = torch.split(indices, batch_sizes)
    splited_data = [ma_data[batch_indices] for batch_indices in indices]
    return splited_data


if __name__ == '__main__':
    new_model_name = 'pos_weight8_lr-4_old_tags_30epochs'
    new_embedder = train(tb_dir=new_model_name, n_epochs=30, positive_weight=8, lr=1e-4)
    with open(f'trained_models/{new_model_name}.pkl', 'wb') as file:
        pickle.dump(new_embedder, file)
