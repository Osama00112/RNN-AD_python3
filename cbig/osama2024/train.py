from __future__ import print_function, division
import argparse
import json
import time
import pickle

import numpy as np
import torch

import cbig.osama2024.misc as misc
from cbig.osama2024.model import MODEL_DICT


def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        #mask.squeeze(1).astype(np.uint8), dtype=torch.uint8)] #modified
        mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]

    return torch.nn.functional.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects


def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    # indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.uint8)  #modified
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.bool)
    assert pred.shape == indices.shape
    pred[indices] = 0

    return torch.nn.functional.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects


def to_cat_seq(labels):
    """
    Return one-hot representation of a sequence of class labels
    Args:
        labels: [nb_subjects, nb_timpoints]
    Returns:
        [nb_subjects, nb_timpoints, nb_classes]
    """
    return np.asarray([misc.to_categorical(c, 3) for c in labels])


def train_1epoch(args, model, dataset, optimizer):
    """
    Train an recurrent model for 1 epoch
    Args:
        args: include training hyperparametres and input/output paths
        model: pytorch model object
        dataset: training data
        optimizer: optimizer
    Returns:
        cross-entropy loss of epoch
        mean absolute error (MAE) loss of epoch
    """
    model.train()
    total_ent = total_mae = 0
    
    #print("Number of batches:", len(dataset))
    #print("Total number of subjects:", len(dataset.subjects))

    
    for batch in dataset:
        #print("Input data shape:", batch['cat'].shape)
        #print("Mask data shape:", batch['cat_msk'].shape)

        if len(batch['tp']) == 1:
            continue

        optimizer.zero_grad()
        seq_class_labels = to_cat_seq(batch['cat'])
        seq_val_labels = batch['val']
        
        #convert to tensors
        seq_class_labels = torch.tensor(seq_class_labels, dtype=torch.float32)
        seq_val_labels = torch.tensor(seq_val_labels, dtype=torch.float32)
        
        #feed to model
        #pred_cat = model(seq_class_labels, seq_val_labels)
        # set pred_value to random
        #pred_val = torch.rand(seq_val_labels.shape)
        
        
        # print("Output type:", type(pred_cat))  # Assuming pred_cat and pred_val have similar types
        # print("Output structure:", pred_cat.shape)  # Assuming pred_cat and pred_val have similar structures
        
        #pred_cat, pred_val = model(to_cat_seq(batch['cat']), batch['val'])
        #pred_cat, pred_val = model(seq_class_labels, seq_val_labels)

        out_cat_seq, out_val_seq = [], []
        
        # copy
        cat_seq = seq_class_labels.clone()
        val_seq = seq_val_labels.clone()

        hidden = model.init_hidden_state(val_seq.shape[1])
        masks = model.dropout_mask(val_seq.shape[1])

        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, hidden = model.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = torch.isnan(val_seq[j])
            #_val_seq[j][idx] = o_val.data.cpu().numpy()[idx]
            val_seq[j][idx] = torch.tensor(o_val.data.cpu().numpy()[idx], dtype=torch.float32)


            idx = torch.isnan(cat_seq[j])
            #_cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]
            cat_seq[j][idx] = torch.tensor(o_cat.data.cpu().numpy()[idx], dtype=torch.float32)

            
        pred_cat, pred_val = torch.stack(out_cat_seq), torch.stack(out_val_seq)


        # output = model(seq_class_labels, seq_val_labels)
        
        mask_cat = batch['cat_msk'][1:]
        assert mask_cat.sum() > 0

        ent = ent_loss(pred_cat, batch['true_cat'][1:], mask_cat)
        mae = mae_loss(pred_val, batch['true_val'][1:], batch['val_msk'][1:])
        #mae = 0

        
        total_loss = mae + args.w_ent * ent

        total_loss.backward()
        optimizer.step()
        batch_size = mask_cat.shape[1]
        total_ent += ent.item() * batch_size
        total_mae += mae.item() * batch_size
        #total_mae += 0

    return total_ent / len(dataset.subjects), total_mae / len(dataset.subjects)


def save_config(args, config_path):
    """
    Save training configuration as json file
    Args:
        args: include training hyperparametres and input/output paths
        config_path: path of output json file
    Returns:
        None
    """
    with open(config_path, 'w') as fhandler:
        print(json.dumps(vars(args), sort_keys=True), file=fhandler)


def train(args):
    """
    Train an recurrent model
    Args:
        args: include training hyperparametres and input/output paths
    Returns:
        None
    """
    log = print if args.verbose else lambda *x, **i: None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)
    nb_measures = len(data['train'].value_fields())

    model_class = MODEL_DICT[args.model]
    model = model_class(
        nb_classes=3,
        nb_measures=nb_measures,
        nb_layers=args.nb_layers,
        h_size=args.h_size,
        h_drop=args.h_drop,
        i_drop=args.i_drop)
    setattr(model, 'mean', data['mean'])
    setattr(model, 'stds', data['stds'])

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    log(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    #print("Model:", model)
    #print("Optimizer:", optimizer)
    

    start = time.time()
    try:
        for i in range(args.epochs):
            loss = train_1epoch(args, model, data['train'], optimizer)
            log_info = (i + 1, args.epochs, misc.time_from(start)) + loss
            log('%d/%d %s ENT %.3f, MAE %.3f' % log_info)
    except KeyboardInterrupt:
        print('Early exit')

    # Assuming 'model' is your RNN model
    torch.save(model.state_dict(), 'output/model_state_dict.pt')

    torch.save(model, args.out)
    save_config(args, '%s.json' % args.out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', required=True)

    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--w_ent', type=float, default=1.)

    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--h_size', type=int, default=512)
    parser.add_argument('--i_drop', type=float, default=.0)
    parser.add_argument('--h_drop', type=float, default=.0)
    parser.add_argument('--weight_decay', type=float, default=.0)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
