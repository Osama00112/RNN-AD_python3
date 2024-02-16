from __future__ import print_function
import argparse
import pickle

import numpy as np
import torch
from captum.attr import IntegratedGradients

import cbig.osama2024.misc as misc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    return parser.parse_args()


def predict_subject(model, cat_seq, value_seq, time_seq):
    """
    Predict Alzheimer’s disease progression for a subject
    Args:
        model: trained pytorch model
        cat_seq: sequence of diagnosis [nb_input_timpoints, nb_classes]
        value_seq: sequence of other features [nb_input_timpoints, nb_features]
        time_seq: months from baseline [nb_output_timpoints, nb_features]
    nb_input_timpoints <= nb_output_timpoints
    Returns:
        out_cat: predicted diagnosis
        out_val: predicted features
    """
    in_val = np.full((len(time_seq), ) + value_seq.shape[1:], np.nan)
    in_val[:len(value_seq)] = value_seq

    in_cat = np.full((len(time_seq), ) + cat_seq.shape[1:], np.nan)
    in_cat[:len(cat_seq)] = cat_seq

    with torch.no_grad():
        out_cat, out_val = model(in_cat, in_val)
    out_cat = out_cat.cpu().numpy()
    out_val = out_val.cpu().numpy()

    assert out_cat.shape[1] == out_val.shape[1] == 1

    return out_cat, out_val


def predict(model, dataset, pred_start, duration, baseline):
    """
    Predict Alzheimer’s disease progression using a trained model
    Args:
        model: trained pytorch model
        dataset: test data
        pred_start (dictionary): the date at which prediction begins
        duration (dictionary): how many months into the future to predict
        baseline (dictionary): the baseline date
    Returns:
        dictionary which contains the following key/value pairs:
            subjects: list of subject IDs
            DX: list of diagnosis prediction for each subject
            ADAS13: list of ADAS13 prediction for each subject
            Ventricles: list of ventricular volume prediction for each subject
    """
    model.eval()
    ret = {'subjects': dataset.subjects}
    ret['DX'] = []  # 1. likelihood of NL, MCI, and Dementia
    ret['ADAS13'] = []  # 2. (best guess, upper and lower bounds on 50% CI)
    ret['Ventricles'] = []  # 3. (best guess, upper and lower bounds on 50% CI)
    ret['dates'] = misc.make_date_col(
        [pred_start[s] for s in dataset.subjects], duration)

    col = ['ADAS13', 'Ventricles', 'ICV']
    indices = misc.get_index(list(dataset.value_fields()), col)
    mean = model.mean[col].values.reshape(1, -1)
    stds = model.stds[col].values.reshape(1, -1)

    for i in range(len(dataset)):  # Iterate over subjects using indices
        data = dataset[i]  # Access data using index
        # print the columns
        #print(data.keys())
        rid = data['rid']
        all_tp = data['tp'].squeeze(axis=1)
        start = misc.month_between(pred_start[rid], baseline[rid])
        assert np.all(all_tp == np.arange(len(all_tp)))
        mask = all_tp < start
        itime = np.arange(start + duration)
        icat = np.asarray(
            [misc.to_categorical(c, 3) for c in data['cat'][mask]])
        ival = data['val'][:, None, :][mask]

        ocat, oval = predict_subject(model, icat, ival, itime)
        oval = oval[-duration:, 0, indices] * stds + mean

        ret['DX'].append(ocat[-duration:, 0, :])
        ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret['Ventricles'].append(
            misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))

    return ret


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.checkpoint)
    model.to(device)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)

    prediction = predict(model, data['test'], data['pred_start'],
                         data['duration'], data['baseline'])

    # Initialize IntegratedGradients object
    integrated_gradients = IntegratedGradients(model)

    # Load the val.pkl data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    # Prepare input data for the model
    input_cat = data['val_cat']
    input_val = data['val_val']

    # Convert input data to torch tensors
    input_cat_tensor = torch.tensor(input_cat, dtype=torch.float32).to(device)
    input_val_tensor = torch.tensor(input_val, dtype=torch.float32).to(device)

    # Compute integrated gradients
    attr_cat, attr_val = integrated_gradients.attribute(
        inputs=(input_cat_tensor, input_val_tensor),
        baselines=None,  # You can provide baselines if needed
        target=None  # You can specify the target output index if needed
    )

    # Save the integrated gradients along with the predictions
    with open(args.out, 'wb') as f:
        pickle.dump((prediction, attr_cat.cpu().numpy(), attr_val.cpu().numpy()), f)


if __name__ == '__main__':
    main(get_args())
