from __future__ import print_function
import argparse
import pickle

import numpy as np
import torch

import cbig.osama2024.misc as misc


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
    
    # new lines
    device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    input_val = torch.tensor(in_val, dtype=torch.float32).to(device)
    input_cat = torch.tensor(in_cat, dtype=torch.float32).to(device)


    with torch.no_grad():


        out_cat_seq, out_val_seq = [], []
        
        # copy
        cat_seq = input_cat.clone()
        val_seq = input_val.clone()

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

        
    out_cat = torch.stack(out_cat_seq, dim=0)
    out_val = torch.stack(out_val_seq, dim=0)

    return out_cat, out_val

    with torch.no_grad():
        #out_cat, out_val = model(in_cat, in_val)
        out_cat = model(input_cat, input_val)
    out_cat = out_cat.cpu().numpy()
    #out_val = out_val.cpu().numpy()

    #assert out_cat.shape[1] == out_val.shape[1] == 1
    assert out_cat.shape[1] == 1
    
    return out_cat#, out_val


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
        #new lines
        #ocat = predict_subject(model, icat, ival, itime)
        
        #covert to numpy
        oval = oval.detach().numpy()
        ocat = ocat.detach().numpy()
        
        oval = oval[-duration:, 0, indices] * stds + mean

        ret['DX'].append(ocat[-duration:, 0, :])
        #ret['ADAS13'].append(0)
        #ret['Ventricles'].append(0)
        ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret['Ventricles'].append(
           misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))

    return ret



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    return parser.parse_args()


def main(args):
    """
    Predict Alzheimer’s disease progression using a trained model
    Save prediction as a csv file
    Args:
        args: includes model path, input/output paths
    Returns:
        None
    """
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.checkpoint)
    model.to(device)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)

    prediction = predict(model, data['test'], data['pred_start'],
                         data['duration'], data['baseline'])
    misc.build_pred_frame(prediction, args.out)


if __name__ == '__main__':
    main(get_args())
