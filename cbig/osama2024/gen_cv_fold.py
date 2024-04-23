import argparse
import os.path as op
import pandas as pd
import numpy as np

import cbig.osama2024.misc as misc

def split_by_median_date(data, subjects):
    """
    Split timepoints in two halves, use first half to predict second half
    Args:
        data (Pandas data frame): input data
        subjects: list of subjects
    Return:
        first_half (ndarray): boolean mask, rows used as input
        second_half (ndarray): boolean mask, rows to predict
    """
    first_half = np.zeros(data.shape[0], int)
    second_half = np.zeros(data.shape[0], int)   

    for subject in subjects:
        subj_mask = (data.RID == subject) & data.has_data
        median_date = np.sort(data.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
        first_half[subj_mask & (data.EXAMDATE < median_date)] = 1
        second_half[subj_mask & (data.EXAMDATE >= median_date)] = 1
        
    return first_half, second_half

def gen_fold(data, nb_folds, output_dir):
    """ Generate *nb_folds* cross-validation folds from *data """
    subjects = np.unique(data.RID)
    has_2_timepoints = np.array([np.sum(data.RID == subj) > 2 for subj in subjects])
    
    potential_targets = np.random.permutation(subjects[has_2_timepoints])
    folds = np.array_split(potential_targets, nb_folds)
    
    left_out = [subjects[~has_2_timepoints]]
    
    for test_fold in range(nb_folds):
        val_fold = (test_fold + 1) % nb_folds
        train_fold = [f for f in range(nb_folds) if f not in [test_fold, val_fold]]
        
        train_subjects = np.concatenate([folds[i] for i in train_fold] + left_out, axis=0)
        val_subjects = folds[val_fold]
        test_subjects = folds[test_fold]
        
        train_timepoints = (
            np.in1d(data.RID, train_subjects) & data.has_data
        ).astype(int)
        val_in_timepoints, val_out_timepoints = split_by_median_date(data, val_subjects)
        test_in_timepoints, test_out_timepoints = split_by_median_date(data, test_subjects)
        
        # print("Length of val_out_timepoints:", len(val_out_timepoints))
        # print("Length of data:", len(data))
        
        mask_frame = gen_mask_frame(data, train_timepoints, val_in_timepoints, test_in_timepoints)
        mask_frame.to_csv(op.join(output_dir, 'fold_{}_mask.csv'.format(test_fold)), index=False)
        
        val_frame = gen_ref_frame(data, val_out_timepoints)
        val_frame.to_csv(op.join(output_dir, 'fold_{}_val.csv'.format(test_fold)), index=False)
        
        test_frame = gen_ref_frame(data, test_out_timepoints)
        test_frame.to_csv(op.join(output_dir, 'fold_{}_test.csv'.format(test_fold)), index=False)
                

def gen_mask_frame(data, train, val, test):
    """ Create frame with 3 masks for train, val, test """
    col = ['RID', 'EXAMDATE']
    ret = pd.DataFrame(data[col], index=range(train.shape[0]))
    ret['train'] = train
    ret['val'] = val
    ret['test'] = test
    return ret

def gen_ref_frame(data, test_timepoint_mask):
    """ Create reference frame which is used to evalute models' prediction """
    
    columns = [
        'RID', 'CognitiveAssessmentDate', 'Diagnosis', 'ADAS13', 'ScanDate'
    ]
    
    # print("len of test_timepoint_mask:", len(test_timepoint_mask))
    
    # ret = pd.DataFrame(
    #     np.nan, index=range(len(test_timepoint_mask)), columns=columns)
    
    
    # ret[columns] = data[['RID', 'EXAMDATE', 'DXCHANGE', 'ADAS13', 'EXAMDATE']]
    # ret['Ventricles'] = data['Ventricles'] / data['ICV']
    # ret = ret[test_timepoint_mask == 1]
    
    ret = pd.DataFrame(columns=columns)

    # Assign columns explicitly
    ret['RID'] = data['RID']
    ret['CognitiveAssessmentDate'] = data['EXAMDATE']
    ret['Diagnosis'] = data['DXCHANGE']
    ret['ADAS13'] = data['ADAS13']
    ret['ScanDate'] = data['EXAMDATE']

    ret['Ventricles'] = data['Ventricles'] / data['ICV']
    ret = ret[test_timepoint_mask == 1]

    # map diagnosis from numeric categories back to labels
    mapping = {
        1: 'CN',
        7: 'CN',
        9: 'CN',
        2: 'MCI',
        4: 'MCI',
        8: 'MCI',
        3: 'AD',
        5: 'AD',
        6: 'AD'
    }
    ret.replace({'Diagnosis': mapping}, inplace=True)
    ret.reset_index(drop=True, inplace=True)

    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--spreadsheet', required=True)
    parser.add_argument('--features', required=True)    
    parser.add_argument('--folds', type=int, required=True)
    parser.add_argument('--output_dir', required=True)  
    
    
    
    args = parser.parse_args()
    
    # check if features path is valid
    if not op.isfile(args.features):
        raise ValueError('Invalid features path: %s' % args.features)
    
    np.random.seed(args.seed)
    
    columns = ['RID', 'DXCHANGE', 'EXAMDATE']
    
    features = misc.load_feature(args.features)
    frame = pd.read_csv(
        args.spreadsheet, usecols=columns + features,
        converters=misc.CONVERTERS
    )
    
    frame['has_data'] = ~frame[features].isnull().apply(np.all, axis=1)
    
    #save frame to csv
    frame.to_csv(op.join(args.output_dir, 'frame.csv'), index=False)
    
    gen_fold(frame, args.folds, args.output_dir)
    
    
if __name__ == '__main__':
    main()