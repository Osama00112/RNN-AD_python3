from __future__ import print_function, division
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

def load_features(feat_file_path):
    """
    Load features from a file.
    Features are separated by newline
    """
    with open(feat_file_path, 'r') as f:
        features = f.read().split('\n')
    return features

def time_from(start):
    """
    return duration from start time to now
    """
    
    duration = relativedelta(seconds=time.time() - start)
    return '%dm %ds' % (duration.minutes, duration.seconds)

def str_to_date(date_str):
    """
    Convert string to datetime object
    """
    return datetime.strptime(date_str, '%Y-%m-%d')

def has_data_mask(frame):
    """
    check whether rows have any valid value (not NaN)
    
    Args:
        frame: pandas.DataFrame
    Returns:
        (ndarray): boolean mask with the same number of rows as frame
        True if row has atleast one valid value
    """
    
    return np.any(~frame.isnull(), axis=1)

def get_data_dict(frame, features):
    """
    from a frame of all subjects, return a dictionary of frames
    The keys are subject's ID
    the data frames are
        - sorted by *Month_bl* (which are integers)
        - have empty rows dropped (empty row has no value in *features* list)
    
    Args:
        frame (Pandas.DataFrame): data frame with all subjects
        features (list of strings): list of features to keep
    Returns:
        (Pandas.Dataframe) : prediction frame
    """
    ret = {}
    frame_ = frame.copy()
    frame_['Month_bl'] = frame_['Month_bl'].round().astype(int) 
    
    for subj in np.unique(frame_.RID):
        subj_frame = frame_[frame_.RID == subj]
        subj_frame = subj_frame.sort_values('Month_bl')
        subj_frame = subj_frame[has_data_mask(subj_frame[features])]
        
        subj_frame = subj_frame.set_index('Month_bl', drop=True)
        ret[subj] = subj_frame.drop('RID', axis=1)
    return ret


def build_pred_frame(prediction, outpath=''):
    """
    Construct the forecast spreadsheet following TADPOLE format
    Args:
        prediction (dictionary): contains the following key/value pairs:
            dates: dates of predicted timepoints for each subject
            subjects: list of subject IDs
            DX: list of diagnosis prediction for each subject
            ADAS13: list of ADAS13 prediction for each subject
            Ventricles: list of ventricular volume prediction for each subject
        outpath (string): where to save the prediction frame
        If *outpath* is blank, the prediction frame is not saved
    Return:
        (Pandas data frame): prediction frame
    """
    
    table = pd.DataFrame()
    print("Length of DataFrame index after defining", len(table.index))
    dates = prediction['dates']
    table['RID'] = np.concatenate([[subj] * len(sublist) for subj, sublist in zip(prediction['subjects'], dates)])
    table['Forecast Month'] = np.concatenate(
        [np.arange(1, len(sublist) + 1) for sublist in dates]
    )
    
    # Construct the forecast month column using the length of the first Timestamp object

    table['Forecast Date'] = np.concatenate(dates)
    
    diag = np.concatenate(prediction['DX'])
    table['CN related probability'] = diag[:, 0]
    table['MCI related probability'] = diag[:, 1]
    table['AD related probability'] = diag[:, 2]
    
    adas13 = np.concatenate(prediction['ADAS13'])
    table['ADAS13'] = adas13[:, 0]
    table['ADAS13 50% CI lower'] = adas13[:, 1]
    table['ADAS13 50% CI upper'] = adas13[:, 2]
    
    ventricles = np.concatenate(prediction['Ventricles'])
    table['Ventricles_ICV'] = ventricles[:, 0]
    table['Ventricles_ICV 50% CI lower'] = ventricles[:, 1]
    table['Ventricles_ICV 50% CI upper'] = ventricles[:, 2]
    
    assert len(diag) == len(adas13) == len(ventricles)
    
    if outpath:
        table.to_csv(outpath, index=False)
    return table

    

def month_between(end, start):
    """
    return the number of months between two dates
    """
    diff = relativedelta(end, start)
    months = diff.years * 12 + diff.months
    to_next = relativedelta(end, start + relativedelta(months=1, days=-diff.days),
                            end).days
    
    to_prev = diff.days
    
    return months + (to_next > to_prev)

def make_date_col(starts, duration):
    """
    Return a list of list of dates
    The start date of each list of dates is specified by *starts*
    """
    date_range = [relativedelta(months=i) for i in range(duration)]
    ret = []
    for start in starts:
        ret.append([start + d for d in date_range])

    return ret

def get_index(fields, keys):
    """
    return the index of *keys* in *fields*
    """
    assert isinstance(fields, list)
    assert isinstance(keys, list)
    return [fields.index(key) for key in keys]
    
    
    
def to_categorical(y, nb_classes):
    """
    Convert list of labels to one-hot encoding
    """
    
    if (len(y.shape) == 2):
        y = y.squeeze(1)
        
    
    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)
    
    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1
    
    return ret_mat

def log_result(result, path, verbose):
    """ Output result to screen/file """
    frame = pd.DataFrame([result])[['mAUC', 'bca' , 'adasMAE', 'ventsMAE']]
    
    if verbose:
        print(frame)
    if path:
        frame.to_csv(path, index=False)
        
        
        
def PET_conv(value):
    """ Convert PET measures from str to float """
    try:
        return float(value.strip().strip('>'))
    except ValueError:
        return float(np.nan)
    
    
def Diagnosis_conv(value):
    """ Convert diagnosis from str to int """
    if value == 'CN':
        return 0
    elif value == 'MCI':
        return 1
    elif value == 'AD':
        return 2
    else:
        return float('NaN')
    
def DX_conv(value):
    """ Convert diagnosis from int to str """
    if isinstance(value, str):
        if value.endswith('Dementia'):
            return 2.
        if value.endswith('MCI'):
            return 1.
        if value.endswith('CN'):
            return 0.
        
    return float('NaN')

def add_ci_col(values, ci, lo, hi):
    """ add lower/upper confidence interval to a list of values """
    
    return np.clip(np.vstack([values, values - ci, values + ci]).T, lo, hi)
 

def censor_d1_table(_table):
    """ censor the D1 table """
    _table.drop(3229, inplace=True)  # RID 2190, Month = 3, Month_bl = 0.45
    
    #beforing dropping check the row
    #print(_table.loc[4372])
    
    
    _table.drop(4372, inplace=True)  # RID 4579, Month = 3, Month_bl = 0.32
    _table.drop(
        8376, inplace=True # Duplicate row for subject 1088 at 72 months
    )
    _table.drop(
        8586, inplace=True # Duplicate row for subject 1195 at 48 months
    )
    _table.loc[
        12215,
        'Month_bl'
    ] = 48  # Wrong EXAMDATE and Month_bl for subject 4960
    
    _table.drop(10254, inplace=True)  # Abnormaly small ICV for RID 4674
    _table.drop(12245, inplace=True)  # Row without measurements, subject 5204
    

def load_table(csv, columns):
    """ Load CSV, only include specified columns """
    table = pd.read_csv(csv, converters=CONVERTERS, usecols=columns)
    censor_d1_table(table)
    
    return table  

CONVERTERS = {
    'CognitiveAssessmentDate': str_to_date,
    'ScanDate': str_to_date,
    'Forecast Date': str_to_date,
    'EXAMDATE': str_to_date,
    'Diagnosis': Diagnosis_conv,
    'DX': DX_conv,
    'PTAU_UPENNBIOMK9_04_19_17': PET_conv,
    'TAU_UPENNBIOMK9_04_19_17': PET_conv,
    'ABETA_UPENNBIOMK9_04_19_17': PET_conv
}
    

def get_baseline_prediction_start(frame):
    """ Get baseline dates and dates when prediction starts"""
    
    one_month = relativedelta(months=1)
    baseline = {}
    start = {}
    for subj in np.unique(frame.RID):
        dates = frame.loc[frame.RID == subj, 'EXAMDATE']
        baseline[subj] = dates.min()
        start[subj] = dates.min() + one_month
        
    return baseline, start

def get_mask(csv_path, use_validation):
    """ Get masks from CSV"""
    cols = ['RID', 'EXAMDATE', 'train', 'val', 'test']
    frame = load_table(csv_path, cols)
    train_mask = frame.train == 1
    if use_validation:
        pred_mask = frame.val == 1
    else:
        pred_mask = frame.test == 1
        
    return train_mask, pred_mask, frame[pred_mask]

def read_csv(path):
    return pd.read_csv(path, converters=CONVERTERS)