import numpy as np
import torch 
import torch.nn as nn

from cbig.osama2024.rnn import LssCell
from cbig.osama2024.rnn import MinimalRNNCell

def jozefowicz_init(forget_gate):
    """
    Initialize tensor with Jozefowicz's method
    """
    
    forget_gate.data.fill_(1)
    
    

class RnnModelInterp(torch.nn.Module):
    """
    Recurrent neural network (RNN) base class
    Missing values (i.e. NaN) are filled using model prediction
    """

    def __init__(self, celltype, nb_classes, nb_measures, h_size, **kwargs):
        super(RnnModelInterp, self).__init__()
        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']

        self.hid2category = nn.Linear(h_size, nb_classes)
        self.hid2measures = nn.Linear(h_size, nb_measures)

        self.cells = nn.ModuleList()
        self.cells.append(celltype(nb_classes + nb_measures, h_size))
        for _ in range(1, kwargs['nb_layers']):
            self.cells.append(celltype(h_size, h_size))

    def init_hidden_state(self, batch_size):
        raise NotImplementedError

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    # def forward(self, _cat_seq, _val_seq):
    #     out_cat_seq, out_val_seq = [], []

    #     hidden = self.init_hidden_state(_val_seq.shape[1])
    #     masks = self.dropout_mask(_val_seq.shape[1])

    #     cat_seq = _cat_seq.copy()
    #     val_seq = _val_seq.copy()

    #     for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
    #         o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
    #                                             masks)

    #         out_cat_seq.append(o_cat)
    #         out_val_seq.append(o_val)

    #         # fill in the missing features of the next timepoint
    #         idx = np.isnan(val_seq[j])
    #         val_seq[j][idx] = o_val.data.cpu().numpy()[idx]

    #         idx = np.isnan(cat_seq[j])
    #         cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

    #     return torch.stack(out_cat_seq), torch.stack(out_val_seq)
    
    def forward(self, cat_seq, val_seq):
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(val_seq.shape[1])
        masks = self.dropout_mask(val_seq.shape[1])

        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = torch.isnan(val_seq[j])
            val_seq[j][idx] = o_val[idx]  # directly modify the tensor without copy

            idx = torch.isnan(cat_seq[j])
            cat_seq[j][idx] = o_cat[idx]  # directly modify the tensor without copy

        return torch.stack(out_cat_seq), torch.stack(out_val_seq)

    
    
    
    # def forward(self, _cat_seq, _val_seq):
    #     out_cat_seq, out_val_seq = [], []

    #     hidden = self.init_hidden_state(_val_seq.shape[1])
    #     masks = self.dropout_mask(_val_seq.shape[1])

    #     for i, j in zip(range(len(_val_seq)), range(1, len(_val_seq))):
    #         o_cat, o_val, hidden = self.predict(_cat_seq[i], _val_seq[i], hidden,
    #                                             masks)

    #         out_cat_seq.append(o_cat)
    #         out_val_seq.append(o_val)

    #         # fill in the missing features of the next timepoint
    #         idx = torch.isnan(_val_seq[j])
    #         _val_seq[j][idx] = o_val.data.cpu().numpy()[idx]

    #         idx = torch.isnan(_cat_seq[j])
    #         _cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

    #     return torch.stack(out_cat_seq), torch.stack(out_val_seq)

    # def forward(self, _cat_seq, _val_seq):
    #     if isinstance(_val_seq, np.ndarray):  # Handle numpy arrays
    #         out_cat_seq, out_val_seq = [], []

    #         hidden = self.init_hidden_state(_val_seq.shape[1])
    #         masks = self.dropout_mask(_val_seq.shape[1])

    #         cat_seq = _cat_seq.copy()
            
    #         val_seq = _val_seq.copy()

    #         for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
    #             o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
    #                                                 masks)

    #             out_cat_seq.append(o_cat)
    #             out_val_seq.append(o_val)

    #             # Fill in the missing features of the next timepoint
    #             idx = np.isnan(val_seq[j])
    #             val_seq[j][idx] = o_val.data.cpu().numpy()[idx]

    #             idx = np.isnan(cat_seq[j])
    #             cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

    #         return torch.stack(out_cat_seq), torch.stack(out_val_seq)
    #     elif isinstance(_val_seq, torch.Tensor):  # Handle tensors
    #         out_cat_seq, out_val_seq = [], []

    #         hidden = self.init_hidden_state(_val_seq.shape[1])
    #         masks = self.dropout_mask(_val_seq.shape[1])

    #         for i, j in zip(range(len(_val_seq)), range(1, len(_val_seq))):
    #             o_cat, o_val, hidden = self.predict(_cat_seq[i], _val_seq[i], hidden,
    #                                                 masks)

    #             out_cat_seq.append(o_cat)
    #             out_val_seq.append(o_val)

    #             # Fill in the missing features of the next timepoint
    #             idx = torch.isnan(_val_seq[j])
    #             _val_seq[j][idx] = o_val[idx]  # Directly modify the tensor without copy

    #             idx = torch.isnan(_cat_seq[j])
    #             _cat_seq[j][idx] = o_cat[idx]  # Directly modify the tensor without copy

    #         return torch.stack(out_cat_seq), torch.stack(out_val_seq)
    #     else:
    #         raise TypeError("Input type not supported. Please provide either numpy arrays or tensors.")


class SingleStateRNN(RnnModelInterp):
    """
    Base class for RNN model with 1 hidden state (e.g. MinimalRNN)
    (in contrast LSTM has 2 hidden states: c and h)
    """

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = torch.cat([hid[0].new(i_cat), hid[0].new(i_val) * i_mask],
                        dim=-1)

        next_hid = []
        for cell, prev_h, mask in zip(self.cells, hid, r_mask):
            h_t = cell(h_t, prev_h * mask)
            next_hid.append(h_t)

        o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
        o_val = self.hid2measures(h_t) + hid[0].new(i_val)

        return o_cat, o_val, next_hid


class MinimalRNN(SingleStateRNN):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super(MinimalRNN, self).__init__(MinimalRNNCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(cell.bias_hh)


class LSS(SingleStateRNN):
    ''' Linear State-Space '''

    def __init__(self, **kwargs):
        super(LSS, self).__init__(LssCell, **kwargs)


class LSTM(RnnModelInterp):
    ''' LSTM '''

    def __init__(self, **kwargs):
        super(LSTM, self).__init__(nn.LSTMCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(
                cell.bias_hh[cell.hidden_size:cell.hidden_size * 2])

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            h_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            c_x = torch.zeros(batch_size, cell.hidden_size, device=dev)
            state.append((h_x, c_x))
        return state

    def predict(self, i_cat, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = torch.cat([hid[0][0].new(i_cat), hid[0][0].new(i_val) * i_mask],
                        dim=-1)

        states = []
        for cell, prev_state, mask in zip(self.cells, hid, r_mask):
            h_t, c_t = cell(h_t, (prev_state[0] * mask, prev_state[1]))
            states.append((h_t, c_t))

        o_cat = nn.functional.softmax(self.hid2category(h_t), dim=-1)
        o_val = self.hid2measures(h_t) + h_t.new(i_val)

        return o_cat, o_val, states


MODEL_DICT = {'LSTM': LSTM, 'MinRNN': MinimalRNN, 'LSS': LSS}
   