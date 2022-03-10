#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utils in Cython.

Author:
    Erik Johannes Husom

Created:
    2022-03-09 onsdag 14:22:30 

"""
import numpy as np
cimport numpy as np


# def c_split_sequences(
#     sequences,
#     window_size,
#     target_size=1,
#     n_target_columns=1,
#     future_predict=False,
#     overlap=0,
# ):
cpdef c_split_sequences(np.ndarray[np.double_t, ndim=2] sequences, int window_size, int target_size, int n_target_columns, int future_predict, int overlap):
    """Split data sequence into samples with matching input and targets.

    Args:
        sequences (array): The matrix containing the sequences, with the
            targets in the first columns.
        window_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        target_size (int): Size of target window. Default=1, i.e. only one
            value is used as target.
        n_target_columns: Number of target columns. Default=1.
        future_predict (bool): Whether to predict target values backwards or
            forward from the last time step in the input sequence.
            Default=False, which means that the number of target values will be
            counted backwards.
        overlap (int): How many time steps to overlap for each sequence. If
            overlap is greater than window_size, it will be set to
            window_size-1, which is the largest overlap possible.  Default=0,
            which means there will be no overlap.

    Returns:
        X (array): The input samples.
        y (array): The targets.

    """

    # cdef int window_size = window_size
    cdef int start_idx = 0


    # overlap can maximum be one less than window_size
    if overlap >= window_size:
        overlap = window_size - 1

    # Finding number of output sequences:
    length = sequences.shape[0]
    n_features = sequences.shape[1]

    if future_predict > 0:
        dim0 = (length - overlap - target_size) // (window_size - overlap)
    else:
        dim0 = (length - overlap ) // (window_size - overlap)

    print(dim0)

    # X, y = list(), list()
    # cdef list X
    # cdef list y
    cdef np.ndarray[double, ndim=3] X = np.zeros((dim0, window_size, n_features))
    cdef np.ndarray[double, ndim=2] y = np.zeros((dim0, target_size))

    cdef int i = 0
    # for i in range(len(sequences)):
    while start_idx + window_size <= len(sequences):

        # find the end of this pattern
        end_ix = start_idx + window_size

        # find start of target window
        if future_predict > 0:
            target_start_ix = end_ix
            target_end_ix = end_ix + target_size
        else:
            target_start_ix = end_ix - target_size
            target_end_ix = end_ix

        # check if we are beyond the dataset
        # if end_ix > len(sequences):
        if target_end_ix > len(sequences):
            break

        # Select all cols from sequences except target col, which leaves inputs
        cdef np.ndarray[double, ndim=2] seq_x = sequences[start_idx:end_ix, n_target_columns:]
        cdef np.ndarray[double, ndim=2]

        # Extract targets from sequences
        if n_target_columns > 1:
            seq_y = sequences[target_start_ix:target_end_ix, 0:n_target_columns]
            seq_y = seq_y.reshape(-1)
        else:
            seq_y = sequences[target_start_ix:target_end_ix, 0]

        # X.append(seq_x)
        # y.append(seq_y)
        X[i][:][:] = seq_X
        # y[i] = seq_y
        i += 1

        start_idx += window_size - overlap

    # X = np.array(X)
    # y = np.array(y)

    print(X.shape)
    print(y.shape)

    return X, y
