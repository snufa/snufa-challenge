import numpy as np


def h5labels_to_array(datafile):
    labels_array = np.zeros(datafile['spikes']['units'].shape[0], dtype=object)

    # Unlike SNUFA100, here, datafile["labels"] is a HDF5 group
    for dset in datafile["labels"].keys():
        # Each dset is a HDF5 dataset corresponding to the sentence ID and it contains an array of shape (x,
        # 3) where x is the total number of keyword hits. The array has 3 columns which correspond to the keyword
        # label, start time of keyword hit and end time of keyword hit. Each row of the array contains the
        # information for one keyword hit.

        ds_data = datafile["labels"][dset][:]

        a1 = np.empty((ds_data.shape[0],), dtype=object)
        a1[:] = [tuple(row) for row in ds_data]
        labels_array[int(dset)] = a1

    # We return a list that contains arrays of 3-tuples.
    # Each index of the list corresponds to the sentence ID.
    return labels_array
