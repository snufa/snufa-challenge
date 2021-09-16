import os
from tqdm.autonotebook import tqdm
import torch
import numpy as np
import h5py


def create_submission_file(model, datafile, basepath):
    """
    Create a sample submission file from the data.
    Expected format:
        |-spikes
            |-hidden_layer1[]
            |-hidden_layer2[]
            |-hidden_layer3[]
        |-labels[]
    """
    hid_spks = []
    labels = []

    # predict
    with tqdm(model.sparse_datagen_from_hdf5(datafile, shuffle=False)) as tepoch:
        for x_local, y_local in tepoch:
            output, recs = model.run_snn(x_local.to_dense())
            _, spks = recs
            hid_spks.append(spks.detach().cpu().numpy())

            m, _ = torch.max(output, 1)
            _, am = torch.max(m, 1)  # argmax over output units
            labels.append(am.detach().cpu().numpy().flatten())

    hid_stacked = np.vstack(hid_spks)
    labels_stacked = np.array(labels).reshape(-1, 1)
    print("Shape of hidden layer spikes and units: ", hid_stacked.shape)
    print("Shape of predicted labels: ", labels_stacked.shape)

    # dump into HDF5 file
    submission_file_path = os.path.join(basepath, 'test_submission.h5')
    with h5py.File(submission_file_path, "w") as hf:
        hf.create_dataset('labels', data=labels_stacked)
        g1 = hf.create_group('spikes')
        g1.create_dataset('hidden1', data=hid_stacked)
        # add all hidden layer spikes using:
        # g1.create_dataset('hidden2',data=..)

    print("You can find the submission file at: ")
    print(submission_file_path)
