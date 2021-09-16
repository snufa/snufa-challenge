import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_spike_trains(datafile, dim=(1, 4), sample_idx=None, label=None):
    """
    Function to plot samples from the SNUFA100 Dataset
    datafile: HDF5 file
    dim: tuple of ints
    sample_idx: id's of samples you want to plot. If None, we pick random samples.
    label: label_id if you want to plot a specific label. If None, we pick randomly
    """
    firing_times = datafile['spikes']['times'][:]
    units_fired = datafile['spikes']['units'][:]
    labels = datafile['labels'][:]

    if not sample_idx:
        if label:
            sample_idx = np.where(labels == label)[0]
        else:
            sample_idx = np.arange(0, len(units_fired), 1)

    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:np.multiply(*dim)]

    fig = plt.figure(figsize=tuple(i * 4 for i in dim)[::-1])
    for i, k in enumerate(sample_idx):
        ax = plt.subplot(*dim, i + 1)
        ax.scatter(firing_times[k], 300 - units_fired[k], color="k", alpha=0.33, s=2)
        ax.set_title(f"Label {labels[k]} (id={k})")
    plt.show()


def h5labels_to_array(datafile):
    labels_array = np.zeros([(datafile['spikes']['units'].shape[0])], dtype=object)

    # Unlike SNUFA100, here, datafile["labels"] is a HDF5 group
    for dset in datafile["labels"].keys():
        # Each dset is a HDF5 dataset corresponding to the sentence ID
        # and it contains an array of shape (x, 3) where x is the total number of keyword hits.
        # The array has 3 columns which correspond to the keyword label, start time of keyword hit and end time of keyword hit.
        # Each row of the array contains the information for one keyword hit.

        ds_data = datafile["labels"][dset][:]
        a1 = np.empty((ds_data.shape[0],), dtype=object)
        a1[:] = [tuple(row) for row in ds_data]
        labels_array[int(dset)] = a1

    # We return a list that contains arrays of 3-tuples.
    # Each index of the list corresponds to the sentence ID.
    return labels_array


def plot_acc_curves(model):
    """
    Plot accuracy, loss and spike_count curves for your model
    model: instance of SpyTrainer_Sentences class or SpyTrainer_SNUFA100 class
    """

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    x = np.arange(model.nb_epochs)
    ax[0].plot(x, model.loss_hist)
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    y_test = np.array(model.train_acc_list) * 100
    ax[1].plot(x, y_test, label="Train")
    x_test = list(range(0, model.nb_epochs, 5))
    if int(model.nb_epochs - 1) not in x_test:
        x_test.append(model.nb_epochs - 1)
    y_test = np.array(model.test_acc_list) * 100
    ax[1].plot(x_test, y_test, "-rx", label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Percentage")
    ax[1].legend()

    ax[2].plot(x, model.count_hidden_spikes)
    ax[2].set_title("Hidden layer activity")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Avg. no. of spikes")

    sns.despine()

    plt.show()


def plot_acc_vs_hspikes(model):
    plt.plot(model.count_hidden_spikes, model.train_acc_list)
    plt.show()


def plot_raster(model):
    fin_out = model.output[-1]
    print(fin_out.shape)
    nb_plt = 4
    gs = gridspec.GridSpec(1, nb_plt)
    fig = plt.figure(figsize=(7, 3), dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(fin_out[i].T, cmap=plt.cm.gray_r, origin="lower")
        if i == 0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        sns.despine()
    plt.show()


def plot_raster2(model):
    fin_out = model.spike_fn(torch.from_numpy(model.output[-1]))
    # print(fin_out)
    nb_plt = 4
    gs = gridspec.GridSpec(nb_plt, 1)
    plt.figure(figsize=(10, 3))
    for i in range(nb_plt):
        plt.subplot(gs[i])
        # plt.scatter(np.arange(0, model.nb_steps, 1), fin)
        plt.imshow(fin_out[i].T, cmap=plt.cm.gray_r, origin="lower")
        if i == nb_plt - 1:
            plt.xlabel("Time")
            plt.ylabel("Units")

        sns.despine()
    plt.show()


def plot_raster_hidden(model, datafile):
    """
    Plot spike rasters based on the activity of your model by using a minibatch of data from the specified datafile
    model: instance of SpyTrainer_Sentences class or SpyTrainer_SNUFA100 class
    datafile: HDF5 file
    """
    x_batch, _ = model.get_mini_batch(datafile)
    output, other_recordings = model.run_snn(x_batch.to_dense())
    _, spk_rec = other_recordings

    spk_rec = spk_rec.detach().cpu().numpy()

    m, _ = torch.max(output, 1)
    _, y_pred = torch.max(m, 1)  # argmax over output units
    y_pred = y_pred.detach().cpu().numpy()

    nb_plt = 5
    gs = gridspec.GridSpec(1, nb_plt)
    fig = plt.figure(figsize=(7, 3), dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[i].T, cmap=plt.cm.gray_r, origin="lower")
        plt.title(f"{y_pred[i]}\nspikes = {spk_rec[i].sum()}", fontdict={'fontsize': 8})
        if i == 0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        else:
            plt.yticks([])
        sns.despine()
    plt.show()  # plot_utils.plot_raster_hidden(model)
