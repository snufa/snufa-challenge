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
            sample_idx = np.where(labels==label)[0]
        else:
            sample_idx = np.arange(0, len(units_fired), 1)

    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:np.multiply(*dim)]

    fig = plt.figure(figsize=tuple(i*4 for i in dim)[::-1])
    for i,k in enumerate(sample_idx):
        ax = plt.subplot(*dim,i+1)
        ax.scatter(firing_times[k],300-units_fired[k], color="k", alpha=0.33, s=2)
        ax.set_title(f"Label {labels[k]} (id={k})")
    plt.show()

def plot_acc_curves(model):
    """
    Function to accuray, loss and spike_count curves for your model
    model: your model instance
    """
    fig, ax = plt.subplot(3)

    ax[0].plot(np.arange(0, model.nb_epochs, 1), model.train_acc_list, 'k-', label="Train Accuracy")
    ax[0].plot(np.append(np.arange(0, model.nb_epochs, 5), model.nb_epochs-1), model.test_acc_list, 'rx', label="Test accuracy")
    ax[0].set_title("Train and Test Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(np.arange(0, model.nb_epochs, 1), model.loss_hist, 'm', label="Loss")
    ax[1].set_title("Loss Curve")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")

    ax[2].plot(np.arange(0, model.nb_epochs, 1), model.count_hidden_spikes, 'b--', label="hidden layer spikes")
    ax[2].set_title("Average Spike count")
    ax[2].set_ylabel("Spikes per input")
    ax[2].set_xlabel("Epoch")
    plt.show()

def plot_raster_hidden(model, datafile):
    """
    Function to plot spike raster on your model. This function runs a mini batch of your datafile through your model
    model: your model instance
    datafile: HDF5 file
    """
    x_batch, _ = model.get_mini_batch(datafile)
    output, other_recordings = model.run_snn(x_batch.to_dense())
    _ , spk_rec = other_recordings

    spk_rec = spk_rec.detach().cpu().numpy()

    m,_=torch.max(output,1)
    _,y_pred=torch.max(m,1)      # argmax over output units
    y_pred = y_pred.detach().cpu().numpy()

    nb_plt = 5
    gs = gridspec.GridSpec(1,nb_plt)
    fig = plt.figure(figsize=(7,3),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[i].T,cmap=plt.cm.gray_r, origin="lower" )
        plt.title(f"label = {y_pred[i]}\n spikes# = {spk_rec[i].sum()}", fontdict = {'fontsize' : 8})
        if i==0:
            plt.xlabel("Time Step")
            plt.ylabel("Units")
        else:
            plt.yticks([])
        sns.despine()
    plt.show()           # plot_utils.plot_raster_hidden(model)

