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

def h5labels_to_array(datafile):
    labels_array = np.zeros([(datafile['spikes']['units'].shape[0])], dtype=object)
    for dset in datafile["labels"].keys() :
        ds_data = datafile["labels"][dset][:]

        a1=np.empty((ds_data.shape[0],), dtype=object)
        a1[:]=[tuple(row) for row in ds_data]
        labels_array[int(dset)] = a1
    
    return labels_array


def plot_acc_curves(model):

    ax1 = plt.figure(figsize=(7,5), dpi=150).gca()
    ax1.plot(np.arange(0, model.nb_epochs, 1), model.train_acc_list, 'k-', label="Train Accuracy")
    ax1.plot(np.arange(0, model.nb_epochs, 1), model.loss_hist, 'm', label="Loss")
    ax1.plot(np.append(np.arange(0, model.nb_epochs, 5), model.nb_epochs-1), model.test_acc_list, 'rx', label="Test accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
  
    ax2 =ax1.twinx()
    ax2.plot(np.arange(0, model.nb_epochs, 1), model.count_out_spikes, 'g--', label="readout spikes")
    ax2.plot(np.arange(0, model.nb_epochs, 1), model.count_hidden_spikes, 'b--', label="hidden layer spikes")
    ax2.plot(np.arange(0, model.nb_epochs, 1), np.asarray(model.count_hidden_spikes)+np.asarray(model.count_out_spikes), 'c--', label="total spikes")
    ax2.set_ylabel("Spikes per input")
    ax2.yaxis.label.set_color('green')
    # ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.legend(loc=4)
    ax2.legend(loc=1)
    plt.show()

def plot_acc_vs_hspikes(model):
    plt.plot(model.count_hidden_spikes, model.train_acc_list)
    plt.show()
  
def plot_raster(model):
    fin_out = model.output[-1]
    print(fin_out.shape)
    nb_plt = 4
    gs = gridspec.GridSpec(1,nb_plt)
    fig = plt.figure(figsize=(7,3),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(fin_out[i].T,cmap=plt.cm.gray_r, origin="lower" )
        if i==0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        sns.despine()
    plt.show()
  
def plot_raster2(model):
    fin_out = model.spike_fn(torch.from_numpy(model.output[-1]))
    # print(fin_out)
    nb_plt = 4
    gs = gridspec.GridSpec(nb_plt,1)
    plt.figure(figsize=(10, 3))
    for i in range(nb_plt):
        plt.subplot(gs[i])
        # plt.scatter(np.arange(0, model.nb_steps, 1), fin)
        plt.imshow(fin_out[i].T,cmap=plt.cm.gray_r, origin="lower")
        if i==nb_plt-1:
            plt.xlabel("Time")
            plt.ylabel("Units")
  
        sns.despine()
    plt.show()

def plot_raster_hidden(model, datafile):
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
        plt.title(f"{y_pred[i]}\nspikes = {spk_rec[i].sum()}", fontdict = {'fontsize' : 8})
        if i==0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        else:
            plt.yticks([])
        sns.despine()
    plt.show()           # plot_utils.plot_raster_hidden(model)

