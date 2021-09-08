import numpy as np
import os
import torch
import torch.nn as nn
from utils import plot_utils

def setup_env():

    dtype = torch.float

    # Check whether a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    try:
            import google.colab
            IN_COLAB = True
            import tqdm.notebook as tqdm

    except:
            IN_COLAB = False
            import tqdm

    return dtype, device, IN_COLAB

def h5labels_to_array(datafile):

    labels_array = np.zeros([(datafile['spikes']['units'].shape[0])], dtype=object)
    for dset in datafile["labels"].keys() :
        ds_data = datafile["labels"][dset][:]

        a1=np.empty((ds_data.shape[0],), dtype=object)
        a1[:]=[tuple(row) for row in ds_data]
        labels_array[int(dset)] = a1
    
    return labels_array

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

class SpyTrainer_Sentences():
    def __init__(self, challenge, nb_inputs, nb_hidden, nb_outputs, time_step=1e-3, nb_steps=100, max_time=1.4, batch_size=32, seed=None):
        
        self.nb_inputs  = nb_inputs
        self.nb_hidden  = nb_hidden
        self.nb_outputs = nb_outputs
        self.time_step = time_step
        self.nb_steps = nb_steps
        self.max_time = max_time
        self.batch_size = batch_size
        self.tau_mem = 10e-3
        self.tau_syn = 5e-3
        self.challenge = challenge
        self.ignore_index = 100
        self.seed = seed

        if seed is not None:
                np.random.seed(seed)

        self.alpha = float(np.exp(-self.time_step/self.tau_syn))
        self.beta = float(np.exp(-self.time_step/self.tau_mem))

        self.weight_scale = 0.2

        self.w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w1, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))

        self.w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w2, mean=0.0, std=self.weight_scale/np.sqrt(nb_hidden))


        self.v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.v1, mean=0.0, std=self.weight_scale/np.sqrt(nb_hidden))

        print("init done")

    def get_batches(self, datafile, shuffle=True):
        firing_times = datafile['spikes']['times']
        labels_ = h5labels_to_array(datafile)
        list_label_tuples = []

        for sen in range(len(firing_times)):
            sentence_length = firing_times[sen][-1]
            label_ids = [int(_[0]) for _ in labels_[sen]]
            label_times = [(_[1]+_[2])/2 for _ in labels_[sen]]
            split_indices = np.arange(0, sentence_length, self.max_time)

            # (sentence_id, split_start_time, label_id) tuple. label_id=-1 if no label

            for i in range(len(split_indices)-1):
                lexists = False
                for lid, ltime in zip(label_ids, label_times):
                    if split_indices[i] <= ltime < split_indices[i+1]:
                        list_label_tuples.append(tuple([sen, split_indices[i], lid]))
                        lexists = True
                        break
                if not lexists:
                    list_label_tuples.append(tuple([sen, split_indices[i], 100]))

        if shuffle:
            np.random.shuffle(list_label_tuples)

        return list_label_tuples

    def sparse_datagen_from_hdf5(self, datafile, shuffle=True):

        X = datafile['spikes']
        list_label_tuples = self.get_batches(datafile, shuffle=shuffle)

        self.nb_batches = len(list_label_tuples)//self.batch_size

        firing_times = X['times']
        units_fired = X['units']
        time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        for b in range(self.nb_batches):
            samples = list_label_tuples[b*self.batch_size:self.batch_size*(b+1)]
            coo = [ [] for i in range(3)]
            y = []

            for bc, (sen_id, split_start, lid) in enumerate(samples):
                start_idx = np.abs(firing_times[sen_id] - split_start).argmin()
                end_idx = np.abs(firing_times[sen_id] - (split_start+self.max_time)).argmin()
                times = np.digitize(firing_times[sen_id][start_idx:end_idx]-split_start, time_bins)
                units = units_fired[sen_id][start_idx:end_idx]
                batch = [bc for _ in range(len(times))]
                y.append(lid)

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            
            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([self.batch_size, self.nb_steps+1, self.nb_inputs])).to(device)
            y_batch = torch.tensor(y).to(device)

            yield X_batch, y_batch
    
    spike_fn = SurrGradSpike.apply

    def run_snn(self, inputs):
        syn = torch.zeros((self.batch_size,self.nb_hidden), device=device, dtype=dtype)
        mem = torch.zeros((self.batch_size,self.nb_hidden), device=device, dtype=dtype)

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros((self.batch_size, self.nb_hidden), device=device, dtype=dtype)
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))
        for t in range(self.nb_steps):
            self.h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, self.v1))
            mthr = mem-1.0
            out = self.spike_fn(mthr)
            rst = out.detach() # We do not want to backprop through the reset

            new_syn = self.alpha*syn +self.h1
            new_mem =(self.beta*mem +syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)

        # Readout layer
        self.h2= torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros((self.batch_size,self.nb_outputs), device=device, dtype=dtype)
        out = torch.zeros((self.batch_size,self.nb_outputs), device=device, dtype=dtype)
        # out_rec = [out]
        out_rec = []
        for t in range(self.nb_steps):
            new_flt = self.alpha*flt +self.h2[:,t]
            new_out = self.beta*out +flt

            flt = new_flt
            out = new_out
            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs

    def train(self, train_data, test_data, lr=1e-3, nb_epochs=10, checkpointPath=None):
        self.nb_epochs = nb_epochs
        self.lr = lr
        params = [self.w1,self.w2,self.v1]

        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))
        log_softmax_fn = nn.LogSoftmax(dim=1)

        self.class_weights = torch.ones(self.nb_outputs).to(device)
        self.class_weights[self.ignore_index] = 1/10000
        self.class_weights /= torch.sum(self.class_weights)
        
        loss_fn = nn.NLLLoss(weight=self.class_weights)
        print("Training Started")

        self.loss_hist = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.count_out_spikes = []
        self.count_hidden_spikes = []

        for e in range(nb_epochs):
            local_loss = []
            local_acc = []
            out_spks = []
            hid_spks = []
            with tqdm.tqdm(self.sparse_datagen_from_hdf5(train_data),  unit="batch", leave=False) as tepoch:
                for x_local, y_local in tepoch:
                    tepoch.set_description(f"Epoch {e+1}")
                    output,recs = self.run_snn(x_local.to_dense())
                    _,spks=recs
                    out_spks.append(output.detach().cpu().numpy())
                    hid_spks.append(spks.detach().cpu().numpy())

                    m,_=torch.max(output,1)
                    _,am=torch.max(m,1)          # argmax over output units
                    log_p_y = log_softmax_fn(m)
                    # Here we set up our regularizer loss
                    # The strength paramters here are merely a guess and there should be ample room for improvement by
                    # tuning these paramters.
                    reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
                    reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron

                    # Here we combine supervised loss and the regularizer
                    loss_val = loss_fn(log_p_y, y_local) + reg_loss
                    acc_val = np.mean((y_local==am).detach().cpu().numpy())
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
                    local_loss.append(loss_val.item())
                    local_acc.append(acc_val)
                    tepoch.set_postfix(loss=loss_val.item(), accuracy=acc_val)
                    # tepoch.reset(total=self.nb_batches)

            mean_loss = np.mean(local_loss)
            mean_acc = np.mean(local_acc)
            self.loss_hist.append(mean_loss)
            self.train_acc_list.append(mean_acc)

            out_stacked = np.vstack(out_spks)
            hid_stacked = np.vstack(hid_spks)

            self.count_out_spikes.append(np.count_nonzero(out_stacked>0)/out_stacked.shape[0])
            self.count_hidden_spikes.append(np.count_nonzero(hid_stacked)/hid_stacked.shape[0])

            print("Epoch %i: loss=%.5f, accuracy=%.5f"%(e+1,mean_loss, mean_acc))
            if e%5==0 or e==(self.nb_epochs-1):
                if e == self.nb_epochs-1:
                        test_acc = self.compute_classification_accuracy(test_data, plot_confusion_matrix=False)
                else:
                        test_acc = self.compute_classification_accuracy(test_data, plot_confusion_matrix=False)
                print("Test accuracy: %.3f \n"%(test_acc))
                self.test_acc_list.append(test_acc)

                if checkpointPath is not None:
                        print(f"Saving Checkpoint at {checkpointPath}")
                        np.savez_compressed(os.path.join(checkpointPath, f"features_{e}"), w1=self.w1.detach().cpu().numpy(), w2=self.w2.detach().cpu().numpy(),    v1=self.v1.detach().cpu().numpy())
                        print(f"saved{e}")

    def compute_classification_accuracy(self, datafile, plot_confusion_matrix=False):
        """ Computes classification accuracy on supplied data in batches. """
        accs = []
        y_true = []
        y_pred = []
        for x_local, y_local in self.sparse_datagen_from_hdf5(datafile, shuffle=False):
            output,_ = self.run_snn(x_local.to_dense())
            m,_= torch.max(output,1) # max over time
            _,am=torch.max(m,1)          # argmax over output units
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
            y_true.append(y_local.detach().cpu().numpy().flatten())
            y_pred.append(am.detach().cpu().numpy().flatten())

        if plot_confusion_matrix:
                plot_utils.plot_cmatrix(np.ravel(y_true), np.ravel(y_pred))
        return np.mean(accs)

    def get_mini_batch(self, datafile, shuffle=False):
        for ret in self.sparse_datagen_from_hdf5(datafile):
                return ret

class SpyTrainer_SNUFA100():
    def __init__(self, challenge, nb_inputs, nb_hidden, nb_outputs, time_step=1e-3, nb_steps=100, max_time=1.4, batch_size=32, seed=None, dtype=torch.float):

        self.nb_inputs  = nb_inputs
        self.nb_hidden  = nb_hidden
        self.nb_outputs = nb_outputs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.time_step = time_step
        self.nb_steps = nb_steps
        self.max_time = max_time
        self.batch_size = batch_size
        self.tau_mem = 10e-3
        self.tau_syn = 5e-3
        self.challenge = challenge
        self.seed = seed
        self.dtype = dtype

        if seed is not None:
            np.random.seed(seed)

        self.alpha   = float(np.exp(-self.time_step/self.tau_syn))
        self.beta    = float(np.exp(-self.time_step/self.tau_mem))
        self.weight_scale = 0.2

        self.w1 = torch.empty((nb_inputs, nb_hidden),  device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.w1, mean=0.0, std=self.weight_scale/np.sqrt(nb_inputs))
        self.w2 = torch.empty((nb_hidden, nb_outputs), device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.w2, mean=0.0, std=self.weight_scale/np.sqrt(nb_hidden))
        self.v1 = torch.empty((nb_hidden, nb_hidden), device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.v1, mean=0.0, std=self.weight_scale/np.sqrt(nb_hidden))

        print("Initialized the network")

    def sparse_datagen_from_hdf5(self, datafile, shuffle=True):
        X = datafile['spikes']
        y = datafile['labels']

        labels_ = np.array(y, dtype=int)
        self.nb_batches = len(labels_)//self.batch_size
        sample_index = np.arange(len(labels_))

        firing_times = X['times']
        units_fired = X['units']
        time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        if shuffle:
          np.random.shuffle(sample_index)

        for b in range(self.nb_batches):
            samples = sample_index[b*self.batch_size:self.batch_size*(b+1)]

            coo = [ [] for i in range(3)]
            for bc, idx in enumerate(samples):
                times = np.digitize(firing_times[idx], time_bins)
                units = units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)
            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([self.batch_size, self.nb_steps, self.nb_inputs])).to(self.device)
            y_batch = torch.tensor(labels_[samples]).to(self.device)

            yield X_batch, y_batch
      
    spike_fn  = SurrGradSpike.apply

    def run_snn(self, inputs):
        syn = torch.zeros((self.batch_size,self.nb_hidden), device=self.device, dtype=self.dtype)
        mem = torch.zeros((self.batch_size,self.nb_hidden), device=self.device, dtype=self.dtype)

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros((self.batch_size, self.nb_hidden), device=self.device, dtype=self.dtype)
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))
        for t in range(self.nb_steps):
            self.h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, self.v1))
            mthr = mem-1.0
            out = self.spike_fn(mthr)
            rst = out.detach() # We do not want to backprop through the reset

            new_syn = self.alpha*syn +self.h1
            new_mem =(self.beta*mem +syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)

        # Readout layer
        self.h2= torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros((self.batch_size,self.nb_outputs), device=self.device, dtype=self.dtype)
        out = torch.zeros((self.batch_size,self.nb_outputs), device=self.device, dtype=self.dtype)
        # out_rec = [out]
        out_rec = []
        for t in range(self.nb_steps):
            new_flt = self.alpha*flt +self.h2[:,t]
            new_out = self.beta*out +flt

            flt = new_flt
            out = new_out
            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs

    def train(self, train_data, test_data, lr=1e-3, nb_epochs=10, checkpointPath=None):
        self.nb_epochs = nb_epochs
        self.lr = lr
        params = [self.w1,self.w2,self.v1]

        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()
        print("Training Started")

        self.loss_hist = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.count_out_spikes = []
        self.count_hidden_spikes = []

        for e in range(nb_epochs):
            local_loss = []
            local_acc = []
            out_spks = []
            hid_spks = []
            with tqdm.tqdm(self.sparse_datagen_from_hdf5(train_data),  unit="batch", leave=False) as tepoch:
                for x_local, y_local in tepoch:
                    tepoch.set_description(f"Epoch {e+1}")
                    output,recs = self.run_snn(x_local.to_dense())
                    _,spks=recs
                    out_spks.append(output.detach().cpu().numpy())
                    hid_spks.append(spks.detach().cpu().numpy())

                    m,_=torch.max(output,1)
                    _,am=torch.max(m,1)      # argmax over output units
                    log_p_y = log_softmax_fn(m)
                    # Here we set up our regularizer loss
                    # The strength paramters here are merely a guess and there should be ample room for improvement by
                    # tuning these paramters.
                    reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
                    reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron

                    # Here we combine supervised loss and the regularizer
                    loss_val = loss_fn(log_p_y, y_local) + reg_loss
                    acc_val = np.mean((y_local==am).detach().cpu().numpy())

                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
                    local_loss.append(loss_val.item())
                    local_acc.append(acc_val)
                    tepoch.set_postfix(loss=loss_val.item(), accuracy=acc_val)

            mean_loss = np.mean(local_loss)
            mean_acc = np.mean(local_acc)
            self.loss_hist.append(mean_loss)
            self.train_acc_list.append(mean_acc)

            out_stacked = np.vstack(out_spks)
            hid_stacked = np.vstack(hid_spks)

            self.count_out_spikes.append(np.count_nonzero(out_stacked>0)/out_stacked.shape[0])
            self.count_hidden_spikes.append(np.count_nonzero(hid_stacked)/hid_stacked.shape[0])

            print("Epoch %i: loss=%.5f, accuracy=%.5f"%(e+1,mean_loss, mean_acc))
            if e%5==0 or e==(self.nb_epochs-1):
                if e == self.nb_epochs-1:
                    test_acc = self.compute_classification_accuracy(test_data, plot_confusion_matrix=False)
                else:
                    test_acc = self.compute_classification_accuracy(test_data, plot_confusion_matrix=False)
                print("Test accuracy: %.3f \n"%(test_acc))
                self.test_acc_list.append(test_acc)

                if checkpointPath is not None:
                    print(f"Saving Checkpoint at {checkpointPath}")
                    np.savez_compressed(os.path.join(checkpointPath, f"features_{e}"), w1=self.w1.detach().cpu().numpy(), w2=self.w2.detach().cpu().numpy(),  v1=self.v1.detach().cpu().numpy())
                    print(f"saved{e}")

    def compute_classification_accuracy(self, datafile, plot_confusion_matrix=False):
        """ Computes classification accuracy on supplied data in batches. """
        accs = []
        y_true = []
        y_pred = []
        for x_local, y_local in self.sparse_datagen_from_hdf5(datafile, shuffle=False):
            output,_ = self.run_snn(x_local.to_dense())
            m,_= torch.max(output,1) # max over time
            _,am=torch.max(m,1)      # argmax over output units
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
            y_true.append(y_local.detach().cpu().numpy().flatten())
            y_pred.append(am.detach().cpu().numpy().flatten())

        if plot_confusion_matrix:
            plot_utils.plot_cmatrix(np.ravel(y_true), np.ravel(y_pred))
        return np.mean(accs)

    def get_mini_batch(self, datafile, shuffle=False):
        for ret in self.sparse_datagen_from_hdf5(datafile, shuffle=shuffle):
            return ret
