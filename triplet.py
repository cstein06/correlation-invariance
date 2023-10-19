import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

import sys, os
sys.path.insert(0, os.getcwd() + "/bindsnet/")

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, CIFAR10, DataLoader, NatImages
from bindsnet.encoding import BernoulliEncoder, RepeatEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network.monitors import Monitor
from bindsnet.network import network as bnetwork
from bindsnet.models import EINetwork, EINetwork_twotypes
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)

import wandb

n_workers = -1
gpu = True
seed = 0
progress_interval = 10 
plot = True
imshow_lims = 1.
update_steps = 30 

batch_size = 128 
dt = 1.0

onoff = True
n_neurons = 16
pat_size = 16

update_interval = update_steps * batch_size

# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

np.random.seed(seed)
    
# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

ninputs = pat_size**2

network_type = EINetwork

time = 100
I_fac = 2
intensity = 0.1*I_fac

onofffac = 1+onoff
onoff_FR = [1.,1.]

subtract_input_mean = 1
subt_data_mean = 0

spike_input = True

if spike_input:
  encoder = BernoulliEncoder
else:
  encoder = RepeatEncoder 

data_threshold = 0.

mem_tau = 15.
trace_tau = 30.
noise = 0.0
reset = -65. 

I_tau = 1.

wmin = 0.

target_rate = 0.001
dead_cell_rate = target_rate/5
theta = target_rate*3

i_weight_decay = 3.e-0
wd_fac = None
e_weight_decay = 0.1e-1

inh=0.1
exc=0.02

nu0 = 0.1e-3 
rec_nu0 = 0.1e-3 

beta = 0.99
beta2 = 0.999
use_adam = False

bcm_rule = True

###########
# Triplet figure single
bcm_rule = True
e_weight_decay = 0.05
target_rate = 0.001
theta = 0.003
exc=0.3
nu0 = 0.1e-3 
i_weight_decay = 0.e-0
rec_nu0 = 0.
inh = 0.
#########

###########
# Triplet figure multiple
i_weight_decay = 3.e-0
rec_nu0 = 1.e-3
inh = 0.2
###########

###########
# Oja like
# bcm_rule = False
# e_weight_decay = 0.
# target_rate = 0.005
# theta = 0.01
# exc=0.3
# nu0 = 0.02e-3
# rec_nu0 = 0.2e-3 
# i_weight_decay = 6.e-1
#########

#################
# No rec.
# rec_nu0 = 0.
# inh = 0.
##################
  
rate_tau = 2000.*time/batch_size

n_epochs = 8
quick_run = 0
saving_learned = 0

note = "triplet, recurrent."
print("Run note:", note)

model_folder = "./wandb/"
model_name = "spiking_second_order"

load_pre_trained = 0
no_learning = False
do_sta = False

if not load_pre_trained:
  # Build network.
  network = network_type(
      n_inpt=ninputs,
      n_neurons=n_neurons,
      inpt_shape=(1, pat_size, pat_size),
      dt=dt,

      I_tau = I_tau,
      noise = noise, 
      mem_tau = mem_tau,
      reset = reset,

      inh=inh,
      exc=exc,

      nu0=nu0,
      rec_nu0=rec_nu0,

      weight_decay=e_weight_decay,
      i_weight_decay=i_weight_decay,

      wmin = wmin, #None,
      nu=np.array([1, 1/target_rate]), # multiplying factors for LTP and LTD in triplet.
      theta=theta,

      rate_tau = rate_tau,
      x_trace_tau = trace_tau, 
      y_trace_tau_LTD = trace_tau, 
      y_trace_tau_LTP = trace_tau, 
      dead_cell_rate = dead_cell_rate,
      subtract_input_mean = subtract_input_mean,
    
      beta = beta,
      beta2 = beta2,
      use_adam = use_adam,
      bcm_rule = bcm_rule,
  )
else:
  print("Loading pre-trained model:", load_pre_trained)
  network = bnetwork.load(model_folder+load_pre_trained)
  
config = dict(load_pre_trained = load_pre_trained, no_learning = no_learning,
              do_sta = do_sta,
  I_tau = I_tau, noise = noise, reset = reset, inh = inh, exc = exc,
    subtract_input_mean = subtract_input_mean, subt_data_mean = subt_data_mean,
    nu0 = nu0, rec_nu0 = rec_nu0, wmin = wmin,
    dead_cell_rate = dead_cell_rate, trace_tau = trace_tau,
    wd_fac = wd_fac, i_weight_decay = i_weight_decay, e_weight_decay = e_weight_decay, theta = theta, spike_input = spike_input,
    target_rate = target_rate, mem_tau = mem_tau, data_threshold = data_threshold, rate_tau = rate_tau,
    time = time, intensity = intensity,
    onoff = onoff, n_neurons = n_neurons, pat_size = pat_size,
    n_epochs = n_epochs, batch_size = batch_size, dt = dt,
    n_workers = n_workers, gpu = gpu, seed = seed, plot = plot, imshow_lims = imshow_lims,
      beta = beta, use_adam = use_adam,
      beta2 = beta2,
      bcm_rule = bcm_rule,)

# run = wandb.init(...)

# Directs network to GPU
if gpu:
    print("with gpu")
    network.to("cuda")
else: 
    print("with cpu")

to_onoff_f = lambda x: torch.cat((x*(x>0.)*onoff_FR[0], -x*(x<0.)*onoff_FR[1]), dim=0)
if onoff: 
    to_onoff = transforms.Lambda(to_onoff_f) 
else:
    to_onoff = transforms.Lambda(lambda x: x) 
    
if subt_data_mean: 
    sub_mean = transforms.Normalize(mean=0.33,std=1.)
else:
    sub_mean = transforms.Lambda(lambda x: x) 
    
threshold = transforms.Lambda(lambda x, t=data_threshold: (x>t)*(x-t)) 

# Load nat image data.
all_transforms = transforms.Compose(
                    [transforms.CenterCrop(pat_size), 
                     transforms.ToTensor(), 
                    to_onoff, 
                     threshold,
                    sub_mean,
                    transforms.Lambda(lambda x: x * intensity)])

run_samples = 4

dataset = NatImages(
    image_encoder=encoder(time=time, dt=dt),
    transform=all_transforms,
)

# Create a dataloader to iterate and batch data
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=gpu,
    drop_last=True
)

if no_learning:
  network.train(False)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im, weights_ax = None, None
onoff_im, onoff_ax = None, None
rec_weights_im, rec_weights_ax = None, None
assigns_im = None
perf_ax = None
trace_axes, trace_ims = None, None
norm_axes, norm_ims = None, None
long_axes, long_ims = None, None
long_norm_axes, long_norm_ims = None, None
rate_axes, rate_ims = None, None
voltage_axes, voltage_ims = None, None
voltage_axes2, voltage_ims2 = None, None

plot_time = time*run_samples
print('plot time', plot_time)

spike_layers = ["X", "Y"]
  
if not load_pre_trained:
  
  # Voltage recording for excitatory and inhibitory layers.
  exc_voltage_monitor = Monitor(network.layers["Y"], ["v"], time=plot_time)
  exc_trace_monitor = Monitor(network.layers["Y"], ["x"], time=plot_time)
  exc_rate_monitor = Monitor(network.layers["Y"], ["rate"], time=plot_time)
  network.add_monitor(exc_voltage_monitor, name="exc_voltage")
  network.add_monitor(exc_rate_monitor, name="exc_rate")
  network.add_monitor(exc_trace_monitor, name="exc_trace")

  # Set up monitors for spikes and voltages
  spikes_monitors = {}
  for layer in spike_layers:
      spikes_monitors[layer] = Monitor(network.layers[layer], state_vars=["s"], time=plot_time)
      network.add_monitor(spikes_monitors[layer], name="%s_spikes" % layer)

  voltages = {}
  for layer in ["Y"]:
      voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=plot_time)
      network.add_monitor(voltages[layer], name="%s_voltages" % layer)

  long_rate_mon = Monitor(network.layers["Y"], state_vars=["rate"], time=plot_time, freq=5*time)
  network.add_monitor(long_rate_mon, name="long rate")

  long_norm_mon = Monitor(network.connections[("X","Y")], state_vars=["w_abs_sum"], time=plot_time, freq=5*time)
  network.add_monitor(long_norm_mon, name="long norm")

else:
  exc_voltage_monitor = network.monitors['exc_voltage']
  exc_trace_monitor = network.monitors['exc_trace']
  exc_rate_monitor = network.monitors['exc_rate']
  spikes_monitors = {}
  spikes_monitors["X"] = network.monitors['X_spikes']
  spikes_monitors["Y"] = network.monitors['Y_spikes']
  voltages = {}
  voltages['Y'] = network.monitors['Y_voltages']
  long_rate_mon = network.monitors['long rate']
  long_norm_mon = network.monitors['long norm']
  
# Train the network.
print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, batch in enumerate(tqdm(dataloader)):
        # Get next input sample.

        if quick_run:
          if step > quick_run: # for quick code testing
            break
        
        input_data = batch["encoded_image"]
        if gpu:
            input_data = input_data.cuda()

        inputs = {"X": input_data}

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        long_norm_mon.obj.get_norm()
          
        # Optionally plot various simulation information.
        if (plot and step % update_steps == 0) or (step == len(dataloader)-1): # with step=0 plot

            if step > run_samples:
              spikes_ = {}
              for layer in spike_layers:
                  spikes = spikes_monitors[layer].get("s")
                  spikes = spikes[:, 0]
                  spikes_[layer] = spikes.view(plot_time,-1) 

              spike_ims, spike_axes = plot_spikes(spikes_, 
                                                  ims=spike_ims, axes=spike_axes)

              # Get voltage recording.
              exc_voltages = exc_voltage_monitor.get("v")
              exc_voltages = exc_voltages[:,0] # only first of batch

              exc_traces = exc_trace_monitor.get("x") # [time, ntraces, batchsize, nneurons]
              exc_traces = exc_traces[:,:,0] # only first of batch

              exc_rates = exc_rate_monitor.get("rate") # [time, nneurons]

              voltages = {"Y": exc_voltages}
              traces = {"rate": exc_rates, "LTD": exc_traces[:,0], "LTP": exc_traces[:,1]}
              
              
              voltage_ims, voltage_axes = plot_voltages(
              voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
              )
              
              trace_ims, trace_axes = plot_voltages(
                 traces, ims=trace_ims, axes=trace_axes, plot_type="line"
              )
              
              rate_pl = {"rate": exc_rates[0:1].T}
              
              wandb.log({"voltages": wandb.Image(voltage_axes.get_figure(), caption="voltages"),
                       "rates": wandb.Image(trace_axes[0].get_figure(), caption="rates"),
                      "avg_FRs": exc_rates.mean(),
                      "spikes": wandb.Image(spike_axes[0].get_figure(), caption="spikes"),})
              
              
              if saving_learned:
                ut.mysave(traces,load_pre_trained+"_traces") 
                ut.mysave([spikes_monitors["X"].get("s"),spikes_monitors["Y"].get("s")],load_pre_trained+"spikes")
                ut.mysave(batch["image"],load_pre_trained+"image")
                ut.mysave(batch["encoded_image"],load_pre_trained+"encoded_image")
              
              
            image = batch["image"][:, 0].reshape(pat_size, pat_size) # view?
            inpt = inputs["X"][:, 0].view(time, pat_size**2).sum(0).view(pat_size, pat_size)

            input_exc_weights = network.connections[("X", "Y")].w.detach()

            #print(input_exc_weights.shape)
            square_weights = get_square_weights(
                input_exc_weights.view(1,pat_size**2,-1).permute(1,2,0).reshape(pat_size**2, -1),
                n_sqrt, (pat_size, pat_size), n_sqrt_2=n_sqrt
            )

            inpt_axes, inpt_ims = plot_input(
                image, inpt, #label=labels[step], 
                axes=inpt_axes, ims=inpt_ims
            )
            
            norms = torch.norm(input_exc_weights.view(onofffac*pat_size**2, n_neurons), dim=0).unsqueeze(0).T
            norms_pl = {"norm w": norms}

            if two_type:
              rec_weights = network.connections[("Y", "I")].w @ network.connections[("I", "Y")].w
              
            else:
              rec_weights = network.connections[("Y", "Y")].w

            square_recweights = get_square_weights(
                rec_weights.view(n_neurons, n_neurons), int(np.sqrt(n_neurons)), int(np.sqrt(n_neurons))
            )
            rec_weights_im, rec_weights_ax = plot_weights(square_recweights, im=rec_weights_im,
                                            wmin=-imshow_lims, wmax=0., 
                                            ax=rec_weights_ax)

            weights_im, weights_ax = plot_weights(square_weights, im=weights_im, ax=weights_ax,
                                        figsize=(onofffac*5, 5),
                                        wmin=0, wmax=imshow_lims)

            #print(input_exc_weights.shape)
            if onoff:
                aux_weights = input_exc_weights.reshape(1,onofffac,pat_size**2,-1).permute(1,2,0,3).reshape(onofffac,pat_size**2,-1)
                square_weights = get_square_weights(
                    aux_weights.view(onofffac,pat_size**2,-1)[0],
                    n_sqrt, (pat_size, pat_size),n_sqrt_2=n_sqrt,
                )
                onoff_im, onoff_ax = plot_weights(square_weights, im=onoff_im, figsize=(5, 5),
                                        ax=onoff_ax,
                                            wmin=-imshow_lims, wmax=imshow_lims, alpha=1.)

                square_weights = get_square_weights(
                    aux_weights.view(onofffac,pat_size**2,-1)[1],
                    n_sqrt, (pat_size, pat_size),n_sqrt_2=n_sqrt,
                )
                onoff_im, onoff_ax = plot_weights(-square_weights, im=onoff_im, ax=onoff_ax,
                                        figsize=(5, 5),
                                            wmin=-imshow_lims, wmax=imshow_lims, alpha=0.5)

            wandb.log({"epoch": epoch, "batch": step, 
                       "rfs": wandb.Image(weights_im, caption="rfs"),
                       "rec_weights": wandb.Image(rec_weights_im, caption="rec_weights"),
                       "onoff_rfs": wandb.Image(onoff_im, caption="onoff_rfs"),
                       
                      "avg_rec_weights": rec_weights.mean(),
                      "avg_ff_weights": input_exc_weights.mean(),
                       "exc_w": wandb.Histogram(input_exc_weights.cpu().flatten(), num_bins=20),
                      })   
          
            wandb.log({
              "avg_w_rep0": input_exc_weights.reshape(2,-1)[0].mean(),
              "avg_w_rep1": input_exc_weights.reshape(2,-1)[1].mean(),
            })
                
            plt.pause(1)
            plt.show()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

square_weights = get_square_weights(
    input_exc_weights.view(onofffac,pat_size**2,-1)[1], # off only weights?
    n_sqrt, (pat_size, pat_size),
)
onoff_im, onoff_ax = plot_weights(-square_weights, im=onoff_im, ax=onoff_ax,
                        figsize=(5, 5),
                            wmin=-4., wmax=4., alpha=0.5)
 
images = wandb.Image(square_weights, caption="Final Gabors")
wandb.log({"final_rfs": images})

file_name = model_name + f"_{wandb.run.name}.pt"

artifact = wandb.Artifact(model_name, type="model", description="EI spiking model for second order invariance paper.")

network.save(model_folder+file_name)

artifact.add_file(model_folder+file_name)
