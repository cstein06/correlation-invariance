from pylab import *
import scipy.stats as st
import torch 
from torch import optim
from matplotlib import animation, rc
from tqdm import tqdm
rc('animation', html='html5')

set_printoptions(precision=4)

def run_multiple(xs, ny = 9, eta_adam = 0.02, w0 = None, 
                 mb = 100, nt = 100000, rule = 'bcm', hyk = 1., theta = 1., 
                 w0norm = 1., wrec_fac = 0.6, w_decay = 0.,
                 tau_rec = 3, t_rec = 10, fi = 'relu', with_rec = True,
                 fi_thres = 0., w_rec = -0.5, eta_rec = 0.1, 
                 pos_w = False, fix_norm = None, hdw_offset = 0.,
                permute=True, w0_rec=None, 
                hy_exp = 2, tau_h = 2000., print_log=False): 

  nd,nx = xs.shape

  nb = nt//mb
  perm = permutation(nd)

  ws = zeros([nb,nx,ny])
  if w0 is None:
      ws[0] = w0norm*st.norm.rvs(size=[1,nx,ny])/sqrt(nx)
  else:
    if w0.ndim == 1:
      ws[0,:] = w0.reshape(-1,1)
    else:
      ws[0] = w0

  ys = zeros([nb, ny])
  us = zeros([nb, ny])

  hy = zeros([nb,ny])
  hy[0] = hyk*ones(ny)
  hym = zeros([nb,ny])
  hym[0] = ones(ny)
  net_dw = zeros([nb,ny])
  hdw = zeros([nb,ny])
  hdw[0] = 2*ones(ny)
  sig_dw = zeros([nb,ny])
  sig_dw[0] = ones(ny)
  norm_dw = zeros([nb,ny])
  norm_dw[0] = ones(ny)

  if rule == 'bcm' or rule == 'bcm+wd' or \
      rule == 'bcm+hetero' or rule == 'bcm+hdw' or \
      rule == 'ltp+wd':
      fy = lambda y: (y>0)*y**2
  elif rule == 'bcm_kurt':
      fy = lambda y: (y>0)*y**3

  if fi == 'lin':
      fif = lambda u: u
  elif fi == 'relu':
      fif = lambda u,a=fi_thres: (u-a)*(u>a)

  beta2 = 0.9
  tw = torch.tensor(ws[0], requires_grad=True)
  optimizer = optim.Adam([tw], lr=eta_adam, betas=(0.9, beta2))

  tau_h /= mb
  
  if w0_rec is None:
    wrec = w_rec*ones([ny,ny])
    wrec[diag_indices(ny)] = 0
  else:
    wrec = w0_rec
  
  tw_rec = torch.tensor(-wrec, requires_grad=True)
  optimizer_rec = optim.Adam([tw_rec], lr=eta_rec, betas=(0.9, beta2))

  for t in tqdm(arange(1,nb)):

    if (t*mb)%nd==0:
      perm = permutation(nd)

    ix = (t*mb)%nd
    if permute:
      xb = xs[perm[ix:ix+mb]]
    else:
      xb = xs[ix:ix+mb]

    u0 = xb @ ws[t-1]
    
    if (not with_rec) or t_rec <= 1:
      yt = fif(u0)
    else:
      ut = u0.copy()
      yt = fif(u0)
      
      for rt in arange(t_rec):
        uf = u0 + yt @ wrec
        ut += (uf - ut)/tau_rec
        yt = fif(ut)

    if rule == 'bcm' or rule == 'thres' or rule == 'bcm_kurt':
        #print(yt.reshape(-1,ny).shape)
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) - \
            xb.reshape(-1,nx,1)*yt.reshape(-1,1,ny)*(hy[t-1]/hyk) # bcm

    elif rule == 'hetero':
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) - \
             ws[t-1].reshape(1,nx,ny)* \
             yt.reshape(-1,1,ny)**4/(hyk**2)  # not inv.

    elif rule == 'oja':
        dw = xb.reshape(-1,nx,1)*yt.reshape(-1,1,ny) - \
             ws[t-1].reshape(1,nx,ny)* \
             yt.reshape(-1,1,ny)**2/hyk  # not inv.

    elif rule == 'bcm+wd':
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) - \
            xb.reshape(-1,nx,1)*yt.reshape(-1,1,ny)*(hy[t-1]/hyk) - \
            w_decay*ws[t-1].reshape(1,nx,ny)
    
    elif rule == 'bcm+hetero':
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) - \
             hyk*xb.reshape(-1,nx,1)*yt.reshape(-1,1,ny) - \
             w_decay*ws[t-1].reshape(1,nx,ny)* \
             yt.reshape(-1,1,ny)**4  
        
    elif rule == 'bcm+hdw':
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) \
             - hdw[t-1]*xb.reshape(-1,nx,1)*yt.reshape(-1,1,ny) \
            - w_decay*ws[t-1].reshape(1,nx,ny)

    elif rule == 'ltp+wd':
        dw = xb.reshape(-1,nx,1)*fy(yt).reshape(-1,1,ny) \
            - w_decay*ws[t-1].reshape(1,nx,ny)*yt.reshape(-1,1,ny)**2

    dw = dw.sum(axis=0)

    hy[t] = hy[t-1]*(1-1./tau_h) + mean(yt**hy_exp,axis=0)/tau_h
    
    hym[t] = hym[t-1]*(1-1./tau_h) + mean(abs(yt),axis=0)/tau_h
    
    sig_dw[t] = sig_dw[t-1]*(1-1/tau_h) + mean((dw/mb)**2,axis=0)*1/tau_h
    norm_dw[t] = norm_dw[t-1]*(1-1/tau_h) + norm(dw/mb,axis=0)*0.2/tau_h
    net_dw[t] = net_dw[t-1]*(1-1/tau_h) + (ws[t-1]*dw/mb).mean(axis=0)*1/tau_h

    hdw[t] = hdw[t-1] + 0.04*(net_dw[t]*nx/norm_dw[t] - hdw_offset)**3

    hdw[t][hdw[t]<0] = 0 
    
    tw.grad = torch.tensor(-dw)
    optimizer.step()

    if (t%5000 == 101 or t < 1) and print_log:
      print(t)
      
      print('dws', dw[:3])
      print('sig', sig_dw[t])
      print('norm dw', norm_dw[t]/nx)
      print('netdw', net_dw[t])
      print('hdw', hdw[t])
      print(norm(dw,axis=0))
      print(norm(ws[t-1],axis=0))
    
    if pos_w:
      tw.data.clamp_(0)
      
    if fix_norm is not None:
      tw.data = fix_norm*tw.data/tw.data.norm(dim=0)

    ys[t] = yt[0]
    us[t] = u0[0] 
    ws[t] = tw.data
    
    if with_rec:
      
      dw_rec = (yt.reshape(-1,ny,1))*(yt.reshape(-1,1,ny)-theta) -  wrec_fac*abs(wrec).reshape(1,ny,ny)

      dw_rec = dw_rec.sum(axis=0)

      tw_rec.grad = torch.tensor(-dw_rec)
      optimizer_rec.step()
      for i in arange(ny):
        tw_rec.data[i,i] = 0.
      tw_rec.data.clamp_(0)

      wrec[:] = -tw_rec.data

  return ws, ys, wrec, hdw, us
  
n_sign = 20
n_netw = 20
n_back = 20

sig_sign = 2.
sig_netw = 1.2
sig_back = 1.
sign_noise_std = np.array([0.1])

def sign_data(nt, tau_on = 100, tau_decay = 30,
              scale_decay = 0.3, tau_poisson = 1000):
    global mean_data

    data = np.zeros([n_sign,nt])
    
    on_shape = np.ones(tau_on) + scale_decay* \
                      np.exp(-np.linspace(0,tau_on/tau_decay,tau_on))
    
    times = np.int32(np.random.exponential(scale=tau_poisson,size=2*nt//tau_poisson))
    current = 0
    for tt in times:
        current += tt
        if current+100>=nt:
            break
        data[:,current:current+tau_on] = on_shape
        current += tau_on
    
    mean_data = data
    
    data = data + sign_noise_std.reshape(-1,1)*np.random.randn(n_sign,nt)
    
    data = data*np.array(sig_sign).reshape(-1,1)
    
    return data
    
def OU(nt,tau,nx=1):
    # O.U. data
    x = np.zeros([nx,nt])
    for t in range(0,nt-1):
        x[:,t+1] = x[:,t] + -x[:,t] / tau + \
            (1/np.sqrt(tau/2)) * np.random.randn(nx)
    return x
    
def netw_data(nt, OU_tau = 200, noise = 0.2):
    global mean_net
    x = sig_netw*OU(nt,OU_tau,1).repeat(n_netw, axis=0)
    mean_net = x
    x += noise*np.random.randn(*x.shape)
    return x
    
def back_data(nt):
    return np.random.randn(n_netw,nt)*sig_back
    
def make_data(nt):
    signd = sign_data(nt)
    netwd = netw_data(nt)
    backd = back_data(nt)
    return np.concatenate([signd,netwd,backd],axis=0)
  


  