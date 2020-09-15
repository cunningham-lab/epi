"""SNPE: RNN stable amplification. """

import numpy as np
import os
import pickle
#import matplotlib.pyplot as plt
import argparse

import delfi
from delfi.simulator.BaseSimulator import BaseSimulator
import delfi.distribution as dd
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.generator as dg
import delfi.inference as infer

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--n_train', type=int, default=2000)
parser.add_argument('--n_mades', type=int, default=5)
parser.add_argument('--n_atoms', type=int, default=100)
parser.add_argument('--rs', type=int, default=1)
args = parser.parse_args()

N = args.N
n_train = args.n_train
n_mades = args.n_mades
n_atoms = args.n_atoms
rs = args.rs

print('Running SNPE on RNN conditioned on stable amplification with:')
print('N = %d, n_train = %d, n_mades = %d, n_atoms = %d, seed=%d' \
      % (N, n_train, n_mades, n_atoms, rs))

def Jeigs(params):
    """Calculates Jeigs.

        Parameters
        ----------
        params : np.array, 1d of length dim_param
            Parameter vector
        seed : int
        """

    U = np.reshape(params[0,:(2*N)], (N,2))
    V = np.reshape(params[0,(2*N):], (N,2))

    J = np.matmul(U, np.transpose(V))
    Js = (J + np.transpose(J)) / 2.
    Js_eigs = np.linalg.eigvalsh(Js)
    Js_eig_max = np.max(Js_eigs, axis=0)

    # Take eig of low rank similar mat
    Jr = np.matmul(np.transpose(V), U) + 0.0001*np.eye(2)
    Jr_tr = np.trace(Jr)
    sqrt_term = np.square(Jr_tr) + -4.*np.linalg.det(Jr)
    J_eig_realmax = 0.5 * Jr_tr
    if (sqrt_term > 0.):
        J_eig_realmax += 0.5*np.sqrt(sqrt_term)

    return np.array([J_eig_realmax, Js_eig_max])

class RNN(BaseSimulator):
    def __init__(self, N):
        """Hodgkin-Huxley simulator

        Parameters
        ----------
        I : array
            Numpy array with the input current
        dt : float
            Timestep
        V0 : float
            Voltage at first time step
        seed : int or None
            If set, randomness across runs is disabled
        """
        self.N = N
        self.r = 2
        dim_param = self.N*self.r*2

        super().__init__(dim_param=dim_param, seed=seed)
        self.Jeigs = Jeigs

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'

        hh_seed = self.gen_newseed()

        states = self.Jeigs(params.reshape(1, -1))

        return {'data': states}

seed_p = 1
prior_min = -np.ones((4*N,))
prior_max = np.ones((4*N,))
prior = dd.Uniform(lower=prior_min, upper=prior_max,seed=seed_p)

class RNNStats(BaseSummaryStats):
    """Moment based SummaryStats class for the Hodgkin-Huxley model

    Calculates summary statistics
    """
    def __init__(self, seed=None):
        """See SummaryStats.py for docstring"""
        super(RNNStats, self).__init__(seed=seed)

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        if len(repetition_list) > 1:
            print(repetition_list)
            raise NotImplementedError()
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            stats.append(x['data'])
        return np.asarray(stats)

seed = 0
# define model, prior, summary statistics and generator classes
m = RNN(N=N)
s = RNNStats()
g = dg.Default(model=m, prior=prior, summary=s)

n_processes = 10

seeds_m = np.arange(1,n_processes+1,1)
m = []
for i in range(n_processes):
    m.append(RNN(N=N))
g = dg.MPGenerator(models=m, prior=prior, summary=s)

# true parameters and respective labels
true_params = np.random.uniform(-1., 1., (4*N,))
labels_params = [r'$T_1$', r'$T_2$']

# observed data: simulation given true parameters
obs = m[0].gen_single(true_params)

obs_stats = np.array([0.5, 1.5])

pilot_samples = 2000

# training schedule
n_rounds = 100

# fitting setup
minibatch = 100
epochs = 500
val_frac = 0.05

# network setup
n_hiddens = [50,50]

# convenience
prior_norm = False

# MAF parameters
density = 'maf'

# inference object
res = infer.SNPEC(g,
                obs=obs_stats,
                n_hiddens=n_hiddens,
                pilot_samples=pilot_samples,
                n_mades=n_mades,
                prior_norm=prior_norm,
                density=density,
                seed=rs)

# train
logs, trn_datasets, posteriors, times = res.run(
                    n_train=n_train,
                    n_rounds=n_rounds,
                    n_atoms=n_atoms,
                    minibatch=minibatch,
                    epochs=epochs,
                    silent_fail=False,
                    proposal='prior',
                    val_frac=val_frac,
                    verbose=True,)

"""def plot_SNPEC_opt(logs, trn_datasets, posteriors):
    losses = [logs[i]['val_loss'] for i in range(len(logs))]
    loss = np.concatenate(losses, axis=0)
    plt.plot(loss,lw=2)
    plt.xlabel('iteration')
    plt.ylabel('val loss')

#plot_SNPEC_opt(logs, trn_datasets, posteriors)
#plt.show()"""
optim = {'logs':logs,
         'trn_datasets':trn_datasets,
         'times':times}
nets = {'posteriors':posteriors}

base_path = os.path.join("data", "snpe")
save_dir = "SNPE_RNN_stab_amp_N=%d_ntrain=%dk_nmades=%d_natoms=%d_rs=%d" \
        % (N, n_train//1000, n_mades, n_atoms, rs)

save_path = os.path.join(base_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('Saving', save_path, '...')
with open(os.path.join(base_path, save_dir, "optim.pkl"), "wb") as f:
    pickle.dump(optim, f)
with open(os.path.join(base_path, save_dir, "networks.pkl"), "wb") as f:
    pickle.dump(nets, f)
print('done.')
