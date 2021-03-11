"""SNPE: RNN stable amplification. """
from neural_circuits.LRRNN import get_W_eigs_np
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--num_batch', type=int, default=1000)
parser.add_argument('--num_transforms', type=int, default=3)
parser.add_argument('--num_atoms', type=int, default=10)
parser.add_argument('--g', type=float, default=0.01)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--rs', type=int, default=1)
args = parser.parse_args()

N = args.N
num_batch = args.num_batch
num_transforms = args.num_transforms
num_atoms = args.num_atoms
g = args.g
K = args.K
rs = args.rs
torch.manual_seed(rs)

print('\n\nRunning SNPE on RNN conditioned on stable amplification with:')
print("N = %d, \nnum_batch = %d, \nnum_transforms = %d, \nnum_atoms = %d, \ng=%.4f, \nK=%d, \nseed=%d\n\n"  \
      % (N, num_batch, num_transforms, num_atoms, g, K, rs))

base_path = os.path.join("data", "snpe")
save_dir = "SNPE_RNN_stab_amp_N=%d_batch=%d_transforms=%d_atoms=%d_g=%.4f_K=%d_rs=%d" \
        % (N, num_batch, num_transforms, num_atoms, g, K, rs)

save_path = os.path.join(base_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.exists(os.path.join(base_path, save_dir, "optim.pkl")):
    print("SNPE optimization already run. Exitting.")
    exit()

_W_eigs = get_W_eigs_np(g, K)

N = 2
RANK = 2
num_dim = 2*N*RANK
prior = utils.BoxUniform(low=-1.*torch.ones(num_dim), high=1.*torch.ones(num_dim))

def simulator(params):
    params = params.numpy()
    U = np.reshape(params[:(RANK*N)], (N,RANK))
    V = np.reshape(params[(RANK*N):], (N,RANK))
    x = _W_eigs(U, V)
    return x

simulator, prior = prepare_for_sbi(simulator, prior)
density_estimator_build_fun = posterior_nn(model='maf', hidden_features=50, num_transforms=3,
                                           z_score_x=False, z_score_theta=False,
                                           support_map=True)
x_0 = torch.tensor([0.5, 1.5])

inference = SNPE(prior, density_estimator=density_estimator_build_fun)

max_rounds = 20
persist_rounds = 5
best_round = 0

posteriors = []
proposal = prior
for r in range(max_rounds):
    if r == 0:
        print('Round %d:' % (r+1))
    else:
        print('Round %d, Best (%d):' % (r+1, best_round+1))
    theta, x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=num_batch)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(training_batch_size=50)
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    best_round = np.argmax(inference.summary['best_validation_log_probs'])
    if best_round + persist_rounds == r:
        break
    proposal = posterior.set_default_x(x_0)

zs = []
for posterior in posteriors:
    zs.append(posterior.sample((1000,), x=x_0).numpy())

optim = {'summary':inference._summary, 'zs':np.array(zs)}

base_path = os.path.join("data", "snpe")

print('Saving', save_path, '...')
with open(os.path.join(base_path, save_dir, "optim.pkl"), "wb") as f:
    pickle.dump(optim, f)
print('done.')
