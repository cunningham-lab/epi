"""SNPE: RNN stable amplification. """
from neural_circuits.LRRNN import get_W_eigs_np
import numpy as np
import os
import pickle
import argparse
import time

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

DTYPE = np.float32

# Get random seed.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--num_sims', type=int, default=1000)
parser.add_argument('--num_batch', type=int, default=50)
parser.add_argument('--num_transforms', type=int, default=3)
parser.add_argument('--num_atoms', type=int, default=10)
parser.add_argument('--g', type=float, default=0.01)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--max_rounds', type=int, default=50)
parser.add_argument('--persist_rounds', type=int, default=2)
parser.add_argument('--min_epochs', type=int, default=10)
parser.add_argument('--stop_distance', type=float, default=None)
parser.add_argument('--rs', type=int, default=1)
args = parser.parse_args()

N = args.N
num_sims = args.num_sims
num_batch = args.num_batch
num_transforms = args.num_transforms
num_atoms = args.num_atoms
g = args.g
K = args.K
max_rounds = args.max_rounds
persist_rounds = args.persist_rounds
min_epochs = args.min_epochs
stop_distance = args.stop_distance
rs = args.rs
torch.manual_seed(rs)

print('\n\nRunning SNPE on RNN conditioned on stable amplification with:')
print("N = %d, \nnum_sims = %d, \nnum_batch = %d, \nnum_transforms = %d, \nnum_atoms = %d, \ng=%.4f, \nK=%d, \nseed=%d\n\n"  \
      % (N, num_sims, num_batch, num_transforms, num_atoms, g, K, rs), flush=True)

base_path = os.path.join("data", "snpe")
save_dir = "SNPE_RNN_stab_amp_N=%d_sims=%d_batch=%d_transforms=%d_atoms=%d_g=%.4f_K=%d_rs=%d" \
        % (N, num_sims, num_batch, num_transforms, num_atoms, g, K, rs)

save_path = os.path.join(base_path, save_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.exists(os.path.join(base_path, save_dir, "optim.pkl")):
    print("SNPE optimization already run. Exitting.")
    exit()

_W_eigs = get_W_eigs_np(g, K)

M = 1000
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
density_estimator_build_fun = posterior_nn(model='maf', hidden_features=50, 
                                           num_transforms=num_transforms,
                                           z_score_x=False, z_score_theta=False,
                                           support_map=True)
x_0 = torch.tensor([0.5, 1.5])

inference = SNPE(prior, density_estimator=density_estimator_build_fun)

best_round = 0

posteriors = []
times = []
proposal = prior
round_val_log_probs = []
zs = [] 
xs = []
log_probs = [] 
distances = []
for r in range(max_rounds):
    time1 = time.time()
    if r == 0:
        print('Round %d/%d:' % (r+1, max_rounds), flush=True)
    else:
        print('Round %d/%d, Best (%d), Avg (%.1f min)):' % (r+1, max_rounds, best_round+1, np.mean(times)/60.), 
               flush=True)
    theta, x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=num_sims)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(
        training_batch_size=num_batch, 
        num_atoms=num_atoms, 
        stop_after_epochs=min_epochs)
    posterior = inference.build_posterior(density_estimator)
    time2 = time.time()
    times.append(time2-time1)
    posteriors.append(posterior)
    round_val_log_probs.append(inference.summary['validation_log_probs'][-1])
    best_round = np.argmax(round_val_log_probs)
    proposal = posterior.set_default_x(x_0)

    z = posterior.sample((M,), x=x_0)
    x = simulator(z).numpy()
    log_prob = posterior.log_prob(z, x=x_0)
    median_x = np.median(x, axis=0)
    distance = np.linalg.norm(median_x - x_0.numpy())

    zs.append(z.numpy())
    xs.append(x)
    log_probs.append(log_prob.numpy())
    distances.append(distance)

    optim = {'summary':inference._summary, 
             'round_val_log_probs':np.array(round_val_log_probs), 
             'zs':np.array(zs), 
             'xs':np.array(xs), 
             'log_probs':np.array(log_probs),
             'distances':np.array(distances),
             'args':args,
             'times':times}

    print('Saving', save_path, '...', flush=True)
    with open(os.path.join(save_path, "optim.pkl"), "wb") as f:
        pickle.dump(optim, f)

    if best_round + persist_rounds == r:
        print("Log prob has converged.", flush=True)
        break
    if stop_distance is not None:
        if distance < stop_distance:
            print("Distance has converged.", flush=True)
            break

print('done.', flush=True)
