import os
import argparse
import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
import time

DTYPE = tf.float32
BETA = 500.

# Set constant parameters.
# conductances
C_m = 1.0e-9

g_synB = 5e-9
# volatages
V_leak = -40.0e-3  # 40 mV
V_Ca = 100.0e-3  # 100mV
V_k = -80.0e-3  # -80mV
V_h = -20.0e-3  # -20mV
V_syn = -75.0e-3  # -75mV

v_1 = 0.0  # 0mV
v_2 = 20.0e-3  # 20mV
v_3 = 0.0  # 0mV
v_4 = 15.0e-3  # 15mV
v_5 = 78.3e-3  # 78.3mV
v_6 = 10.5e-3  # 10.5mV
v_7 = -42.2e-3  # -42.2mV
v_8 = 87.3e-3  # 87.3mV
v_9 = 5.0e-3  # 5.0mV

v_th = -25.0e-3  # -25mV

# neuron specific conductances
g_Ca_f = 1.9e-2 * (1e-6)  # 1.9e-2 \mu S
g_Ca_h = 1.7e-2 * (1e-6)  # 1.7e-2 \mu S
g_Ca_s = 8.5e-3 * (1e-6)  # 8.5e-3 \mu S

g_k_f = 3.9e-2 * (1e-6)  # 3.9e-2 \mu S
g_k_h = 1.9e-2 * (1e-6)  # 1.9e-2 \mu S
g_k_s = 1.5e-2 * (1e-6)  # 1.5e-2 \mu S

g_h_f = 2.5e-2 * (1e-6)  # 2.5e-2 \mu S
g_h_h = 8.0e-3 * (1e-6)  # 8.0e-3 \mu S
g_h_s = 1.0e-2 * (1e-6)  # 1.0e-2 \mu S

g_Ca = np.array([g_Ca_f, g_Ca_f, g_Ca_h, g_Ca_s, g_Ca_s], dtype=np.float32)
g_k = np.array([g_k_f, g_k_f, g_k_h, g_k_s, g_k_s], dtype=np.float32)
g_h = np.array([g_h_f, g_h_f, g_h_h, g_h_s, g_h_s], dtype=np.float32)

g_leak = 1.0e-4 * (1e-6)  # 1e-4 \mu S

phi_N = 2  # 0.002 ms^-1

# sampling frequency
fft_start = 20
w = 20
# num samples for freq measurement

min_freq = 0.0
max_freq = 1.0
num_freqs = 101
freqs = np.linspace(min_freq, max_freq, num_freqs, dtype=np.float32)[:,None]
dFreq = freqs[1,0] - freqs[0,0]


def get_Phi(dt, T):
    Fs = 1.0 / dt
    N = T - fft_start + 1 - (w - 1)
    ns = np.arange(0, N)
    phis = []
    for i in range(num_freqs):
        k = N * freqs[i] / Fs
        phi = np.cos(2 * np.pi * k * ns / N) - 1j * np.sin(2 * np.pi * k * ns / N)
        phis.append(phi)
    Phi = tf.constant(np.array(phis).T, dtype=tf.complex64)
    return Phi

# [T, K]
avg_filter = (1.0 / w) * tf.ones((w, 1, 1), dtype=DTYPE)
#freqs = np.linspace(min_freq, max_freq, num_freqs, dtype=np.float32)[None,:]
_x_range = np.arange(num_freqs)[None,:]

def Simulate(dt, T, sigma_I):
    Phi = get_Phi(dt, T)
    def simulate(g_el, g_synA):
        g_el = 1e-9*g_el[:,0]
        g_synA = 1e-9*g_synA[:,0]

        # get number of batch samples
        M = g_el.shape[0]
        _zeros = tf.zeros((M,), dtype=DTYPE)


        def f(x):
            # x contains
            V_m = x[:, :5]
            N = x[:, 5:10]
            H = x[:, 10:]

            M_inf = 0.5 * (1.0 + tf.tanh((V_m - v_1) / v_2))
            N_inf = 0.5 * (1.0 + tf.tanh((V_m - v_3) / v_4))
            H_inf = 1.0 / (1.0 + tf.exp((V_m + v_5) / v_6))

            S_inf = 1.0 / (1.0 + tf.exp((v_th - V_m) / v_9))

            I_leak = g_leak * (V_m - V_leak)
            I_Ca = g_Ca * M_inf * (V_m - V_Ca)
            I_k = g_k * N * (V_m - V_k)
            I_h = g_h * H * (V_m - V_h)

            I_elec = tf.stack(
                [
                    _zeros,
                    g_el * (V_m[:, 1] - V_m[:, 2]),
                    g_el * (V_m[:, 2] - V_m[:, 1] + V_m[:, 2] - V_m[:, 4]),
                    _zeros,
                    g_el * (V_m[:, 4] - V_m[:, 2]),
                ],
                axis=1,
            )
            
            I_syn = tf.stack(
                [
                    g_synB * S_inf[:, 1] * (V_m[:, 0] - V_syn),
                    g_synB * S_inf[:, 0] * (V_m[:, 1] - V_syn),
                    g_synA * S_inf[:, 0] * (V_m[:, 2] - V_syn)
                    + g_synA * S_inf[:, 3] * (V_m[:, 2] - V_syn),
                    g_synB * S_inf[:, 4] * (V_m[:, 3] - V_syn),
                    g_synB * S_inf[:, 3] * (V_m[:, 4] - V_syn),
                ],
                axis=1,
            )

            I_total = I_leak + I_Ca + I_k + I_h + I_elec + I_syn

            lambda_N = (phi_N) * tf.math.cosh((V_m - v_3) / (2 * v_4))
            tau_h = (272.0 - (-1499.0 / (1.0 + tf.exp((-V_m + v_7) / v_8)))) / 1000.0

            dVmdt = (1.0 / C_m) * (-I_total + sigma_I*tf.random.normal(I_total.shape, 0., 1.))
            dNdt = lambda_N * (N_inf - N)
            dHdt = (H_inf - H) / tau_h

            dxdt = tf.concat((dVmdt, dNdt, dHdt), axis=1)
            return dxdt

        x0 = tf.constant(
                [
                    -0.04169771,
                    -0.04319491,
                    0.00883992,
                    -0.06879824,
                    0.03048103,
                    0.00151316,
                    0.19784773,
                    0.56514935,
                    0.12214069,
                    0.35290397,
                    0.08614699,
                    0.04938177,
                    0.05568701,
                    0.07007949,
                    0.05790969,
                ],
                dtype=DTYPE
            )


        x0 = tf.tile(x0[None,:], [M, 1])

        x = x0
        vs = [x[:, 2]]
        for i in range(T):
            dxdt = f(x)
            x = x + dxdt * dt
            vs.append(x[:, 2])
            
        x_t = tf.stack(vs, axis=0)
        return x_t

    return simulate

def NetworkFreq(dt, T, sigma_I, mu):
    simulate = Simulate(dt, T, sigma_I)
    Phi = get_Phi(dt, T)
    def network_freq(g_el, g_synA):
        """Simulate the STG circuit given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        x_t = simulate(g_el, g_synA)

        v = tf.transpose(x_t[:,:,None], [1,0,2])[:,fft_start:, :]
        v_rect = tf.nn.relu(v)  # [M5,T-fft,1]
        v_rect_LPF = tf.nn.conv1d(v_rect, avg_filter, stride=1, padding="VALID")[
            :, :, 0
        ]
        v_rect_LPF = v_rect_LPF - tf.expand_dims(tf.reduce_mean(v_rect_LPF, 1), 1)

        V = tf.matmul(tf.cast(v_rect_LPF, tf.complex64), Phi)

        soft_argmax = tf.reduce_sum(tf.nn.softmax(BETA*tf.abs(V), axis=1)*_x_range, axis=1, keepdims=True)
        f_h = soft_argmax*dFreq + min_freq
        T_x = tf.concat((f_h, tf.square(f_h - mu[0])), 1)
        return T_x

    return network_freq

def Simulate_all(dt, T, sigma_I):
    Phi = get_Phi(dt, T)
    def simulate(g_el, g_synA):
        g_el = 1e-9*g_el[:,0]
        g_synA = 1e-9*g_synA[:,0]

        # get number of batch samples
        M = g_el.shape[0]
        _zeros = tf.zeros((M,), dtype=DTYPE)


        def f(x):
            # x contains
            V_m = x[:, :5]
            N = x[:, 5:10]
            H = x[:, 10:]

            M_inf = 0.5 * (1.0 + tf.tanh((V_m - v_1) / v_2))
            N_inf = 0.5 * (1.0 + tf.tanh((V_m - v_3) / v_4))
            H_inf = 1.0 / (1.0 + tf.exp((V_m + v_5) / v_6))

            S_inf = 1.0 / (1.0 + tf.exp((v_th - V_m) / v_9))

            I_leak = g_leak * (V_m - V_leak)
            I_Ca = g_Ca * M_inf * (V_m - V_Ca)
            I_k = g_k * N * (V_m - V_k)
            I_h = g_h * H * (V_m - V_h)

            I_elec = tf.stack(
                [
                    _zeros,
                    g_el * (V_m[:, 1] - V_m[:, 2]),
                    g_el * (V_m[:, 2] - V_m[:, 1] + V_m[:, 2] - V_m[:, 4]),
                    _zeros,
                    g_el * (V_m[:, 4] - V_m[:, 2]),
                ],
                axis=1,
            )
            
            I_syn = tf.stack(
                [
                    g_synB * S_inf[:, 1] * (V_m[:, 0] - V_syn),
                    g_synB * S_inf[:, 0] * (V_m[:, 1] - V_syn),
                    g_synA * S_inf[:, 0] * (V_m[:, 2] - V_syn)
                    + g_synA * S_inf[:, 3] * (V_m[:, 2] - V_syn),
                    g_synB * S_inf[:, 4] * (V_m[:, 3] - V_syn),
                    g_synB * S_inf[:, 3] * (V_m[:, 4] - V_syn),
                ],
                axis=1,
            )

            I_total = I_leak + I_Ca + I_k + I_h + I_elec + I_syn

            lambda_N = (phi_N) * tf.math.cosh((V_m - v_3) / (2 * v_4))
            tau_h = (272.0 - (-1499.0 / (1.0 + tf.exp((-V_m + v_7) / v_8)))) / 1000.0

            dVmdt = (1.0 / C_m) * (-I_total + sigma_I*tf.random.normal(I_total.shape, 0., 1.))
            dNdt = lambda_N * (N_inf - N)
            dHdt = (H_inf - H) / tau_h

            dxdt = tf.concat((dVmdt, dNdt, dHdt), axis=1)
            return dxdt

        x0 = tf.constant(
                [
                    -0.04169771,
                    -0.04319491,
                    0.00883992,
                    -0.06879824,
                    0.03048103,
                    0.00151316,
                    0.19784773,
                    0.56514935,
                    0.12214069,
                    0.35290397,
                    0.08614699,
                    0.04938177,
                    0.05568701,
                    0.07007949,
                    0.05790969,
                ],
                dtype=DTYPE
            )


        x0 = tf.tile(x0[None,:], [M, 1])

        x = x0
        vs = [x]
        for i in range(T):
            dxdt = f(x)
            x = x + dxdt * dt
            vs.append(x)
            
        x_t = tf.concat(vs, axis=0)
        return x_t

    return simulate


def NetworkFreq_all(dt, T, sigma_I, mu):
    simulate = Simulate_all(dt, T, sigma_I)
    Phi = get_Phi(dt, T)
    def network_freq(g_el, g_synA):
        """Simulate the STG circuit given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        x_t = simulate(g_el, g_synA)

        v = tf.transpose(x_t[:,:,None], [1,0,2])[:,fft_start:, :]
        v_rect = tf.nn.relu(v)  # [M5,T-fft,1]
        v_rect_LPF = tf.nn.conv1d(v_rect, avg_filter, stride=1, padding="VALID")[
            :, :, 0
        ]

        v_rect_LPF = v_rect_LPF - tf.expand_dims(tf.reduce_mean(v_rect_LPF, 1), 1)

        V = tf.matmul(tf.cast(v_rect_LPF, tf.complex64), Phi)

        soft_argmax = tf.reduce_sum(tf.nn.softmax(BETA*tf.abs(V), axis=1)*_x_range, axis=1, keepdims=True)
        f_h = soft_argmax*dFreq + min_freq
        T_x = tf.concat((f_h, tf.square(f_h - mu[0])), 1)
        return x_t, T_x

    return network_freq
