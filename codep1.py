"""
Este script integra la ec. de movimiento de un pendulo real
usando el algoritmo de Runge-Kutta de orden 2.
"""

import numpy as np
import matplotlib.pyplot as plt


g = 9.8  # en m/s^2, aceleracion de gravedad
R = 5.704  # en m, largo del péndulo
gamma=2.704 # en 1/s, coeficiente de fricción


# Implementando solucion usando RK4

def func_pendulo(t, y):
    output = np.array([y[1], -g / R * np.sin(y[0])] - gamma * y[1])
    return output


def K1(func, dt, t_n, y_n):
    """Constante K1 del algoritmo de Runge Kutta 2
    """
    output = dt * func(t_n, y_n)
    return output


def K2(func, dt, t_n, y_n):
    """Constante K2 del algoritmo de Runge Kutta 2
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    return k2_n

def K3(func, dt, t_n, y_n):
    """Constante K3 del algoritmo de Runge Kutta 2
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    k3_n = dt * func(t_n + dt/2, y_n + k2_n / 2)
    return k3_n


def K4(func, dt, t_n, y_n):
    """Constante K4 del algoritmo de Runge Kutta 2
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    k3_n = dt * func(t_n + dt/2, y_n + k2_n / 2)
    k4_n = dt * func(t_n + dt, y_n + k3_n)
    return k4_n



def paso_rk4(func, dt, t_n, y_n):
    output = y_n + 1/6 * (K1(func, dt, t_n, y_n) + 2 * K2(func, dt, t_n, y_n) +
                2 * K3(func, dt, t_n, y_n) + K4(func, dt, t_n, y_n))
    return output


dt = 0.01
T = 2 * np.pi 
t_eval_rk4 = np.arange(0, 2 * T, dt)
y_rk4 = np.zeros((len(t_eval_rk4), 2))

# cond inicial
phi_0 = np.pi/20
omega_0 = 0
y_rk4[0] = [phi_0, omega_0]
for i in range(1, len(t_eval_rk4)):
    y_rk4[i] = paso_rk4(func_pendulo, dt, t_eval_rk4[i-1], y_rk4[i-1])

def peqosci(t):
    det = np.sqrt(gamma ** 2 - 4 * (g / R))
    lambda1 = (-gamma - det) / 2
    lambda2 = (-gamma + det) / 2
    A1 = (phi_0 * lambda1) / (lambda1 - lambda2)
    A2 = (phi_0 * lambda1) / (lambda2 - lambda1)
    ypo = A1 * np.exp(lambda1 * t) + A2 * np.exp(lambda2 * t)
    return ypo

plt.clf()
plt.plot(t_eval_rk4,y_rk4[:, 1], label='rk4')
plt.plot(t_eval_rk4,peqosci(t_eval_rk4), label='peq osc')
plt.legend()
plt.show()