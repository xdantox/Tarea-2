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
    'Edo a integrar'
    output = np.array([y[1], -g / R * np.sin(y[0]) - gamma * y[1]])
    return output


def K1(func, dt, t_n, y_n):
    """Constante K1 del algoritmo de Runge Kutta 4
    """
    output = dt * func(t_n, y_n)
    return output


def K2(func, dt, t_n, y_n):
    """Constante K2 del algoritmo de Runge Kutta 4
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    return k2_n

def K3(func, dt, t_n, y_n):
    """Constante K3 del algoritmo de Runge Kutta 4
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    k3_n = dt * func(t_n + dt/2, y_n + k2_n / 2)
    return k3_n


def K4(func, dt, t_n, y_n):
    """Constante K4 del algoritmo de Runge Kutta 4
    """
    k1_n = K1(func, dt, t_n, y_n)
    k2_n = dt * func(t_n + dt/2, y_n + k1_n / 2)
    k3_n = dt * func(t_n + dt/2, y_n + k2_n / 2)
    k4_n = dt * func(t_n + dt, y_n + k3_n)
    return k4_n



def paso_rk4(func, dt, t_n, y_n):
    'Algoritmo de 1 paso RK4'
    k1=K1(func, dt, t_n, y_n)
    k2=K2(func, dt, t_n, y_n)
    k3=K3(func, dt, t_n, y_n)
    k4=K4(func, dt, t_n, y_n)
    output = y_n + (1/6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return output

# Resolución
'Paso temporal'

dt = 0.01

'Periodo de integracion'

T = 2 * np.pi 

'Intervalo de integración'

t_eval_rk4 = np.arange(0, 2 * T, dt)

'Definición de la forma de la solución'

y_rk4 = np.zeros((len(t_eval_rk4), 2))
y_rk4G = np.zeros((len(t_eval_rk4), 2))

# Condición inicial

'Pequeña'

phi_0 = np.pi/20
omega_0 = 0
y_rk4[0] = [phi_0, omega_0]

'Grande'

phi_0G = np.pi/2.704
omega_0G = 0
y_rk4G[0] = [phi_0G, omega_0G]

'Loop para dar los pasos de RK4 en el intervalo de integración'

for i in range(1, len(t_eval_rk4)):
    y_rk4[i] = paso_rk4(func_pendulo, dt, t_eval_rk4[i-1], y_rk4[i-1])

for i in range(1, len(t_eval_rk4)):
    y_rk4G[i] = paso_rk4(func_pendulo, dt, t_eval_rk4[i-1], y_rk4G[i-1])

# Solución Analítica

'Condición inicial pequeña'

def peqosci(t):
    det = np.sqrt((gamma) ** 2 - 4 * (g / R))
    lambda1 = (-gamma - det) / 2
    lambda2 = (-gamma + det) / 2
    A1 = phi_0 / (1 - (lambda1 / lambda2))
    A2 = -A1 * lambda1 / lambda2
    ypo = A1 * np.exp(lambda1 * t) + A2 * np.exp(lambda2 * t)
    return ypo

'Condición inicial grande'

def peqosciG(t):
    det = np.sqrt((gamma) ** 2 - 4 * (g / R))
    lambda1 = (-gamma - det) / 2
    lambda2 = (-gamma + det) / 2
    A1 = phi_0G / (1 - (lambda1 / lambda2))
    A2 = -A1 * lambda1 / lambda2
    ypo = A1 * np.exp(lambda1 * t) + A2 * np.exp(lambda2 * t)
    return ypo

'Derivada solución analítica grandes oscilaciones:'

def DpeqosciG(t):
    det = np.sqrt((gamma) ** 2 - 4 * (g / R))
    lambda1 = (-gamma - det) / 2
    lambda2 = (-gamma + det) / 2
    A1 = phi_0G / (1 - (lambda1 / lambda2))
    A2 = -A1 * lambda1 / lambda2
    ypo = A1 * lambda1 * np.exp(lambda1 * t) + A2 * lambda2 * np.exp(lambda2 * t)
    return ypo


# Gráficos:

'Condición inicial pequeña'
plt.figure(1)
plt.clf()
plt.plot(t_eval_rk4,y_rk4[:, 0], label='rk4')
plt.plot(t_eval_rk4,peqosci(t_eval_rk4), label='pequeñas oscilaciones')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('Comparación condicion inicial pequeña')
plt.legend()
plt.show()

'Condición inicial grande'

plt.figure(2)
plt.clf()
plt.plot(t_eval_rk4,y_rk4G[:, 0], label='rk4 grandes oscilaciones')
plt.plot(t_eval_rk4,peqosciG(t_eval_rk4), label='pequeñas oscilaciones, condicion grande')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('Comparación condicion inicial grande')
plt.legend()
plt.show()

'Diagrama de fasese para análisis energético'

plt.figure(3)
plt.clf()
plt.plot(y_rk4G[:, 0],y_rk4G[:, 1], label='rk4 grandes oscilaciones')
plt.plot(peqosciG(t_eval_rk4),DpeqosciG(t_eval_rk4), label='pequeñas oscilaciones, condicion grande')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('Diagrama de fase')
plt.legend()
plt.show()