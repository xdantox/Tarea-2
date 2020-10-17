from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
'define la función func1 tal que dy/dt=func1 con las 3 edos de primer orden a integrar'
def func1(t,y):
    sigma = 10
    rho = 28
    beta = 8/3
    output = (sigma * (y[1] - y[0]) , y[0] * (rho - y[2]) , y[0] * y[1] - beta * y[2])
    return output
# condiciones iniciales:
y0 = 2.6
x0 = 2.5
z0 = 2.4
dt = 0.1
t_eval = np.arange(0 , 100 , dt)
y_rk4 = np.zeros((len(t_eval),3))
y_rk4[0] = (x0 , y0 , z0)
resolvedor = solve_ivp(func1,t_eval,y_rk4[0])
y_de_t = resolvedor.y
fig = plt.figure()
ax = fig.gca(projection='3d')
X = y_de_t[0]
Y = y_de_t[1]
Z = y_de_t[2]
figura = ax.plot_surface(X, Y, Z)