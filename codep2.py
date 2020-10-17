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

# Condiciones iniciales:

y0 = 1.0
x0 = 1.0
z0 = 1.0

# Condiciones iniciales variadas:

dl = 0.01
y0v = 1.0 + dl
x0v = 1.0 + dl
z0v = 1.0 + dl

#Resolución:
'Intervalo parametro t'

t_eval = np.arange(0 , 40 , 0.01)

'Definición de la forma de las soluciones'

y_rk4 = np.zeros((len(t_eval),3))
y_rk4v = np.zeros((len(t_eval),3))

'Identificación de condiciones iniciales en las soluciones'

y_rk4[0] = (x0 , y0 , z0)
y_rk4v[0] = (x0v , y0v , z0v)

'Algoritmo de resolución'

resolvedor = solve_ivp(func1,[0,40],y_rk4[0],t_eval=t_eval)
resolvedorv = solve_ivp(func1,[0,40],y_rk4v[0],t_eval=t_eval)

'Variable que identifica la solución'

y_de_t = resolvedor.y
y_de_tv = resolvedorv.y

# Gráfico:

'Crear la figura'

fig = plt.figure()
ax = fig.gca(projection='3d')

'Identificación de cada variable ya integrada'

X = y_de_t[0]
Y = y_de_t[1]
Z = y_de_t[2]
Xv = y_de_tv[0]
Yv = y_de_tv[1]
Zv = y_de_tv[2]

'Plotting'

figura = ax.plot(X, Y, Z, label='Sin variación')
figurav = ax.plot(Xv, Yv, Zv, label='Con variación')
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)
plt.legend()
plt.show()

