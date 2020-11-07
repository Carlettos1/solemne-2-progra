import random as rnd

import matplotlib.pyplot as plt
import numpy as np

# import ganas.de.vivir as suicidio

σ = 1
μ = 0
tiempos = np.array(range(1, 50 + 1))*100  # tiempos para empezar a medir
k = 100  # numero de muestras
N = 20_000  # pasos totales


def browniano(n, μ, σ, x0=0, y0=0, z0=0):
    x = [x0]
    y = [y0]
    z = [z0]

    for i in range(n):
        dx = rnd.normalvariate(μ, σ)
        dy = rnd.normalvariate(μ, σ)
        dz = rnd.normalvariate(μ, σ)
        x.append(x[i] + dx)
        y.append(y[i] + dy)
        z.append(z[i] + dz)
        pass
    return x, y, z


def get_distancia(graficar=False, eje=111, label=""):
    distancias = []
    X = []
    Y = []
    Z = []

    for j in range(k):  # generar movimiento
        x, y, z = browniano(N, μ, σ)
        X.append(x)
        Y.append(y)
        Z.append(z)
        pass

    if graficar:
        plt.subplot(eje, projection="3d")
        plt.plot(X[0], Y[0], Z[0], label=label)
        plt.legend()

    for n, t in enumerate(tiempos):  # calcular distancias al cuadrado
        promedio = 0
        for x, y, z in zip(X, Y, Z):
            tmp = [x1 ** 2 + y1 ** 2 + z1 ** 2 for x1, y1, z1 in zip(x[t:], y[t:], z[t:])]
            promedio += sum(tmp)/len(tmp)
            pass
        distancias.append(promedio/k)
        pass
    return distancias


def graficar():
    tmp = get_distancia(True, 221, label="Primera partícula")
    plt.subplot(212)
    plt.plot(tmp, label="Primera partícula")
    tmp = get_distancia(True, 222, label="Segunda partícula")
    plt.subplot(212)
    plt.plot(tmp, label="Segunda partícula")
    plt.legend()
    plt.show()


graficar()
σ = 2
graficar()
