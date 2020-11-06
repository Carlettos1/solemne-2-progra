import matplotlib.pyplot as plt
import numpy as np

np.seterr(all='raise')
# mov 2d
L = 50
dt = 0.01
sigma = 9.5
épsilon = 5


def u(r):
    tmp = pow(sigma / r, 6)
    return 4 * épsilon * (tmp**2 - tmp)


def f(r):
    try:
        return 4 * épsilon * (12 * sigma ** 12 / r ** 13 - 6 * sigma ** 6 / r ** 7)
    except:
        print(r)
        exit(2)


def get_acc(posiciones: np.ndarray):
    acc = np.zeros((posiciones.shape[0], 2))
    for i in range(posiciones.shape[0]):
        for j in range(posiciones.shape[0]):
            if i != j:
                d = np.linalg.norm(posiciones[i] - posiciones[j])
                acc[i] += f(d) * (posiciones[i] - posiciones[j]) / d
                pass
            pass
        pass
    return acc  # np.ndarray de las mismas dimensiones que posiciones0


def simulación(tiempos: int, posiciones0: np.ndarray, velocidades: np.ndarray, graficar: bool = True):
    p = np.zeros((tiempos, posiciones0.shape[0], 2))
    p[0] = posiciones0
    p[1] = np.clip(p[0] + velocidades * dt + get_acc(p[0])*dt*dt/2, 0, L)
    for t in range(1, tiempos - 1):
        p[t + 1] = np.clip(2*p[t] - p[t-1] + get_acc(p[t])*dt*dt, 0, L)

    if graficar:
        plt.subplot(1, 1, 1, projection="3d")
        plt.title("Grafico vs tiempo")
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.xlabel("Eje x")
        plt.ylabel("Eje y")
        for i in range(posiciones0.shape[0]):
            plt.plot(p[:, i, 0], p[:, i, 1], range(tiempos), label=f"Particula {i+1}")
        plt.legend()
        plt.show()
    else:
        return p
    pass


def pares():
    pos0 = np.array([[30, 25], [40, 25]], dtype="float64")
    vel0 = np.array([[0, 0], [0, 0]], dtype="float64")

    simulación(10_000, pos0, vel0)
    vel0 = np.array([[0, 0], [2, 0]], dtype="float64")
    simulación(10_000, pos0, vel0)
    vel0 = np.array([[0, 0], [-2, 0]], dtype="float64")
    simulación(10_000, pos0, vel0)
    vel0 = np.array([[0, 0], [-2, 2]], dtype="float64")
    simulación(10_000, pos0, vel0)


def decenas():
    np.random.seed(6365)
    n = 10
    pos0 = np.random.random((n, 2)) * L
    vel0 = np.random.random((n, 2)) * 4
    # (tiempos, partículas, dimensiones)
    p = simulación(10_000, pos0, vel0, False)
    U = np.zeros(p.shape[0])
    K = np.zeros(p.shape[0])
    for k, t in enumerate(p[1:]):
        for i, p1 in enumerate(t):
            for j, p2 in enumerate(t):
                if i != j:
                    U[k] += u(np.linalg.norm(p1 - p2))

    for k, t in enumerate(p[1:-1]):  # t es cada decena de particulas en el k instante de tiempo
        for i, p1 in enumerate(t):  # i es el número de la partícula
            K[k] += (np.linalg.norm(p[k + 1][i] - p[k - 1][i])/2/dt)**2/2
    plt.plot(U[2:-2], label="Energía potencial")
    plt.plot(K[2:-2], label="Energía Cinética")
    plt.plot(K[2:-2] + U[2:-2], label="Energía Total")
    plt.legend()
    plt.show()


decenas()
