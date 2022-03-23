import numpy as np
import pandas as pd
from vae_ld.learning_dynamics.procrustes import Procrustes
from vae_ld.learning_dynamics.cka import CKA
import matplotlib.pyplot as plt


def matrix_gen(shape, seed=0):
    np.random.seed(seed)
    X = np.random.random(shape[::-1]).T
    return X

def almost_sim(shape, shape2, seed = 0, seed2 = 1):
    np.random.seed(seed)
    X1 = np.random.random(shape[::-1])
    np.random.seed(seed2)
    X2 = np.random.random(shape2[::-1])
    X = np.concatenate((X1, X2), axis=0).T
    return X

def test_sim():
    mats = []
    dims = list(range(50, 1000, 40))
    p = Procrustes()
    c = CKA()
    for n in dims:
        mats.append((matrix_gen((n, 50)), matrix_gen((n, 50), seed=42), almost_sim((n, 40), (n, 10)),
                     almost_sim((n, 25), (n, 25)), matrix_gen((n, 5))))
    res = {"n": [], "Procrustes": [], "CKA": []}
    for i, n in enumerate(dims):
        res["n"].append(n)
        C = p.center(mats[i][0])
        D = p.center(mats[i][1])
        res["Procrustes"].append(p(C, D))
        res["CKA"].append(c(C.dot(C.T), D.dot(D.T)))
    df = pd.DataFrame.from_dict(res)
    df.set_index("n", inplace=True)
    fig = df.plot.line(ylim=(0, 1)).get_figure()
    plt.tight_layout()
    fig.savefig("different_ab.pdf")

    res = {"n": [], "Procrustes": [], "CKA": []}
    for i, n in enumerate(dims):
        res["n"].append(n)
        C = p.center(mats[i][0])
        D = p.center(mats[i][2])
        res["Procrustes"].append(p(C, D))
        res["CKA"].append(c(C.dot(C.T), D.dot(D.T)))
    df = pd.DataFrame.from_dict(res)
    df.set_index("n", inplace=True)
    fig = df.plot.line(ylim=(0, 1)).get_figure()
    plt.tight_layout()
    fig.savefig('similar_ab.pdf')

    res = {"n": [], "Procrustes": [], "CKA": []}
    for i, n in enumerate(dims):
        res["n"].append(n)
        C = p.center(mats[i][0])
        D = p.center(mats[i][3])
        res["Procrustes"].append(p(C, D))
        res["CKA"].append(c(C.dot(C.T), D.dot(D.T)))
    df = pd.DataFrame.from_dict(res)
    df.set_index("n", inplace=True)
    fig = df.plot.line(ylim=(0, 1)).get_figure()
    plt.tight_layout()
    fig.savefig('mid_similar_ab.pdf')

    res = {"n": [], "Procrustes": [], "CKA": []}
    for i, n in enumerate(dims):
        res["n"].append(n)
        C = p.center(mats[i][0])
        D = p.center(mats[i][4])
        res["Procrustes"].append(p(C, D))
        res["CKA"].append(c(C.dot(C.T), D.dot(D.T)))
    df = pd.DataFrame.from_dict(res)
    df.set_index("n", inplace=True)
    fig = df.plot.line(ylim=(0, 1)).get_figure()
    plt.tight_layout()
    fig.savefig('dif_dim_ab.pdf')


if __name__ == "__main__":
    test_sim()

