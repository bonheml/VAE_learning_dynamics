import numpy as np


def procrustes(X, Y):
    A_sq_frob = np.sum(X ** 2)
    B_sq_frob = np.sum(Y ** 2)
    AB = X.T @ Y
    AB_nuc = np.linalg.norm(AB, ord="nuc")
    return 1 - (A_sq_frob + B_sq_frob - 2 * AB_nuc) / 2


def matrix_gen(shape, seed=0):
    np.random.seed(seed)
    X = np.random.random(shape[::-1]).T
    X -= X.mean(axis=1, keepdims=True)
    X /= np.linalg.norm(X)
    return X


def test_sim():
    A = matrix_gen((1000, 5))
    B = matrix_gen((1000, 100))
    C = matrix_gen((1000, 1000))
    D = matrix_gen((1000, 1000), seed=42)
    print("A is of shape {}, initialised with seed 0.".format(A.shape))
    print("B is of shape {}, initialised with seed 0. Its 5 first features are the same as A.".format(B.shape))
    print("C is of shape {}, initialised with seed 0. "
          "Its 5 first features are the same as A and its first 100 features are the same as B.".format(C.shape))
    print("D is of shape {}, initialised with seed 42. It is different from every other matrices.".format(D.shape))

    print("Procrustes similarity:\nAB={}, AC={}, AD={}, BC={}, BD={}, CD={}".format(
        procrustes(A, B), procrustes(A, C), procrustes(A, D), procrustes(B, C), procrustes(B, D), procrustes(C, D))
    )


if __name__ == "__main__":
    test_sim()
