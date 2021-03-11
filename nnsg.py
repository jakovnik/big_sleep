import numpy as np
import numpy.linalg as npla
from yall1 import Solver4L1Minimization as SL1

matlab_eps = 2.220446049250313e-16

def orth(A):
    Q, S, _ = npla.svd(A)
    if np.size(S):
        tol = np.max(A.shape) * S[0] * matlab_eps
        r = np.sum(S > tol)
        Q = Q[:, 0:r]
    return Q


def cla_S(X, M, V, parameter_lambda, parameter_beta):
    X = X.T
    T = orth(X)
    X = T.T
    R = parameter_lambda * M + V
    n = X.shape[1]
    S = np.zeros(shape=(n,n))
    for i in range(n):
        XXX = X.copy()
        RRR = R.copy()
        X_i = XXX[:, i]
        XXX = np.delete(XXX, i, 1)
        RR = RRR[:, i]
        RR = np.delete(RR, i, 0)
        p = RR.copy()
        solver = SL1(tol=5e-3, rho=parameter_beta / 2, nonneg=1, weights=p)
        X_i = X_i.reshape((X_i.shape[0], ))
        xss, _, _, _, _ = solver.get_solution(XXX, X_i)
        for j in range(n):
            if j < i:
                S[i, j] = xss[j]
            elif j > i:
                S[i, j] = xss[j-1]
    return S


def nnsg(label, X, parameter_alpha, parameter_lambda, parameter_beta):
    n_class = len(np.unique(label))
    n = X.shape[1]
    m = len(label)
    A = np.matmul(npla.inv(np.matmul(X, X.T) + 0.01 * np.eye(X.shape[0])), X)
    tMM = np.matmul(X.T, A) - np.eye(n)
    Z = np.matmul(tMM.T, tMM)
    U = np.zeros(shape=(n, n))
    for jj in range(m):
        U[jj, jj] = 1
    Y = np.zeros(shape=(n, n_class))
    for ii in range(m):
        for iii in range(n_class):
            if label[ii] == iii+1:
                Y[ii, iii] = 1
    M = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i+1, n):
            M[i, j] = npla.norm(X[:, i] - X[:, j])**2
            M[j, i] = M[i, j]
    F = 10 * np.ones(shape=(n, n_class))
    L = np.ones(shape=(n, n))
    n_iter = 5
    obj = np.zeros(shape=(n_iter,))
    for it in range(n_iter):
        W = np.matmul(A, F)
        F = np.matmul(npla.inv(U + L + parameter_alpha * Z + 0.01 * np.eye(n)), np.matmul(U, Y))
        V = np.zeros(shape=(n, n))
        for i1 in range(n):
            for j1 in range(i1+1, n):
                V[i1, j1] = 0.5 * npla.norm(F[i1, :] - F[j1, :])**2
                V[j1, i1] = V[i1, j1]
        S = cla_S(X, M, V, parameter_lambda, parameter_beta)
        LLL = S + S.T
        L = np.diag(np.sum(LLL, 1)) - LLL
        tmp = F - Y
        B1 = np.trace(np.matmul(tmp.T, np.matmul(U, tmp)))
        B2 = np.trace(np.matmul(F.T, np.matmul(L, F)))
        tmp = np.matmul(X.T, W) - F
        B3 = parameter_alpha * np.trace(np.matmul(tmp.T, tmp))
        B4 = parameter_lambda * np.trace(np.matmul(np.ones(shape=(n, n)), S * M))
        tmp = X - np.matmul(X, S)
        B5 = parameter_beta * np.trace(np.matmul(tmp.T, tmp))
        obj[it] = B1 + B2 + B3 + B4 + B5
        if it > 1 and np.abs(obj[it] - obj[it-1]) < 0.01:
            break
    return W, F, S, obj

