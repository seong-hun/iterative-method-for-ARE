import numpy as np
from scipy.linalg import block_diag, solve_continuous_lyapunov

np.random.seed(1)

n = 2
m = 1

q = np.random.rand(n, 1)
Q = q @ q.T

A = np.random.rand(n, n)
b = np.random.rand(n, m)

P = solve_continuous_lyapunov(A.T, -Q)
s = 1
I = np.eye(n)
w = - (b.T @ P @ np.linalg.inv(A - s * I)).T

r = - 2 * w.T @ b
Qb = block_diag(Q, r)

S = np.block([[A, b], [np.zeros((m, n)), - s]])
H = solve_continuous_lyapunov(S.T, -Qb)
W = H[n:, :n]
G = H[n:, n:]

K = np.linalg.pinv(np.zeros((1, 1))) @ W
P_next = P - W.T @ np.linalg.pinv(G) @ W

s = 0.9
S = np.block([[A, b], [np.zeros((m, n)), - s]])
H = solve_continuous_lyapunov(S.T, -Qb)
W = H[n:, :n]
G = H[n:, n:]

K = np.linalg.pinv(np.zeros((1, 1))) @ W
P_next = P - W.T @ np.linalg.pinv(G) @ W

breakpoint()
