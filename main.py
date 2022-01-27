import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag, solve_lyapunov
import matplotlib.pyplot as plt
from cycler import cycler
from loguru import logger
from tqdm import trange
import sys

import fym

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def ctrb_matrix(A, B, normalized=False):
    assert (n := A.shape[0]) == A.shape[1]
    f = A.max() if normalized else 1
    C = np.hstack([np.linalg.matrix_power(A, i) / f**i @ B for i in range(n)])
    return C


def ctrb(A, B):
    C = ctrb_matrix(A, B, normalized=True)
    return matrix_rank(C) == A.shape[0]


def inertia(X, tol=1e-10, discrete=False):
    eigs = np.linalg.eigvals(X)
    if not discrete:
        eigs = eigs.real
        p = (eigs > tol).sum()
        z = (np.abs(eigs) <= tol).sum()
        n = (eigs < -tol).sum()
    else:
        eigs = np.abs(eigs)
        p = (eigs > 1 + tol).sum()
        z = np.all((eigs <= 1 + tol, eigs >= 1 - tol)).sum()
        n = (eigs < 1 - tol).sum()
    return p, z, n


class Kleinman:
    def __init__(self):
        A, B, Q, R, K, ITERMAX, KTOL = problem()
        self.A, self.B = A, B
        self.n, self.m = B.shape
        self.Q, self.R = Q, R
        self.K = K
        self.ITERMAX = ITERMAX
        self.KTOL = KTOL

        self.Rinv = np.linalg.inv(R)
        self.Kopt, self.Popt = fym.clqr(A, B, Q, R)

    def reset(self):
        return self.K

    def step(self, K):
        Ak = self.A - self.B @ K
        Qk = self.Q + K.T @ self.R @ K
        Pk = solve_lyapunov(Ak.T, -Qk)
        next_K = self.Rinv @ self.B.T @ Pk

        K_error = np.linalg.norm(K - self.Kopt)
        P_error = np.linalg.norm(Pk - self.Popt)

        stable = inertia(Pk)[0] == self.B.shape[0]
        done = stable and K_error < self.KTOL

        info = dict(
            P=Pk,
            K=K,
            Ak_inertia=inertia(Ak),
            P_error=P_error,
            K_error=K_error,
            stable=stable,
        )

        return next_K, done, info


class Proposed(Kleinman):
    def __init__(self, s=1):
        super().__init__()
        self.s = s

    def step(self, K):
        n, m = self.n, self.m

        Ak = self.A - self.B @ K
        Ab = np.block([
            [Ak, self.B],
            [np.zeros((m, n)), -self.s * np.eye(self.m)]
        ])

        Kb = np.block([
            [np.eye(n), np.zeros_like(self.B)],
            [-K, np.eye(m)]
        ])
        Qb = Kb.T @ block_diag(self.Q, self.R) @ Kb

        Hk = solve_lyapunov(Ab.T, -Qb)
        Pk = Hk[:n, :n]
        Wk = Hk[n:, :n]
        Gk = Hk[n:, n:]

        H22inv = np.linalg.inv(Gk)

        next_K = K + H22inv @ Wk

        K_error = np.linalg.norm(K - self.Kopt)
        P_error = np.linalg.norm(Pk - self.Popt)

        stable = inertia(Pk)[0] == n
        done = stable and K_error < self.KTOL

        info = dict(
            P=Pk,
            K=K,
            Ak_inertia=inertia(Ak),
            P_error=P_error,
            K_error=K_error,
            stable=stable,
            W=Wk,
            G=Gk,
            H=Hk,
        )

        return next_K, done, info


def problem():
    ITERMAX = 500
    KTOL = 1e-10

    A = np.array([
        [-1.341, 0.9933, 0, -0.1689, -0.2518],
        [43.223, -0.8693, 0, -17.251, -1.5766],
        [1.341, 0.0067, 0, 0.1689, 0.2518],
        [0, 0, 0, -20, 0],
        [0, 0, 0, 0, -20],
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [20, 0],
        [0, 20],
    ])

    n, m = B.shape

    Q = np.eye(n)
    R = np.eye(m)

    logger.info("Eigenvalues of A (real): " + ", ".join(
        [f"{x:-5.2e}" for x in np.linalg.eigvals(A).real]))
    logger.info("Rank of B: {}/{}", matrix_rank(B), np.min(B.shape))
    logger.info("Rank of C: {}/{}", matrix_rank(ctrb_matrix(A, B)), n)
    logger.info("Rank of O: {}/{}", matrix_rank(ctrb_matrix(A.T, Q)), n)
    logger.debug("Inertia of Q: {}", inertia(Q))
    logger.debug("Inertia of R: {}", inertia(R))

    def play(seed):
        # seed: 193
        np.random.seed(seed)
        K = 5 * np.random.randn(m, n)
        # K = np.array([
        #     [1, 1, -1, 1, 1],
        #     [1, 1, -1, 1, 1],
        # ])
        return inertia(A - B@K)[0] == 5

    # for seed in range(1000):
    #     if play(seed):
    #         break

    # inertia(A - BK) = (5, 0, 0)
    K = np.array([
        [8.4, -5.1, -4.1, -1.1, 0.4],
        [5.8, -7.4, 3.1, -5.2, -5.1],
    ])
    logger.info("Eigenvalues of A_0 (real): " + ", ".join(
        [f"{x:-5.2e}" for x in np.linalg.eigvals(A - B@K).real]))
    logger.info("Inertia of A_0 (p, z, n): {}", inertia(A - B@K))
    return A, B, Q, R, K, ITERMAX, KTOL


def run_kleinman():
    kleinman = Kleinman()

    data_logger = fym.Logger("data-kleinman.h5")

    K = kleinman.reset()
    for i in trange(kleinman.ITERMAX):
        next_K, done, info = kleinman.step(K)
        data_logger.record(i=i, **info)

        if done:
            break

        K = next_K

    data_logger.set_info(Popt=kleinman.Popt, Kopt=kleinman.Kopt)
    data_logger.close()


def run_proposed():
    proposed = Proposed()

    data_logger = fym.Logger("data-proposed.h5")

    K = proposed.reset()
    for i in trange(proposed.ITERMAX):
        next_K, done, info = proposed.step(K)

        data_logger.record(i=i, **info)

        if done:
            break

        K = next_K

    data_logger.set_info(Popt=proposed.Popt, Kopt=proposed.Kopt)
    data_logger.close()


def run_hybrid():
    hybrid = Proposed(s=1)

    data_logger = fym.Logger("data-hybrid.h5")

    K = hybrid.reset()
    for i in trange(hybrid.ITERMAX):
        next_K, done, info = hybrid.step(K)

        data_logger.record(i=i, **info)

        if info["stable"]:
            hybrid = Kleinman()

        if done:
            break

        K = next_K

    data_logger.set_info(Popt=hybrid.Popt, Kopt=hybrid.Kopt)
    data_logger.close()


def resize_callback(event):
    print("figsize: {}".format(event.canvas.figure.get_size_inches()))
    fig = event.canvas.figure
    fig.tight_layout()
    for i, ax in enumerate(fig.axes):
        print(f"Axes {i}: {ax.get_position()}")


def plot():
    data = {
        "Kleinman": fym.load("data-kleinman.h5", with_info=True),
        "Proposed": fym.load("data-proposed.h5", with_info=True),
        "Proposed (hybrid)": fym.load("data-hybrid.h5", with_info=True),
    }

    # Plotting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.linewidth": 0.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
    })

    my_cycler = (
        cycler(color=[plt.cm.get_cmap("Set1").colors[i] for i in [2, 1, 0]])
        + cycler(linestyle=["--", "-", "-."])
        + cycler(marker=[None, None, "o"])
        # + cycler(marker=["^", "s", "o"])
    )

    adjust1 = dict(left=0.18, right=0.907, bottom=0.12)
    adjust2 = adjust1 | dict(bottom=0.12 / 2.05 * 3.44)

    """ Figure 1 """
    fig = plt.figure("P and K histories", figsize=(3.5, 3.44))
    axes = fig.subplots(2, 1, sharex=True)

    ax = axes[0]
    ax.set_prop_cycle(my_cycler)
    for k, v in data.items():
        ax.plot(v[0]["P_error"], label=k)

        print(f"{k}:")

        print(f"\tFinal P ({v[0]['i'][-1]}):")
        for row in v[0]['P'][-1]:
            print("\t" + " & ".join([f"{elem:5.3f}" for elem in row]))

        print("\tOptimal P:")
        for row in v[1]['Popt']:
            print("\t" + " & ".join([f"{elem:5.3f}" for elem in row]))

        print(f"\tFinal K ({v[0]['i'][-1]}):")
        for row in v[0]['K'][-1]:
            print("\t" + " & ".join([f"{elem:5.3f}" for elem in row]))

        print("\tOptimal K:")
        for row in v[1]['Kopt']:
            print("\t" + " & ".join([f"{elem:5.3f}" for elem in row]))

    ax.set_ylabel(r"$\| P_k - P^\ast \|_F$")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.set_prop_cycle(my_cycler)
    ax.set_yscale("log")
    for k, v in data.items():
        ax.plot(v[0]["K_error"], label=k)
    ax.set_ylabel(r"$\| K_k - K^\ast \|_F$")
    ax.set_xlabel("Iteration")
    ax.set_ylim(data["Proposed"][0]["K_error"][-1], 4e3)
    # ax.set_xlim(0, 500)
    ax.set_xlim(0, len(data["Proposed"][0]["i"]))

    fig.tight_layout()
    plt.subplots_adjust(**adjust1)

    fig.canvas.mpl_connect("resize_event", resize_callback)
    fig.savefig("fig1.pdf", dpi=600)

    """ Figure 2 """
    my_cycler = (
        cycler(color=[plt.cm.get_cmap("Set1").colors[i] for i in [2, 1, 0]])
        + cycler(linestyle=["--", "-", "-."])
        + cycler(marker=["^", "s", "o"])
        # + cycler(fillstyle=["none", "none", "full"])
    )

    fig = plt.figure("Inertia of Ak", figsize=(3.5, 2.05))
    ax = fig.subplots()
    ax.set_prop_cycle(my_cycler)
    for k, v in data.items():
        ax.plot(v[0]["Ak_inertia"][:, 0], label=k)
    ax.set_ylabel(r"$\pi(A_k)$", labelpad=13)
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, 30)
    ax.legend()

    fig.tight_layout()
    plt.subplots_adjust(**adjust2)

    fig.canvas.mpl_connect("resize_event", resize_callback)
    fig.savefig("fig2.pdf", dpi=600)

    plt.show()


def main():
    # run_kleinman()
    # run_proposed()
    # run_hybrid()
    plot()


if __name__ == "__main__":
    main()
