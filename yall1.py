import numpy as np
import numpy.linalg as npla


class Solver4L1Minimization:
    def __init__(self, tol=-1, nu=0, rho=0, delta=0, nonneg=0, W=np.array([]), weights=np.array([]), nonorth=-1,
                 stepfreq=1, maxit=9999, gamma=1):
        self.tol = tol  # user must specifiy. It depends on noise levels in b, values [1e-2, 1e-4] should be sufficient
        self.nu = nu  # paremeter in L1-L1 model
        self.rho = rho  # paremeter in L1-L2 model
        self.delta = delta  # paremeter in L1-L2 constrained model
        self.nonneg = nonneg  # 1 for nonnegativity
        self.W = W  # sparsefying basis W
        self.weights = weights if weights.size > 0 else 1 # weights for each element in x (Ax = b)
        self.nonorth = nonorth  # check whether the A is orthogonal matrix
        self.stepfreq = stepfreq  # frequency of calculating the exact steepest descent step-length when AA* != I
        self.maxit = maxit  # number of iteration in solver
        self.gamma = gamma  # ADM parameter
        self.mu = -1  # set by code
        self.eps = 2.2204e-16  # matlab floating point relative accuracy

    def set_options(self, tol, nu=0, rho=0, delta=0, nonneg=0, W=np.array([]), weights=np.array([]), nonorth=-1,
                    stepfreq=1, maxit=9999, gamma=1):
        self.__init__(tol, nu, rho, delta, nonneg, W, weights, nonorth, stepfreq, maxit, gamma)

    def check_conflicts(self):
        if self.delta > 0 and self.rho > 0 or self.delta > 0 and self.nu > 0 or self.rho > 0 and self.nu > 0:
            raise ValueError("Model parameters conflict!")
        if self.delta > 0:
            print("YALL1 is solving the constrained L1-L2 problem")
        elif self.rho > 0:
            print("YALL1 is solving the unconstrained L1-L2 problem")
        elif self.nu > 0:
            print("YALL1 is solving the unconstrained L1-L1 problem")
        else:
            print("YALL1 is solving the basis pursuit problem")

    def check_orth(self, A, At, b):
        s1 = np.random.randn(len(b))
        s2 = A(At(s1))
        err = npla.norm(s1-s2) / npla.norm(s1)
        self.nonorth = 1 if err > 1.0e-12 else 0

    def linear_operators(self, A0, b0):
        self.A0 = A0
        b = b0.copy()
        if A0.shape[0] > A0.shape[1]:
            raise ValueError("Matrix A (m x n) must have m <= n")
        A = lambda x: np.matmul(A0, x)
        At = lambda x: np.matmul(A0.T, x)
        # use sparsefying basis W
        if self.W.size > 0:
            if self.W.shape[1] != A0.shape[1]:  # W is square matrix
                raise ValueError("Matrix A (m x n) and matrix W (n x n) don't have valid dimensions")
            else:
                A = lambda x: np.matmul(np.matmul(A, self.W.T), x)
                At = lambda x: np.matmul(np.matmul(self.W, A.T), x)
        # solving L1-L1 model if nu > 0
        if self.nu > 0:
            C = A
            Ct = At
            m = len(b0)
            t = 1/np.sqrt(1 + self.nu**2)
            A = lambda x: (C(x[:-m]) + self.nu * x[-m:]) * t
            At = lambda x: np.concatenate((Ct(x), self.nu * x), axis=0) * t
            b = b0 * t
        if self.nonorth == -1:
            self.check_orth(A, At, b)
        return A, At, b

    def check_zero_solution(self, At, b):
        Atb = At(b)
        bmax = npla.norm(b, ord=np.inf)
        L2Unc_zsol = self.rho > 0 and npla.norm(Atb, ord=np.inf) <= self.rho
        L2Con_zsol = self.delta > 0 and npla.norm(b) <= self.delta
        L1L1_zsol = self.nu > 0 and bmax < self.tol
        BP_zsol = not self.rho > 0 and not self.delta > 0 and self.nu > 0 and bmax < self.tol
        return 1 if L2Unc_zsol or L2Con_zsol or L1L1_zsol or BP_zsol else 0

    def proj2box(self, z, m):
        if self.nonneg:
            z = np.minimum(self.weights, np.real(z))
            if self.nu > 0:
                z[-m:] = np.maximum(-1, z[-m:])
        else:
            z *= self.weights / np.maximum(self.weights, np.abs(z))
        return z

    def check_stopping(self, rd, z, x, xp, b, y, A, cntA):
        q = 0 if self.delta > 0 else 0.1  # q in [0, 1)
        rdnrm = npla.norm(rd)
        # dual residual
        rel_rd = rdnrm / npla.norm(z)
        # duality gap
        objp = np.sum(abs(self.weights * x))
        objd = np.dot(b, y)
        if self.delta > 0:
            objd -= self.delta * npla.norm(y)
        if self.rho > 0:
            rp = A(x) - b
            rpnrm = npla.norm(rp)
            cntA += 1
            objp += 0.5 / self.rho * rpnrm**2
            objd -= 0.5 * self.rho * npla.norm(y)**2
        rel_gap = np.abs(objd - objp) / np.abs(objp)
        # check relative change
        xrel_chg = npla.norm(x - xp) / npla.norm(x)
        exit_msg = ""
        stop = 0
        if xrel_chg < self.tol * (1 - q):
            exit_msg = "Exit: Stabilized"
            stop = 1
            return stop, exit_msg, rel_gap, rel_rd, cntA
        # decide whetehr to go further
        if xrel_chg >= self.tol * (1 + q):
            return stop, exit_msg, rel_gap, rel_rd, cntA
        if not rel_gap < self.tol:  # small gap
            return stop, exit_msg, rel_gap, rel_rd, cntA
        if not rel_rd < self.tol:  # d feasible
            return stop, exit_msg, rel_gap, rel_rd, cntA
        if self.rho == 0:
            rp = A(x) - b
            rpnrm = npla.norm(rp)
            cntA += 1
        if self.rho > 0:
            p_feasable = True
        elif self.delta > 0:
            p_feasable = rpnrm <= self. delta * (1 + self.tol)
        else:
            p_feasable = rpnrm < self.tol * npla.norm(b)
        if p_feasable:
            stop = 1
            exit_msg = "Exit: Converged"
        return stop, exit_msg, rel_gap, rel_rd, cntA

    def update_mu(self, mu_orig, mu, rel_gap, rel_rd, it):
        mfrac = 0.1
        big = 50
        nup = 8
        mu_min = mfrac ** nup * mu_orig
        do_update = rel_gap > big * rel_rd
        do_update = do_update and mu > 1.1 * mu_min
        do_update = do_update and it > 10
        if not do_update:
            return mu
        return np.maximum(mfrac * mu, mu_min)

    def solve(self, A, At, b, x0, z0):
        # initialization
        self.rho = self.eps if self.rho == 0 else self.rho  # for BP
        m = len(b)
        x = At(b) if x0.size == 0 else x0
        n = len(x)
        z = np.zeros(n) if z0.size == 0 else z0
        if self.nonorth > 0:
            y = np.zeros(m)
            Aty = np.zeros(n)
        print('-- YALL1 v1.4 ---')
        mu = np.mean(np.abs(b)) if self.mu == -1 else self.mu
        mu_orig = mu.copy()
        rdmu = self.rho / mu
        rdmu1 = rdmu + 1
        bdmu = b / mu
        ddmu = self.delta / mu
        cntA = 0
        cntAt = 0
        stop = 0
        # main iterations
        for it in range(1, self.maxit+1):
            xdmu = x / mu
            if self.nonorth == 0:  # orthonormal A
                y = A(z - xdmu) + bdmu
                if self.rho > 0:
                    y /= rdmu1
                elif self.delta > 0:
                    y = np.maximum(0, 1 - ddmu / npla.norm(y)) * y
                Aty = At(y)
            else:  # non-orthonormal A
                ry = A(Aty - z + xdmu) - bdmu
                if self.rho > 0:
                    ry += rdmu * y
                Atry = At(ry)
                denom = np.matmul(Atry.T, Atry)
                if self.rho > 0:
                    denom += rdmu * np.dot(ry, ry)
                stp = np.real(np.dot(ry, ry)) / (np.real(denom) + self.eps)
                cntAt += 1
                y -= stp * ry
                Aty -= stp * Atry
            z = Aty + xdmu
            z = self.proj2box(z, m)
            cntA += 1
            cntAt += 1
            rd = Aty - z
            xp = x.copy()
            x += self.gamma * mu * rd
            # other chores
            if np.remainder(it, 2) == 0:
                stop, exit_msg, rel_gap, rel_rd, cntA = self.check_stopping(rd, z, x, xp, b, y, A, cntA)
                new_mu = self.update_mu(mu_orig, mu, rel_gap, rel_rd, it)
                if new_mu != mu:
                    mu = new_mu
                    rdmu = self.rho / mu
                    rdmu1 = rdmu + 1
                    bdmu = b / mu
                    ddmu = self.delta / mu
            if stop:
                break
        if it == self.maxit:
            exit_msg = "Exit: maxiter"
        return x, it, cntAt, cntA, exit_msg

    def get_solution(self, A, b, x0=np.array([]), z0=np.array([])):
        self.check_conflicts()
        m = len(b)
        n = A.shape[1]
        if self.nu > 0 and np.ndim(self.weights) > 0 and len(self.weights) > 1:  # self.weights are by default 1
            self.weights = np.concatenate((self.weights, np.ones(m)), axis=0)
        A, At, b = self.linear_operators(A, b)
        if self.check_zero_solution(At, b):
            x = np.zeros(n)
            it = 0
            cntAt = 1
            cntA = 0
            exit_msg = "Data b = 0"
            return x, it, cntAt, cntA, exit_msg
        bmax = npla.norm(b, ord=np.inf)
        # scaling data and model parameters
        b1 = b / bmax
        if self.rho > 0:
            self.rho /= bmax
        if self.delta > 0:
            self.delta /= bmax
        # skip opts.xs -- could not find its role
        x, iter, cntAt, cntA, exit_msg = self.solve(A, At, b1, x0, z0)
        # restore solution
        x *= bmax
        if self.nu > 0:
            x = x[:-m]
        if self.W.size > 0:
            x = np.matmul(self.W.T, x)
        if self.nonneg:
            x = np.maximum(x, 0)
        return x, iter, cntAt, cntA, exit_msg
















