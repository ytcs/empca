import numpy as np
import scipy
from scipy.signal import savgol_filter,wiener
from tqdm import *


def polar(x):
    return np.absolute(x), np.angle(x)


def rect(r, theta):
    return r*np.exp(1j*theta)


def ti_rfft(x, truncate=None, axis=-1):
    if x.ndim == 1:
        x=x.reshape([1,-1])
    end_ind= None if truncate is None else -truncate
    r, theta = polar(np.fft.rfft(x, axis=axis)[:,:end_ind])
    theta_shifted = np.roll(theta, -1, axis=axis)
    # we keep last component unchanged for inverse transform
    if theta.ndim == 1:
        theta_shifted[-1] = 0
    else:
        theta_shifted[:, -1] = 0
    theta = theta-theta_shifted
    return np.squeeze(rect(r, theta))

def ti_irfft(x, padding=0,axis=-1):
    if x.ndim == 1:
        x=x.reshape([1,-1])
    
    r, theta = polar(np.concatenate((x,np.zeros([x.shape[0],padding],dtype=complex)),axis=-1))
    theta = np.flip(np.cumsum(np.flip(theta, axis=axis), axis=axis), axis=axis)
    return np.squeeze(np.fft.irfft(rect(r, theta), axis=axis))

def orthonormalize(A):
    n_comp, _ = A.shape
    for i in range(n_comp):
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, n_comp):
        for j in range(0, i):
            A[i] -= np.dot(np.conjugate(A[j]), A[i]) * A[j]
            A[i] /= np.linalg.norm(A[i])
    return A


def random_orthonormals(n_comp, n_var, seed=1):
    if seed is not None:
        np.random.seed(seed)

    return orthonormalize(np.random.normal(size=(n_comp, n_var)))


def smooth(A, window=15, polyord=3, deriv=1):
    return savgol_filter(np.real(A), window, polyord, deriv)+1j*savgol_filter(np.imag(A), window, polyord, deriv)


class empca_solver:
    def __init__(self, n_comp, data, weights):
        self.set_data(data)
        self.weights = weights
        self.n_comp = n_comp
        self.eigvec = random_orthonormals(self.n_comp, self.n_var)
        self.coeff = self.solve_coeff()

    def set_data(self, data):
        self.data = data
        self.n_obs, self.n_var = data.shape

    def chi2(self):
        residual = self.data-self.coeff@self.eigvec
        return np.absolute(np.mean(np.sum(residual@self.weights@(np.conjugate(residual).T), axis=-1)))

    def solve_coeff(self, data=None):
        if data is None:
            data = self.data
        Phi = self.eigvec.T

        # Solve (W \Phi)^\dagger X = (W \Phi)^\dagger \Phi coeff
        WPd = np.conjugate(self.weights @ Phi).T
        return np.stack(
            [scipy.linalg.lstsq(WPd@Phi, WPd@X, lapack_driver='gelsy', check_finite=False)[0] for X in data])

    def solve_eigvec(self, data=None, mode='fast'):
        if mode.lower() == 'fast':
            return self.solve_eigvec_fast(data)
        elif mode.lower() == 'full':
            return self.solve_eigvec_full(data)
        else:
            raise ValueError("Mode should be 'fast' or 'full'.")

    def solve_eigvec_fast(self, data=None):
        if data is None:
            data = self.data

        CC = np.sum(np.square(np.absolute(self.coeff)), axis=0)
        WXC = self.weights@(self.data.T)@np.conjugate(self.coeff)

        self.eigvec = np.zeros((self.n_comp, self.n_var))

        self.eigvec = np.stack([scipy.linalg.lstsq(
            CC[i]*self.weights, WXC[:, i], lapack_driver='gelsy', check_finite=False)[0] for i in range(self.n_comp)])

        return orthonormalize(self.eigvec)

    def solve_eigvec_full(self, data=None):
        if data is None:
            data = self.data
        BigW = np.zeros(
            [self.n_comp*self.n_var, self.n_comp*self.n_var], dtype=complex)
        for n in range(self.n_comp):
            for m in range(self.n_comp):
                BigW[n*self.n_var:(n+1)*self.n_var, m*self.n_var:(m+1)*self.n_var] = np.dot(
                    np.conjugate(self.coeff[:, n]), self.coeff[:, m])*self.weights

        WXC = (self.weights@(self.data.T)@np.conjugate(
            self.coeff)).reshape([-1, 1], order='F')

        self.eigvec = np.zeros((self.n_comp, self.n_var))

        self.eigvec = scipy.linalg.lstsq(
            BigW, WXC, lapack_driver='gelsy', check_finite=False)[0]
        self.eigvec = self.eigvec.reshape([self.n_comp, -1])
        return orthonormalize(self.eigvec)


class EMPCA:
    def __init__(self, n_comp=5):
        self.n_comp = n_comp
        self.solver = None

    def fit(self, X, weights, n_iter=50, window=15, polyord=3, deriv=0, patience=5, mode='fast', verbose=False):
        _patience = patience
        chi2s = []
        if self.solver is None:
            self.solver = empca_solver(self.n_comp, X, weights)
        else:
            self.solver.set_data(X)
        for _ in tqdm(range(n_iter)):
            self.solver.eigvec = smooth(self.solver.solve_eigvec(
                mode=mode), window=window, polyord=polyord, deriv=deriv)  # smoothing is important for performance
            self.solver.coeff = self.solver.solve_coeff()
            chi2 = self.solver.chi2()
            if verbose:
                print(f'chi2= {chi2}')
            if len(chi2s) > 0 and chi2 > chi2s[-1]:
                if patience <= 0:
                    break
                else:
                    patience -= 1
            chi2s.append(chi2)
        self.eigvec = self.solver.eigvec
        self.coeff = self.solver.coeff
        return chi2s

    def project(self, X):
        if self.solver is None:
            raise Exception(
                'Solver has not been initialized. Please run fit() first.')
        return self.solver.solve_coeff(X)
