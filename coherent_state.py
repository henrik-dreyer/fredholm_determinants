import numpy as np
from scipy.integrate import simpson


'''Helper Functions for M(lambda, mu) = J'''

def make_single_f_tilde(fk, flam, k_grid, other_grid, lambda_or_mu='lambda'):

    nX=k_grid.shape[0]
    nQ=other_grid.shape[0]

    f_out = np.kron(fk.reshape((nX, 1)), np.ones((1, nQ))) \
                       - np.kron(np.ones((nX, 1)), flam.reshape((1, nQ)))

    f_out = f_out * np.kron(np.sin(k_grid) .reshape((nX, 1)), np.ones((1, nQ))) \
            / (np.kron(np.ones((nX, 1)), np.cos(other_grid).reshape((1, nQ))) - np.kron(np.cos(k_grid) .reshape((nX, 1)), np.ones((1, nQ))))

    return f_out


def make_v_lam(flam, lambda_grid):
    Q = len(flam)
    flam_lm = np.kron(flam.reshape((Q, 1)), np.ones((1, Q)))
    v_lam = 1 / np.pi ** 2 * flam_lm.conj() / (1 + abs(flam_lm) ** 2) * np.kron(np.sin(lambda_grid), np.ones((Q, 1))).T
    return v_lam.T


def make_J(fk, flam, fmu, k_grid, lambda_grid, mu_grid):

    Q = len(flam)

    glam_sparse = simpson(make_single_f_tilde(fk, flam, k_grid, lambda_grid), x=k_grid, axis=0, even='avg')
    offset_lam = flam * np.log(np.abs((np.cos(lambda_grid) + 1) / (np.cos(lambda_grid) - 1)))
    glam_sparse = glam_sparse + offset_lam

    gmu_sparse = simpson(make_single_f_tilde(fk, fmu, k_grid, mu_grid), x=k_grid, axis=0, even='avg')
    offset_mu = fmu * np.log(np.abs((np.cos(mu_grid) + 1) / (np.cos(mu_grid) - 1)))
    gmu_sparse = gmu_sparse + offset_mu

    J_sparse = np.kron( glam_sparse, np.ones((Q, 1)))
    J_sparse = J_sparse - np.kron( gmu_sparse, np.ones((Q, 1))).T
    J_sparse = J_sparse / ( np.kron(np.cos(lambda_grid).reshape((Q, 1)), np.ones((1, Q)))
                          - np.kron(np.ones((Q, 1)), np.cos(mu_grid).reshape((1, Q))) ).T

    J_sparse = J_sparse * make_v_lam(flam, lambda_grid) * np.pi/Q

    return J_sparse



class CoherentState():

    """
    This class provides the interface for computing observable and time evolving
    coherent states.


    Parameters
    ----------
    model_params : dict
        A dictionary with all the model parameters.
        q: (Integer) Increase to for more accurate Fredholm determinant
                     Determinant converges as exp(-2q).
                     Typically q=10-14 is enough
        r: (Integer) Ratio of system size vs Fredholm determinant
                     e.g. r=1 -> L = 3/2 2^q
                     Increase to capture faster oscillations
        f_initial: Callable function f(k) to initialise the coherent state


    Attributes
    ----------
    Q : Fredholm discretization. Must be a power of 2, Q = 2^q

    R : Effective system size. Must be an half-odd integer multiple of Q,
                                R = (2r+1)/2 * Q

    f_initial: A function f(k) to populate initial values.

    lambda_grid: Array on which f(lambda) is evaluated.
    mu1_grid: Array on which f(mu1) is evaluated.
    mu2_grid: Array on which f(mu2) is evaluated.
    k_grid: Array on which f(k) is evaluated. Grids are spaced
            in such a way as to avoid numerical singularities
            when computing Fredholm determinants.
    """

    def __init__(self,model_params):
        self.q = model_params.get('q', 2)
        self.r = model_params.get('r', 2)
        self.Q = 2**self.q
        self.R = (2*self.r+1) * self.Q//2
        self.f_initial = model_params.get('f_initial', None)
        self.lambda_grid=(np.arange(self.Q)+1/2) * np.pi/self.Q
        self.mu1_grid=(np.arange(self.Q)+1/4) * np.pi/self.Q
        self.mu2_grid=(np.arange(self.Q)+3/4) * np.pi/self.Q
        self.k_grid=(np.arange(self.R+1)) * np.pi/self.R
        print('Preparing a coherent state')
        print('Number of k-points R = {}'.format(self.R))
        print('Lambda-Sampling points Q = {}'.format(self.Q))

        if callable(self.f_initial):
            self.flambda = self.f_initial(self.lambda_grid)
            self.fmu1 = self.f_initial(self.mu1_grid)
            self.fmu2 = self.f_initial(self.mu2_grid)
            self.fk = self.f_initial(self.k_grid)


    def evolve(self, dfdt, t):
        """
        Parameters
        ----------
        dfdt : Callable function dfdt(f,k) -> vector with len(k) entries

        Returns
        ----------
        out : det( Id - M(lambda,mu1) ) + det( Id - M(lambda,mu2) )
        i.e. eq. (20) in v1 of https://arxiv.org/abs/2104.01168
        """
        self.flambda = self.flambda + t*dfdt(self.flambda, self.lambda_grid)
        self.fmu1 = self.fmu1 + t*dfdt(self.fmu1, self.mu1_grid)
        self.fmu2 = self.fmu2 + t*dfdt(self.fmu2, self.mu2_grid)
        self.fk = self.fk + t*dfdt(self.fk, self.k_grid)

    def write_to_txt(self, txt_file):

        with open(txt_file, "w") as output:

            output.write('\n')
            output.write('\n')
            output.write('Re(f) = ')
            output.write('\n')
            for z in np.real(self.fk):
                output.write(str(z))
                output.write(str(','))
            output.write('\n')

            output.write('\n')
            output.write('\n')
            output.write('Im(f) = ')
            output.write('\n')
            for z in np.imag(self.fk):
                output.write(str(z))
                output.write(str(','))
            output.write('\n')

            output.write('\n')
            output.write('\n')
            output.write('k = ')
            output.write('\n')
            for z in np.real(self.k_grid):
                output.write(str(z))
                output.write(str(','))
            output.write('\n')


    def compute_m(self):
        """
        Parameters
        ----------

        Returns
        ----------
        out : det( Id - M(lambda,mu1) ) + det( Id - M(lambda,mu2) )
        i.e. eq. (20) in v1 of https://arxiv.org/abs/2104.01168
        """
        Q = len(self.flambda)
        m1 = np.linalg.det(np.eye(Q) - make_J(self.fk, self.flambda, self.fmu1, self.k_grid, self.lambda_grid, self.mu1_grid))
        m2 = np.linalg.det(np.eye(Q) - make_J(self.fk, self.flambda, self.fmu2, self.k_grid, self.lambda_grid, self.mu2_grid))

        m = np.real((m1 + m2) / 2)

        return m
