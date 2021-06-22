import numpy as np
from scipy.integrate import simpson

'''Helper Functions for XY model'''
def theta(h,gamma,k):
  if h=='inf':
    out = np.zeros(len(k))
  else:
    out = -1j*np.log(h-np.exp(1j*k) / (np.sqrt( 1+h**2-2*h*np.cos(k) )))
  return out

def K(hbar, gammabar, h, gamma, k):
    out = np.tan( (theta(hbar,gammabar,k) - theta(h,gamma,k)) /2 )
    return out


'''Helper Functions for Ising model'''
def eps(h, k):
  #CAREFUL IN RAMOND SECTOR: TREAT k=0 SEPARATELY
  if h=='inf':
    out = 2. * np.ones(len(k))
  else:
    out = 2. * np.sqrt( 1. + h**2 - 2.*h*np.cos(k) )
  return out


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


    def corr_func(self):
        """
        TODO: equation (59) in xy_field.pdf
        self should be of h-type

        """



class SparseState(CoherentState):

    """
    This class provides basic functionality to propagate f(k).
    It is Sparse because f is only tracked in the NS+ sector, allowing
    for faster computation and training.
    Fredholm determinants can *not* be computed.
    Only the Ising submodel is implemented (corresponds to gamma = 1 in parent)


    Parameters
    ----------
    model_params : dict
        A dictionary with all the model parameters.
        L: (Integer) System size. Only tested with even L
        f_initial: Callable function f(k) to initialise the coherent state
        h: Initial magnetic field. Default h='inf', corresponding to state |++...>


    Attributes
    ----------
    L : (Integer) System Size
    h : (Float) Current basis of f.
    NS_plus : (Numpy Array) k-Grid with points pi/L * [1,3,5,..L-1]
    """

    def __init__(self,model_params):
        self.f_initial = model_params.get('f_initial', lambda x: x*0)
        self.h = 'inf'
        self.L = model_params.get('L', 4)
        self.NS_plus=np.arange(1, self.L, 2)*np.pi/self.L

        if callable(self.f_initial):
            self.fk = self.f_initial(self.NS_plus)


    def change_basis(self, h_target):
        """
        changes h and f(k) in-place according to
        cf. equation (A15) in https://arxiv.org/pdf/2104.01168.pdf

        Parameters
        ----------
        h0: float initial transverse field
        h1: float final transverse field
        """
        K_val = K(h_target, 1, self.h, 1, self.NS_plus)
        self.fk = ( 1j * K_val + self.fk ) /  ( 1 + 1j * K_val * self.fk )
        self.h = h_target


    def eigenstate_evolve(self, t):
        """
        changes f(k) in-place according to
        equation (A23) in https://arxiv.org/pdf/2104.01168.pdf

        Parameters
        ----------
        t: float time
        """
        self.fk = np.exp(-2*t*1j*eps(self.h, self.NS_plus)) * self.fk


    def set_Ck(self):
        """
        Sets correlation matrix in momentum space (diagonal)
        C_k = <f| a^\dagger_k a_k  |f>
        cf. equation (28) in xyfield12.pdf

        Parameters
        ----------
        """
        self.Ck = np.abs(self.fk) ** 2 / ( 1 + np.abs(self.fk)**2 )

    def set_Fk(self):
        """
        Sets *anomalous* correlation matrix in momentum space (diagonal)
        F_k = <f| a_k a_{-k}  |f>
        cf. equation (28) in xyfield12.pdf
        Note: This is the *complex conjugate* of https://arxiv.org/pdf/0906.1663.pdf
        ~ eq (17)

        Parameters
        ----------
        """
        self.Fk = self.fk / ( 1 + np.abs(self.fk)**2 )



    def set_Cxy_Fxy(self):
        """
        Sets fourier transformed correlation matrix in real space.
        Note momentum sum runs over both NS+ and NS-.
        We use f(k) = -f(-k)
        C_xy = 1/L * sum_{k \in NS} exp(-ik (x-y))

        Parameters
        ----------

        """
        x = np.arange(self.L)
        NS = np.append(self.NS_plus,-1*self.NS_plus)
        Wxk = np.exp(-1j*np.outer(x,NS))
        Ckk = np.diag(np.append(self.Ck,self.Ck))
        Fkk = np.diag(np.append(self.Fk,-1*self.Fk))
        self.Cxy = 1 / self.L * Wxk.dot( Ckk.dot(Wxk.conj().T))
        self.Fxy = 1 / self.L * Wxk.dot( Fkk.dot(Wxk.conj().T))

    def trace_out(self, sites):
        """
        Removes rows and columns from real space correlation matrices
        Cxy and Fxy in-place. Sets Majorana modes
            M = <a_m a_n>
        where
            a_{2n-1} = c_n + c_n^\dagger    [EVEN]
            a_{2n} = i(c_n - c_n^\dagger)     [ODD]


        Parameters
        ----------
        sites: (List of Integers) The sites to be traces out (complement remains)
        """

        l = self.L - len(sites)
        self.Cxy=np.delete(self.Cxy, sites, axis=0)
        self.Cxy=np.delete(self.Cxy, sites, axis=1)
        self.Fxy=np.delete(self.Fxy, sites, axis=0)
        self.Fxy=np.delete(self.Fxy, sites, axis=1)

        self.iGamma = np.zeros((2*l, 2*l),dtype=complex)

        #odd first, then even
        self.iGamma[0:l, 0:l] = self.Cxy - self.Cxy.T - self.Fxy - self.Fxy.conj().T
        self.iGamma[l:2*l, l:2*l] = self.Cxy - self.Cxy.T + self.Fxy + self.Fxy.conj().T
        self.iGamma[0:l, l:2*l] = 1j*(self.Fxy - self.Cxy - self.Cxy.T - self.Fxy.conj().T + np.eye(l))
        self.iGamma[l:2*l, 0:l] = 1j*(self.Fxy + self.Cxy + self.Cxy.T - self.Fxy.conj().T - np.eye(l))

        #Permuting rows and columns does not change the eigenvalues
        perm=0
        if perm==True:
            a = np.arange(0,l)
            b = np.arange(l,2*l)
            c = np.zeros((a.size + b.size,), dtype=a.dtype)
            c[0::2] = a
            c[1::2] = b

            idx = np.empty_like(c)
            idx[c] = np.arange(len(c))
            self.M = self.M[:, idx]  # return a rearranged copy
            self.M = self.M[idx, :]/2

        self.iGamma = self.iGamma

        self.eigvals=np.sort(np.linalg.eigh(self.iGamma)[0])[::-1]
        self.eigvals = self.eigvals[0:self.eigvals.shape[0]//2]
        self.epsl = np.arctanh(self.eigvals)*2

        #TODO: Set up entropy from epsilon l
        #M = np.transpose(M, c, axis=0)
        #M = np.transpose(M, c, axis=1)