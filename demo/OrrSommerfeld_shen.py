"""
Solve the Orr-Sommerfeld eigenvalue problem

Using Shen's biharmonic basis

"""
import warnings
from scipy.linalg import eig
#from numpy.linalg import eig
#from numpy.linalg import inv
import numpy as np
from shenfun import Basis
from shenfun.spectralbase import inner_product
from shenfun.matrixbase import extract_diagonal_matrix

np.seterr(divide='ignore')

#pylint: disable=no-member

try:
    from matplotlib import pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")

class OrrSommerfeld(object):
    def __init__(self, alfa=1., Re=8000., N=80, quad='GC', **kwargs):
        kwargs.update(dict(alfa=alfa, Re=Re, N=N, quad=quad))
        vars(self).update(kwargs)
        self.P4 = np.zeros(0)
        self.T4x = np.zeros(0)
        self.SB, self.SD, self.CDB = (None,)*3
        self.x, self.w = None, None

    def interp(self, y, eigvals, eigvectors, eigval=1, verbose=False):
        """Interpolate solution eigenvector and it's derivative onto y

        Parameters
        ----------
            y : array
                Interpolation points
            eigvals : array
                All computed eigenvalues
            eigvectors : array
                All computed eigenvectors
            eigval : int, optional
                The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
            verbose : bool, optional
                Print information or not
        """
        N = self.N
        nx, eigval = self.get_eigval(eigval, eigvals, verbose)
        phi_hat = np.zeros(N, np.complex)
        phi_hat[:-4] = np.squeeze(eigvectors[:, nx])

        if not len(self.P4) == len(y):
            SB = Basis(N, 'C', bc='Biharmonic', quad=self.quad)
            self.P4 = SB.evaluate_basis_all(x=y)
            self.T4x = SB.evaluate_basis_derivative_all(x=y, k=1)
        phi = np.dot(self.P4, phi_hat)
        dphidy = np.dot(self.T4x, phi_hat)

        return eigval, phi, dphidy

    def assemble(self):
        N = self.N
        SB = Basis(N, 'C', bc='Biharmonic', quad=self.quad)
        SB.plan((N, N), 0, np.float, {})

        x, _ = self.x, self.w = SB.points_and_weights(N)

        # Trial function
        P4 = SB.evaluate_basis_all(x=x)

        # Second derivatives
        T2x = SB.evaluate_basis_derivative_all(x=x, k=2)

        # (u'', v)
        K = np.zeros((N, N))
        K[:-4, :-4] = inner_product((SB, 0), (SB, 2)).diags().toarray()

        # ((1-x**2)u, v)
        xx = np.broadcast_to((1-x**2)[:, np.newaxis], (N, N))
        #K1 = np.dot(w*P4.T, xx*P4)  # Alternative: K1 = np.dot(w*P4.T, ((1-x**2)*P4.T).T)
        K1 = np.zeros((N, N))
        K1 = SB.scalar_product(xx*P4, K1)
        K1 = extract_diagonal_matrix(K1).diags().toarray() # For improved roundoff

        # ((1-x**2)u'', v)
        K2 = np.zeros((N, N))
        K2 = SB.scalar_product(xx*T2x, K2)
        K2 = extract_diagonal_matrix(K2).diags().toarray() # For improved roundoff

        # (u'''', v)
        Q = np.zeros((self.N, self.N))
        Q[:-4, :-4] = inner_product((SB, 0), (SB, 4)).diags().toarray()

        # (u, v)
        M = np.zeros((self.N, self.N))
        M[:-4, :-4] = inner_product((SB, 0), (SB, 0)).diags().toarray()

        Re = self.Re
        a = self.alfa
        B = -Re*a*1j*(K-a**2*M)
        A = Q-2*a**2*K+a**4*M - 2*a*Re*1j*M - 1j*a*Re*(K2-a**2*K1)
        return A, B

    def solve(self, verbose=False):
        """Solve the Orr-Sommerfeld eigenvalue problem
        """
        if verbose:
            print('Solving the Orr-Sommerfeld eigenvalue problem...')
            print('Re = '+str(self.Re)+' and alfa = '+str(self.alfa))
        A, B = self.assemble()
        return eig(A[:-4, :-4], B[:-4, :-4])
        # return eig(np.dot(inv(B[:-4, :-4]), A[:-4, :-4]))

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """Get the chosen eigenvalue

        Parameters
        ----------
            nx : int
                The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : array
                Computed eigenvalues
            verbose : bool, optional
                Print the value of the chosen eigenvalue. Default is False.

        """
        indices = np.argsort(np.imag(eigvals))
        indi = indices[-1*np.array(nx)]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if np.ndim(eigval) else [eigval]
            indi = list(indi) if np.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
                print('Eigenvalue {} ({}) = {:2.16e}'.format(i+1, v, e))
        return indi, eigval

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orr Sommerfeld parameters')
    parser.add_argument('--N', type=int, default=120,
                        help='Number of discretization points')
    parser.add_argument('--Re', default=8000.0, type=float,
                        help='Reynolds number')
    parser.add_argument('--alfa', default=1.0, type=float,
                        help='Parameter')
    parser.add_argument('--quad', default='GC', type=str, choices=('GC', 'GL'),
                        help='Discretization points: GC: Gauss-Chebyshev, GL: Gauss-Lobatto')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Plot eigenvalues')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print results')
    parser.set_defaults(plot=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    #z = OrrSommerfeld(N=120, Re=5772.2219, alfa=1.02056)
    z = OrrSommerfeld(**vars(args))
    evals, evectors = z.solve(args.verbose)
    d = z.get_eigval(1, evals, args.verbose)
    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80:
        assert abs(d[1] - (0.24707506017508621+0.0026644103710965817j)) < 1e-12

    if args.plot:
        plt.figure()
        evi = evals*z.alfa
        plt.plot(evi.imag, evi.real, 'o')
        plt.axis([-10, 0.1, 0, 1])
        plt.show()
