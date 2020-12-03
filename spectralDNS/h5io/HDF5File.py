import os
import sys
from mpi4py import MPI
from shenfun import ShenfunFile

__all__ = ['HDF5File']

#pylint: disable=dangerous-default-value,unused-argument

comm = MPI.COMM_WORLD

class HDF5File(object):
    """Class for storing and retrieving spectralDNS data

    The class stores two types of data

        - checkpoint
        - results

    Checkpoint data are used to store intermediate simulation results, and can
    be used to restart a simulation at a later stage, with no loss of accuracy.

    Results data are used for visualization.

    Data is provided as dictionaries. The checkpoint dictionary is represented
    as::

        checkpoint = {'space': T,
                      'data': {
                          '0': {'U': [U_hat]},
                          '1': {'U': [U0_hat]},
                          ...
                          }
                      }

    where T is the function space of the data to be stored, and 'data' contains
    solutions to be stored at possibly several different timesteps. The current
    timestep is 0, previous is 1 and so on if more is needed by the integrator.
    Note that checkpoint is storing results from spectral space, i.e., the
    output of a forward transform of the space.

    The results dictionary is like::

        results = {'space': T,
                   'data': {
                       'U': [U, (U, [slice(None), slice(None), 0])],
                       'V': [V, (V, [slice(None), 0, slice(None)])],
                       }
                   }

    The results will be stored as scalars, even if U and V are vectors. Results
    are store for physical space, i.e., the input to a forward transform of the
    space.

    """

    def __init__(self, filename, checkpoint={}, results={}):
        self.cfile = None
        self.wfile = None
        self.filename = filename
        self.checkpoint = checkpoint
        self.results = results

    def update(self, params, **kw):
        # Here, create the 'NS_isotropic_..._c.h5' file which contains the checkpoint
        if self.cfile is None:
            self.cfile = ShenfunFile(self.filename+'_c',
                                     self.checkpoint['space'],
                                     mode=params.filemode)
            self.cfile.open()
            self.cfile.f.attrs.create('tstep', 0)
            self.cfile.f.attrs.create('t', 0.0)
            self.cfile.close()

        # Here, create the 'NS_isotropic_..._w.h5' file which contains the results
        if self.wfile is None:
            self.wfile = ShenfunFile(self.filename+'_w',
                                     self.results['space'],
                                     mode=params.filemode)

        # added 'or abs(params.t - params.T) < 1e-12' to write the last time step
        if params.tstep % params.write_result == 0 or abs(params.t - params.T) < 1e-12:
            self.update_components(**kw)
            self.wfile.write(params.tstep, self.results['data'], as_scalar=True)

        kill = self.check_if_kill()
        # added 'or abs(params.t - params.T) < 1e-12' to write the last time step
        if params.tstep % params.checkpoint == 0 or abs(params.t - params.T) < 1e-12 or kill:
            for key, val in self.checkpoint['data'].items():
                self.cfile.write(int(key), val)
                self.cfile.open()
                self.cfile.f.attrs['tstep'] = params.tstep
                self.cfile.f.attrs['t'] = params.t
                self.cfile.close()

            if kill:
                sys.exit(1)

    def update_components(self, **kw):
        pass

    def open(self):
        self.cfile.open()
        self.wfile.open()

    def close(self):
        if self.cfile.f:
            self.cfile.close()
        if self.wfile.f:
            self.wfile.close()

    @staticmethod
    def check_if_kill():
        """Check if user has put a file named killspectraldns in running folder."""
        found = 0
        if 'killspectraldns' in os.listdir(os.getcwd()):
            found = 1
        collective = comm.allreduce(found)
        if collective > 0:
            if comm.Get_rank() == 0:
                os.remove('killspectraldns')
                print('killspectraldns Found! Stopping simulations cleanly by checkpointing...')
            return True
        else:
            return False
