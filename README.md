# spectralDNS
The backcoupled Fibre/DNS solver is implemented in demo/2waymultiFibreDNS.py use -h flag to parameters.
Before running it make sure:
1. Pythonpath is set to src (export PYTHONPATH='path/to/src')
2. Create and activate conda environment conda create --name spectralDNS -c conda-forge shenfun mpi4py-fft cython numba pythran mpich pip h5py=*=mpi*

Backcoupling is implemented for the slab (default) decomposition and for Navier Stokes equation only.
To run on multiple cores type mpirun -np #cores python 2waymultiFibreDNS.py --your_arguments NS.
If fibres should be free on both ends, set --no_spider True
