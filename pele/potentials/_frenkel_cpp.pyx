"""
# distutils: language = C++


Frenkel Potential as defined in https://arxiv.org/abs/1910.05746


"""
import numpy as np
cimport cython
cimport numpy as np
#import pele.potentials._pele as _pele
cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport BasePotential
from cpython cimport bool
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np
# note: it is required to explicitly import BasePotential.  The compilation
# will fail if you try to use it as _pele.BasePotential.  I don't know why this
# is






# use external c++ class
# cdef extern from "pele/_frenkel.h" namespace "pele":
#     cdef cppclass  cFrenkel "pele::Frenkel":
#         cFrenkel(double sig, double eps, double rcut) except +




cdef extern from "pele/frenkel.h" namespace "pele":
    cdef cppclass  cFrenkel "pele::Frenkel":
        cFrenkel(double sig, double eps, double rcut) except +
    cdef cppclass  cFrenkelPeriodic "pele::FrenkelPeriodic":
        cFrenkelPeriodic(double sig, double eps, double rcut, _pele.Array[double] boxvec) except +
    cdef cppclass  cFrenkelNeighborList "pele::FrenkelNeighborList":
        cFrenkelNeighborList(_pele.Array[size_t] & ilist, double sig, double eps, double rcut) except +

cdef class Frenkel(_pele.BasePotential):
    """define the python interface to the c++ Frenkel implementation
    """
    cpdef bool periodic
    def __cinit__(self,  sig=1.0, eps=1.0, rcut=2.0, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * 3
        cdef np.ndarray[double, ndim=1] bv
        if boxvec is None:
            self.periodic = False
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cFrenkel(sig, eps, rcut) )
        else:
            self.periodic = True
            bv = np.array(boxvec, dtype=float)
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cFrenkelPeriodic(sig, eps, rcut,
                                                              array_wrap_np(bv)) )





# cdef class Frenkel(_pele.BasePotential):
#     """define the python interface to the c++ Frenkel implementation
#     """
#     cpdef bool periodic
#     def __cinit__(self, eps=1.0, sig=1.0, boxvec=None, boxl=None):
#         assert not (boxvec is not None and boxl is not None)
#         if boxl is not None:
#             boxvec = [boxl] * 3
#         cdef np.ndarray[double, ndim=1] bv
#         if boxvec is None:
#             self.periodic = False
#             self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cFrenkel(sig, eps, rcut) )
#         else:
#             self.periodic = True
#             bv = np.array(boxvec, dtype=float)
#             self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cFrenkelPeriodic(sig, 4.*eps*sig**12,
#                                                               array_wrap_np(bv)) )

