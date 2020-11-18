#include "pele/cvode.hpp"
#include "autodiff/reverse/eigen.hpp"
#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "nvector/nvector_serial.h"
#include "pele/base_potential.hpp"
#include "pele/optimizer.hpp"
#include "pele/debug.hpp"

// cvode imports
#include "petscsnes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_dense.h"
#include <nvector/nvector_petsc.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <petscsys.h>
#include <sys/types.h>
// #include <petscmat.h>

namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                                     const pele::Array<double> x0,
                                     double tol,
                                     double rtol,
                                     double atol)
    : GradientOptimizer(potential, x0, tol),
      cvode_mem(CVodeCreate(CV_BDF)), // create cvode memory
      N_size(x0.size()),
      t0(0),
      tN(10000000.0)
{
    // dummy t0
    // initialize petsc
    PetscInitializeNoArguments();
    // this assumes that the number of non zeros aren't different
    blocksize = 1;
    hessav = 10;
    // matrix initialization
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N_size, N_size, hessav, NULL, &petsc_hess);
    VecCreateSeq(PETSC_COMM_SELF, N_size, &petsc_grad);
    // wrap nvector into petsc array
    nvec_grad_petsc = N_VMake_Petsc(petsc_grad);
    VecDuplicate(petsc_grad, &residual);


    // SNES create
    SNESCreate(PETSC_COMM_SELF, &snes);
    


    // we need to pass the potential pointer on to the
    // Jacobian function as the application context
    // The potential pointer allows us to use the base potential class to calculate the Jacobian
    // of the gradient i.e the hessian
    udataptr = &udata;
    SNESSetJacobian(snes, petsc_hess, petsc_hess, hessian_wrapper,udataptr);
     
    
    
    

    
    
    
    
    
     double t0 = 0;
    std::cout << x0 << "\n";
    Array<double> x0copy = x0.copy();
    x0_N = N_Vector_eq_pele(x0copy);
    // initialization of everything CVODE needs
    int ret = CVodeInit(cvode_mem, f, t0, x0_N);
    
    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    udata.pot_ = potential_;
    udata.stored_grad = Array<double>(x0.size(), 0);
    CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    
    ret = CVodeSetUserData(cvode_mem, &udata);
    A = SUNDenseMatrix(N_size, N_size);
    LS = SUNLinSol_Dense(x0_N, A);
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac);
    g_ = udata.stored_grad;
    CVodeSetMaxNumSteps(cvode_mem, 1000000);
    CVodeSetStopTime(cvode_mem, tN);
};



void CVODEBDFOptimizer::one_iteration() {
    /* advance solver just one internal step */
    Array<double> xold = x_;
    int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
    iter_number_ +=1;
    x_ = pele_eq_N_Vector(x0_N);
    g_ = udata.stored_grad;
    rms_ = (norm(g_)/sqrt(x_.size()));
    f_ = udata.stored_energy;
    nfev_ = udata.nfev;
    Array<double> step = xold-x_;

};

int f(double t, N_Vector y, N_Vector ydot, void *user_data) {

    UserData udata = (UserData) user_data;
    pele::Array<double> yw = pele_eq_N_Vector(y);

    
    Array<double> g;
    // double energy = udata->pot_->get_energy_gradient(yw, g);
    double *fdata = NV_DATA_S(ydot);
    g = Array<double>(fdata, NV_LENGTH_S(ydot));

    // calculate negative grad g
    double energy = udata->pot_->get_energy_gradient(yw, g);
    udata->nfev += 1;
#pragma simd
    for (size_t i = 0; i < yw.size(); ++i) {
        fdata[i] = -fdata[i];
    }
    udata->stored_grad = (g);
    udata->stored_energy = energy;
    return 0;
}

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    UserData udata = (UserData) user_data;

    pele::Array<double> yw = pele_eq_N_Vector(y);
    Array<double> g = Array<double>(yw.size());
    Array<double> h = Array<double>(yw.size()*yw.size());
    udata->pot_->get_energy_gradient_hessian(pele_eq_N_Vector(y), g, h);
    udata->nhev += 1;
    double * hessdata = SUNDenseMatrix_Data(J);
    for (size_t i=0; i<h.size(); ++i) {
        hessdata[i] = -h[i];
    }
    return 0;
};



PetscErrorCode hessian_wrapper(SNES NLS,Vec x,  Mat Amat, Mat Precon, void* user_data)
{
    UserData udata = (UserData) user_data;
    // wrap petsc data into pele vec
    Array<double> x_pele = pele_eq_PetscVec(x);
    x_pele = pele_eq_PetscVec(x);
    Vec dummy;
    udata->pot_->get_energy_gradient_hessian_sparse(x_pele, dummy, Amat);
    return 0;
};


} // namespace pele