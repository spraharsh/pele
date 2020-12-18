#ifndef _CVODE_PETSC_MN_H
#define _CVODE_PETSC_MN_H


/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/
#include <petscksp.h>
#include <petscmat.h>


#include <petscvec.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <cvode/cvode.h>
#include <petscsnes.h>

/* TODO remove this*/
#include "/home/praharsh/Dropbox/research/bv-libraries/sundials/src/cvode/cvode_impl.h"

/*****************************************************************************/
/*                                   General structs                         */
/*****************************************************************************/

/* User supplied Jacobian function prototype */
typedef PetscErrorCode (*CVSNESJacFn)(PetscReal t, Vec x, Mat J,
                                      void *user_data);

/*
  context passed on to the delayed hessian
*/
typedef struct {
  /* memory information */
  void *cvode_mem; /* cvode memory */

  /* TODO: remove this since user data is in cvode_mem */
  void *user_mem; /* user data */

    /* TODO */
  CVSNESJacFn user_jac_func; /* user defined Jacobian function */  

  /* jacobian calculation information */
  booleantype jok;   /* check for whether jacobian needs to be updated */
  booleantype *jcur; /* whether to use saved copy of jacobian */

  /* Linear solver, matrix and vector objects/pointers */
  /* NOTE: some of this might be uneccessary since it maybe stored
     in the KSP solver */
  Mat savedJ;                /* savedJ = old Jacobian                        */
  Vec ycur;                  /* CVODE current y vector in Newton Iteration   */
  Vec fcur;                  /* fcur = f(tn, ycur)                           */

  booleantype
      scalesol; /* exposed to user (Check delayed matrix versions later)*/
} * CVMNPETScMem;


/*****************************************************************************/
/*                           function declarations                           */
/*****************************************************************************/


/* Main Jacobian to be passed to SNES for delayed jacobian calculation */
PetscErrorCode CVDelayedJSNES(SNES snes, Vec X, Mat A, Mat Jpre, void *context);

/* presolve function for the KSP solver KSPPreOnly could do this for the conditioner */
PetscErrorCode cvLSPresolveKSP(KSP ksp, Vec b, Vec x, void *ctx);

/* postsolve function for the KSP solver */
PetscErrorCode cvLSPostSolveKSP(KSP ksp, Vec b, Vec x, void *context);


/* Memory setup for modified newton */
CVMNPETScMem CVODEMNPETScCreate(void *cvode_mem, void *user_mem,
                                CVSNESJacFn func, booleantype scalesol, Mat Jac, Vec y);


PetscErrorCode CVODEMNPTScFree(CVMNPETScMem *cvmnpetscmem);






















      

#endif

