
// TODO:: template the test class to avoid repetition
#include "pele/array.hpp"
#include "pele/atlj.hpp"
#include "pele/cvode.hpp"
#include "pele/gradient_descent.hpp"
#include "pele/harmonic.hpp"
#include "pele/inversepower.hpp"
#include "pele/lj.hpp"
#include "pele/rosenbrock.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using pele::Array;
using std::cout;

/**
 * Gets the cell scale for tests in c++. NOTE: this needs to be tested for all
 * cases
 */
double get_ncellx_scale_2(pele::Array<double> radii, pele::Array<double> boxv,
                          size_t omp_threads) {
  double ndim = boxv.size();
  double ncellsx_max =
      std::max<size_t>(omp_threads, pow(radii.size(), 1 / ndim));
  double rcut = radii.get_max() * 2;
  size_t ncellsx = boxv[0] / rcut;
  int ncellsx_scale;
  if (ncellsx <= ncellsx_max) {
    if (ncellsx >= omp_threads) {
      ncellsx_scale = 1;
    } else {
      ncellsx_scale = ceil(omp_threads / ncellsx);
    }
  } else {
    ncellsx_scale = ncellsx_max / ncellsx;
  }
  return ncellsx_scale;
}

TEST(CVODE, DenseSparseComparisonRosenBrock) {
  auto rosenbrock = std::make_shared<pele::RosenBrock>();
  Array<double> x0(2, 0);
  pele::CVODEBDFOptimizer cvode_sparse(rosenbrock, x0, 1e-5, 1e-4, 1e-4, true);
  pele::CVODEBDFOptimizer cvode_dense(rosenbrock, x0, 1e-5, 1e-4, 1e-4, false);
  // bird function
  // 
  cvode_sparse.run(2000);
  cvode_dense.run(2000);

  pele::Array<double> x_sparse = cvode_sparse.get_x();
  int nfev_sparse = cvode_sparse.get_nfev();
  int niter_sparse = cvode_sparse.get_niter();
  int rms_sparse = cvode_sparse.get_rms();
  int nhev_sparse = cvode_sparse.get_nhev();

  pele::Array<double> x_dense = cvode_dense.get_x();
  int nfev_dense = cvode_dense.get_nfev();
  int niter_dense = cvode_dense.get_niter();
  int rms_dense = cvode_dense.get_rms();
  int nhev_dense = cvode_dense.get_nhev();

  // std::cout << nfev_dense << "nfev_dense\n";
  // std::cout << nfev_sparse << "nfev_sparse\n";
  // std::cout << niter_sparse << "niter_sparse\n";
  // std::cout << niter_dense << "niter_dense\n";
  // std::cout << nhev_dense << "nhev_dense\n";
  // std::cout << nhev_sparse << "nhev_sparse\n";

  // check ensures the same path is being followed
  // unlikely that different minima will be reached if these
  // parameters are different if they reach different minima
  ASSERT_EQ(nfev_sparse, nfev_dense);
  ASSERT_EQ(nhev_sparse, nhev_dense);
  ASSERT_EQ(niter_sparse, niter_dense);
  ASSERT_EQ(rms_dense, rms_sparse);
}

TEST(CVODE, DenseSparseComparisonHarmonic) {
  Array<double> origin(3, 0);
  auto harmonic = std::make_shared<pele::Harmonic>(origin, 1, 3);
  Array<double> x0(3, 1);
  pele::CVODEBDFOptimizer cvode_sparse(harmonic, x0, 1e-5, 1e-4, 1e-4, true);
  pele::CVODEBDFOptimizer cvode_dense(harmonic, x0, 1e-5, 1e-4, 1e-4, false);

  cvode_sparse.run(2000);
  cvode_dense.run(2000);

  pele::Array<double> x_sparse = cvode_sparse.get_x();
  int nfev_sparse = cvode_sparse.get_nfev();
  int niter_sparse = cvode_sparse.get_niter();
  int rms_sparse = cvode_sparse.get_rms();
  int nhev_sparse = cvode_sparse.get_nhev();

  pele::Array<double> x_dense = cvode_dense.get_x();
  int nfev_dense = cvode_dense.get_nfev();
  int niter_dense = cvode_dense.get_niter();
  int rms_dense = cvode_dense.get_rms();
  int nhev_dense = cvode_dense.get_nhev();

  std::cout << nfev_dense << "nfev_dense\n";
  std::cout << nfev_sparse << "nfev_sparse\n";
  std::cout << niter_sparse << "niter_sparse\n";
  std::cout << niter_dense << "niter_dense\n";
  std::cout << nhev_dense << "nhev_dense\n";
  std::cout << nhev_sparse << "nhev_sparse\n";

  // check ensures the same path is being followed
  // unlikely that different minima will be reached if these
  // parameters are different if they reach different minima
  ASSERT_EQ(nfev_sparse, nfev_dense);
  ASSERT_EQ(nhev_sparse, nhev_dense);
  ASSERT_EQ(niter_sparse, niter_dense);
  ASSERT_EQ(rms_dense, rms_sparse);
}

TEST(CVODE, DenseSparseComparisonInversePower) {
  static const size_t _ndim = 3;
  size_t nr_particles;
  size_t nr_dof;
  double eps;
  double sca;
  double rsca;
  double energy;
  //////////// parameters for inverse power potential at packing fraction 0.7 in
  ///3d
  //////////// as generated from code params.py in basinerror library

  pele::Array<double> x;
  pele::Array<double> hs_radii;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;
  double power = 2.5;
  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  // phi = 0.7
  nr_particles = 32;
  nr_dof = nr_particles * _ndim;
  eps = 1.0;
  double box_length = 7.240260952504541;
  boxvec = {box_length, box_length, box_length};
  radii = {1.08820262, 1.02000786, 1.0489369,  1.11204466, 1.0933779,
           0.95113611, 1.04750442, 0.99243214, 0.99483906, 1.02052993,
           1.00720218, 1.07271368, 1.03805189, 1.00608375, 1.02219316,
           1.01668372, 1.50458554, 1.38563892, 1.42191474, 1.3402133,
           1.22129071, 1.4457533,  1.46051053, 1.34804845, 1.55888282,
           1.2981944,  1.4032031,  1.38689713, 1.50729455, 1.50285511,
           1.41084632, 1.42647138};
  x = {2.60293101, 3.16422539, 5.05103191, 0.43604813, 4.82756501, 4.85559318,
       1.52322464, 0.93346004, 2.28378357, 2.63336089, 4.12837341, 3.17558941,
       7.15608451, 0.73883106, 1.51232222, 1.167923,   4.72867471, 1.8338973,
       3.37621168, 1.76970507, 1.15098127, 0.79914482, 4.7519975,  1.00048063,
       1.4233076,  2.66966646, 5.94420522, 0.70303858, 6.06693979, 0.69577755,
       7.06982134, 3.393157,   7.07200517, 4.3792394,  5.35246123, 0.28372984,
       2.04759621, 0.87025447, 2.14413231, 0.85961967, 2.3022812,  2.99937218,
       0.46444461, 5.01367885, 4.10234238, 1.92148917, 3.78845245, 0.68015381,
       4.17000292, 6.72834697, 2.30652235, 4.83222531, 0.95425092, 5.18639589,
       2.09537563, 1.32635327, 4.2465067,  0.14558388, 6.00174213, 0.03399647,
       4.9075686,  1.95492819, 5.32299657, 6.96649615, 1.80103767, 4.17152945,
       4.28653808, 4.14325313, 1.61516923, 6.89815147, 3.23730442, 6.12821966,
       5.06441248, 2.15352114, 5.89210858, 2.87080503, 6.37941707, 4.20856728,
       6.38399411, 5.01410943, 5.25103024, 3.62971935, 6.92229501, 4.66265709,
       3.06882116, 4.39044511, 0.13896376, 2.18348037, 4.77982869, 2.10023757,
       4.47459298, 3.10439728, 0.98086758, 2.15964188, 4.12669469, 4.27807298};

  double ncellsx_scale = get_ncellx_scale_2(radii, boxvec, 1);
  std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
          power, eps, radii, boxvec, ncellsx_scale);

  pele::CVODEBDFOptimizer cvode_sparse(potcell, x, 1e-5, 1e-4, 1e-4, true);
  pele::CVODEBDFOptimizer cvode_dense(potcell, x, 1e-5, 1e-4, 1e-4, false);

  cvode_sparse.run(2000);
  cvode_dense.run(2000);

  pele::Array<double> x_sparse = cvode_sparse.get_x();
  int nfev_sparse = cvode_sparse.get_nfev();
  int niter_sparse = cvode_sparse.get_niter();
  int rms_sparse = cvode_sparse.get_rms();
  int nhev_sparse = cvode_sparse.get_nhev();

  pele::Array<double> x_dense = cvode_dense.get_x();
  int nfev_dense = cvode_dense.get_nfev();
  int niter_dense = cvode_dense.get_niter();
  int rms_dense = cvode_dense.get_rms();
  int nhev_dense = cvode_dense.get_nhev();

  std::cout << nfev_dense << "nfev_dense\n";
  std::cout << nfev_sparse << "nfev_sparse\n";
  std::cout << niter_sparse << "niter_sparse\n";
  std::cout << niter_dense << "niter_dense\n";
  std::cout << nhev_dense << "nhev_dense\n";
  std::cout << nhev_sparse << "nhev_sparse\n";

  // check ensures the same path is being followed
  // unlikely that different minima will be reached if these
  // parameters are different if they reach different minima
  ASSERT_EQ(nfev_sparse, nfev_dense);
  ASSERT_EQ(nhev_sparse, nhev_dense);
  ASSERT_EQ(niter_sparse, niter_dense);
  ASSERT_EQ(rms_dense, rms_sparse);
}



TEST(CVODE, DenseRun) {
  static const size_t _ndim = 3;
  size_t nr_particles;
  size_t nr_dof;
  double eps;
  double sca;
  double rsca;
  double energy;
  //////////// parameters for inverse power potential at packing fraction 0.7 in
  ///3d
  //////////// as generated from code params.py in basinerror library

  pele::Array<double> x;
  pele::Array<double> hs_radii;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;
  double power = 2.5;
  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  // phi = 0.7
  nr_particles = 32;
  nr_dof = nr_particles * _ndim;
  eps = 1.0;
  double box_length = 7.240260952504541;
  boxvec = {box_length, box_length, box_length};
  radii = {1.08820262, 1.02000786, 1.0489369,  1.11204466, 1.0933779,
           0.95113611, 1.04750442, 0.99243214, 0.99483906, 1.02052993,
           1.00720218, 1.07271368, 1.03805189, 1.00608375, 1.02219316,
           1.01668372, 1.50458554, 1.38563892, 1.42191474, 1.3402133,
           1.22129071, 1.4457533,  1.46051053, 1.34804845, 1.55888282,
           1.2981944,  1.4032031,  1.38689713, 1.50729455, 1.50285511,
           1.41084632, 1.42647138};
  x = {2.60293101, 3.16422539, 5.05103191, 0.43604813, 4.82756501, 4.85559318,
       1.52322464, 0.93346004, 2.28378357, 2.63336089, 4.12837341, 3.17558941,
       7.15608451, 0.73883106, 1.51232222, 1.167923,   4.72867471, 1.8338973,
       3.37621168, 1.76970507, 1.15098127, 0.79914482, 4.7519975,  1.00048063,
       1.4233076,  2.66966646, 5.94420522, 0.70303858, 6.06693979, 0.69577755,
       7.06982134, 3.393157,   7.07200517, 4.3792394,  5.35246123, 0.28372984,
       2.04759621, 0.87025447, 2.14413231, 0.85961967, 2.3022812,  2.99937218,
       0.46444461, 5.01367885, 4.10234238, 1.92148917, 3.78845245, 0.68015381,
       4.17000292, 6.72834697, 2.30652235, 4.83222531, 0.95425092, 5.18639589,
       2.09537563, 1.32635327, 4.2465067,  0.14558388, 6.00174213, 0.03399647,
       4.9075686,  1.95492819, 5.32299657, 6.96649615, 1.80103767, 4.17152945,
       4.28653808, 4.14325313, 1.61516923, 6.89815147, 3.23730442, 6.12821966,
       5.06441248, 2.15352114, 5.89210858, 2.87080503, 6.37941707, 4.20856728,
       6.38399411, 5.01410943, 5.25103024, 3.62971935, 6.92229501, 4.66265709,
       3.06882116, 4.39044511, 0.13896376, 2.18348037, 4.77982869, 2.10023757,
       4.47459298, 3.10439728, 0.98086758, 2.15964188, 4.12669469, 4.27807298};

  double ncellsx_scale = get_ncellx_scale_2(radii, boxvec, 1);
  std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
          power, eps, radii, boxvec, ncellsx_scale);
  pele::CVODEBDFOptimizer cvode_dense(potcell, x, 1e-6, 1e-4, 1e-4, false);


  cvode_dense.run(20000);



  pele::Array<double> x_dense = cvode_dense.get_x();
  int nfev_dense = cvode_dense.get_nfev();
  int niter_dense = cvode_dense.get_niter();
  int rms_dense = cvode_dense.get_rms();
  int nhev_dense = cvode_dense.get_nhev();
  
  std::cout << nfev_dense << "nfev_dense\n";
  std::cout << niter_dense << "niter_dense\n";
  std::cout << nhev_dense << "nhev_dense\n";
}

TEST(CVODE, SparseRun) {
  static const size_t _ndim = 3;
  size_t nr_particles;
  size_t nr_dof;
  double eps;
  double sca;
  double rsca;
  double energy;
  //////////// parameters for inverse power potential at packing fraction 0.7 in
  ///  //////////// as generated from code params.py in basinerror library

  pele::Array<double> x;
  pele::Array<double> hs_radii;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;
  double power = 2.5;
  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  // phi = 0.7
  nr_particles = 32;
  nr_dof = nr_particles * _ndim;
  eps = 1.0;
  double box_length = 7.240260952504541;
  boxvec = {box_length, box_length, box_length};
  radii = {1.08820262, 1.02000786, 1.0489369,  1.11204466, 1.0933779,
           0.95113611, 1.04750442, 0.99243214, 0.99483906, 1.02052993,
           1.00720218, 1.07271368, 1.03805189, 1.00608375, 1.02219316,
           1.01668372, 1.50458554, 1.38563892, 1.42191474, 1.3402133,
           1.22129071, 1.4457533,  1.46051053, 1.34804845, 1.55888282,
           1.2981944,  1.4032031,  1.38689713, 1.50729455, 1.50285511,
           1.41084632, 1.42647138};
  x = {2.60293101, 3.16422539, 5.05103191, 0.43604813, 4.82756501, 4.85559318,
       1.52322464, 0.93346004, 2.28378357, 2.63336089, 4.12837341, 3.17558941,
       7.15608451, 0.73883106, 1.51232222, 1.167923,   4.72867471, 1.8338973,
       3.37621168, 1.76970507, 1.15098127, 0.79914482, 4.7519975,  1.00048063,
       1.4233076,  2.66966646, 5.94420522, 0.70303858, 6.06693979, 0.69577755,
       7.06982134, 3.393157,   7.07200517, 4.3792394,  5.35246123, 0.28372984,
       2.04759621, 0.87025447, 2.14413231, 0.85961967, 2.3022812,  2.99937218,
       0.46444461, 5.01367885, 4.10234238, 1.92148917, 3.78845245, 0.68015381,
       4.17000292, 6.72834697, 2.30652235, 4.83222531, 0.95425092, 5.18639589,
       2.09537563, 1.32635327, 4.2465067,  0.14558388, 6.00174213, 0.03399647,
       4.9075686,  1.95492819, 5.32299657, 6.96649615, 1.80103767, 4.17152945,
       4.28653808, 4.14325313, 1.61516923, 6.89815147, 3.23730442, 6.12821966,
       5.06441248, 2.15352114, 5.89210858, 2.87080503, 6.37941707, 4.20856728,
       6.38399411, 5.01410943, 5.25103024, 3.62971935, 6.92229501, 4.66265709,
       3.06882116, 4.39044511, 0.13896376, 2.18348037, 4.77982869, 2.10023757,
       4.47459298, 3.10439728, 0.98086758, 2.15964188, 4.12669469, 4.27807298};

  double ncellsx_scale = get_ncellx_scale_2(radii, boxvec, 1);
  std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
          power, eps, radii, boxvec, ncellsx_scale);
  pele::CVODEBDFOptimizer cvode_sparse(potcell, x, 1e-6, 1e-4, 1e-4, true);
  cvode_sparse.run(20000);
  pele::Array<double> x_sparse = cvode_sparse.get_x();
  int nfev_sparse = cvode_sparse.get_nfev();
  int niter_sparse = cvode_sparse.get_niter();
  int rms_sparse = cvode_sparse.get_rms();
  int nhev_sparse = cvode_sparse.get_nhev();
  std::cout << nfev_sparse << "nfev_sparse\n";
  std::cout << niter_sparse << "niter_sparse\n";
  std::cout << nhev_sparse << "nhev_sparse\n";
}






















