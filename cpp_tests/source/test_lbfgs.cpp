#include "pele/lj.hpp"
#include "pele/atlj.hpp"
#include "pele/lbfgs.hpp"
#include "pele/mxopt.hpp"
#include "pele/rosenbrock.hpp"
#include "pele/harmonic.hpp"
#include "pele/cvode.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include "pele/gradient_descent.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "pele/inversepower.hpp"
using pele::Array;
using std::cout;

/**
 * Gets the cell scale for tests in c++. NOTE: this needs to be tested for all cases
 */
double get_ncellx_scale(pele::Array<double> radii, pele::Array<double> boxv,
                        size_t omp_threads) {
    double ndim = boxv.size();
    double ncellsx_max = std::max<size_t>(omp_threads, pow(radii.size(), 1/ndim));
    double rcut = radii.get_max()*2;
    size_t ncellsx= boxv[0]/rcut;
    int ncellsx_scale;
    if (ncellsx<=ncellsx_max) {
        if (ncellsx>= omp_threads) {
            ncellsx_scale = 1;
        }
        else {
            ncellsx_scale = ceil(omp_threads/ncellsx);
        }
    }
    else {
        ncellsx_scale = ncellsx_max/ncellsx;
    }
    return ncellsx_scale;
}



TEST(LbfgsLJ, TwoAtom_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    std::cout << x0 << "\n";
    x0[0] = 2.;
    pele::LBFGS lbfgs(lj, x0);
    lbfgs.run();
    ASSERT_GT(lbfgs.get_nfev(), 1);
    ASSERT_GT(lbfgs.get_niter(), 1);
    ASSERT_LT(lbfgs.get_rms(), 1e-4);
    ASSERT_LT(lbfgs.get_rms(), 1e-4);
    ASSERT_NEAR(lbfgs.get_f(), -.25, 1e-10);
    Array<double> x = lbfgs.get_x();
    double dr, dr2 = 0;
    for (size_t i = 0; i < 3; ++i){
        dr = (x[i] - x[3+i]);
        dr2 += dr * dr;
    }
    dr = sqrt(dr2);
    ASSERT_NEAR(dr, pow(2., 1./6), 1e-5);
    Array<double> g = lbfgs.get_g();
    ASSERT_NEAR(g[0], -g[3], 1e-10);
    ASSERT_NEAR(g[1], -g[4], 1e-10);
    ASSERT_NEAR(g[2], -g[5], 1e-10);
    double rms = pele::norm(g) / sqrt(g.size());
    ASSERT_NEAR(rms, lbfgs.get_rms(), 1e-10);
}

TEST(LbfgsLJ, Reset_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    x0[0] = 2.;
    // lbfgs1 will minimize straight from x0
    pele::LBFGS lbfgs1(lj, x0);
    lbfgs1.run();

    // lbfgs2 will first minimize from x2 (!=x0) then reset from x0
    // it should end up exactly the same as lbfgs1
    Array<double> x2 = x0.copy();
    x2[1] = 2;
    pele::LBFGS lbfgs2(lj, x2);
    double H0 = lbfgs2.get_H0();
    lbfgs2.run();
    // now reset from x0
    lbfgs2.reset(x0);
    lbfgs2.set_H0(H0);
    lbfgs2.run();

    cout << lbfgs1.get_x() << "\n";
    cout << lbfgs2.get_x() << "\n";

    ASSERT_EQ(lbfgs1.get_nfev(), lbfgs2.get_nfev());
    ASSERT_EQ(lbfgs1.get_niter(), lbfgs2.get_niter());

    for (size_t i=0; i<x0.size(); ++i){
        ASSERT_DOUBLE_EQ(lbfgs1.get_x()[i], lbfgs2.get_x()[i]);
    }
    ASSERT_DOUBLE_EQ(lbfgs1.get_f(), lbfgs2.get_f());
    ASSERT_DOUBLE_EQ(lbfgs1.get_rms(), lbfgs2.get_rms());
//    ASSERT_EQ(lbfgs1.get_niter(), lbfgs1.get_niter());
//    ASSERT_GT(lbfgs.get_niter(), 1);
//    ASSERT_LT(lbfgs.get_rms(), 1e-4);
//    ASSERT_LT(lbfgs.get_rms(), 1e-4);
//    ASSERT_NEAR(lbfgs.get_f(), -.25, 1e-10);


}

TEST(LbfgsLJ, SetFuncGradientWorks){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    x0[0] = 2.;
    pele::LBFGS lbfgs1(lj, x0);
    pele::LBFGS lbfgs2(lj, x0);
    auto grad = x0.copy();
    double e = lj->get_energy_gradient(x0, grad);

    // set the gradient for  lbfgs2.  It should have the same result, but
    // one fewer function evaluation.
    lbfgs2.set_func_gradient(e, grad);
    lbfgs1.run();
    lbfgs2.run();
    ASSERT_EQ(lbfgs1.get_nfev(), lbfgs2.get_nfev() + 1);
    ASSERT_EQ(lbfgs1.get_niter(), lbfgs2.get_niter());
    ASSERT_DOUBLE_EQ(lbfgs1.get_f(), lbfgs2.get_f());
}


TEST(LbfgsRosenbrock, Rosebrock_works){
    auto rosenbrock = std::make_shared<pele::RosenBrock> ();
    Array<double> x0(2, 0);
    pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
    // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
    // pele ::GradientDescent lbfgs(rosenbrock, x0);
    lbfgs.run(2000);
    Array<double> x = lbfgs.get_x();
    std::cout << x << "\n";
    cout << lbfgs.get_nfev() << " get_nfev() \n";
    cout << lbfgs.get_niter() << " get_niter() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    std::cout << lbfgs.get_nhev() << "get nhev() \n";
    std::cout << x0 << "\n" << " \n";
    std::cout << x << "\n";
    std::cout << "this is okay" << "\n";
}




TEST(Hm, Hm_works){
    Array<double> origin(3, 0);    
    auto rosenbrock = std::make_shared<pele::Harmonic> (origin, 1, 3);
    Array<double> x0(3, 1);
    pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
    // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
    // pele ::GradientDescent lbfgs(rosenbrock, x0);
    lbfgs.run(2000);
    Array<double> x = lbfgs.get_x();
    std::cout << x << "\n";
    cout << lbfgs.get_nfev() << " get_nfev() \n";
    cout << lbfgs.get_niter() << " get_niter() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    std::cout << lbfgs.get_nhev() << "get nhev() \n";
    std::cout << x0 << "\n" << " \n";
    std::cout << x << "\n";
    std::cout << "this is okay" << "\n";
}

TEST(CVST, CVODESolverWorksIP){
    static const size_t _ndim = 3;
    size_t nr_particles;
    size_t nr_dof;
    double eps;
    double sca;
    double rsca;
    double energy;
    //////////// parameters for inverse power potential at packing fraction 0.7 in 3d
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
    eps =1.0;
    double box_length = 7.240260952504541;
    boxvec = {box_length, box_length, box_length} ;
    radii = {1.08820262, 1.02000786, 1.0489369,
        1.11204466, 1.0933779,  0.95113611,
        1.04750442, 0.99243214, 0.99483906,
        1.02052993, 1.00720218, 1.07271368,
        1.03805189, 1.00608375, 1.02219316,
        1.01668372, 1.50458554, 1.38563892,
        1.42191474, 1.3402133,  1.22129071,
        1.4457533 , 1.46051053, 1.34804845,
        1.55888282, 1.2981944,  1.4032031,
        1.38689713, 1.50729455, 1.50285511,
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

    double ncellsx_scale = get_ncellx_scale(radii, boxvec,
                                            1);
    std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot = std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii, boxvec);

    
    pele::CVODEBDFOptimizer optimizer(potcell, x);
    optimizer.run(1000);
    std::cout << optimizer.get_nfev() << "nfev \n";
    std::cout << optimizer.get_nhev() << "nhev \n";
    std::cout << optimizer.get_rms() << "rms \n";
    std::cout << optimizer.get_niter() << "\n";

}


// TEST(LbfgsSaddle, Saddle_works){
//     auto saddle = std::make_shared<pele::Saddle> ();
//     Array<double> x0(2, 0);
//     // Start from a point that ends on a saddle
//     x0[0] = 1.;
//     x0[1] = 0;
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }


// TEST(LbfgsATLJ, heavyconcaveworks){
//     auto saddle = std::make_shared<pele::ATLJ> (1., 1., 1.);
//     Array<double> x0 = {0.97303831232970894, 0, 0, 0, 0, 0, -1.8451195244938898, 0.66954397800263088, 0};
//     // Start from a point that ends on a saddle
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }



// test(LbfgsATLJ, why){
//     auto saddle = std::make_shared<pele::ATLJ> (1., 1., 2.);
//     Array<double> x0 = { 1.1596938257996146
//                          , 0.92275271036961359
//                          , 0
//                          , -1.03917561305288
//                          , 0.60339721865003093
//                          , 0
//                          , 0.11585897591687914
//                          , 0.5349653769730589
//                          , 0};
//     // Start from a point that ends on a saddle
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }



// TEST(LbfgsSaddlexcubed, Saddle_works){
//     auto saddlex3 = std::make_shared<pele::XCube> ();
//     Array<double> x0(1, 0);
//     // Start from a point that ends on a saddle
//     x0[0] = 1.;
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddlex3, x0, 1e-4, 1, 1);
//     lbfgs.run(30);
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }




