#ifndef ADAPTIVE_DAMPING_CONTROLLER_H
#define ADAPTIVE_DAMPING_CONTROLLER_H

#include <memory>
#include <Eigen/Dense>

class Adaptive_damping_controller {

public:
    double T = 0.002; 
    Adaptive_damping_controller();
    virtual ~Adaptive_damping_controller() = default;

    void cal_adaptive_new_target(const Eigen::Matrix<double, 6, 1> &ActualBaseForce, const Eigen::Matrix<double, 6, 1> &TargetBaseForce, Eigen::Matrix<double, 6, 1> &TargetBaseVelocity_new_delta);
    void transition_sigmod_cal(const Eigen::Matrix<double, 6, 1> &actual_force, const Eigen::Matrix<double, 6, 1> &target_force, double &ktrans);
    Eigen::Matrix<double, 6, 1> Fai;
private:
    double sigma;
    Eigen::Matrix<double, 6, 1> F_e_prev;
    Eigen::Matrix<double, 6, 1> F_d_prev;
    Eigen::Matrix<double, 6, 1> F_e;
    Eigen::Matrix<double, 6, 1> F_d;
    Eigen::Matrix<double, 6, 6> Mv;
    Eigen::Matrix<double, 6, 6> Bv;
    Eigen::Matrix<double, 6, 1> Fai_prev;
    Eigen::Matrix<double, 6, 1> de_prev;
    Eigen::Matrix<double, 6, 1> de;
    
    Eigen::Matrix<double, 6, 6> M_B_inv;
    Eigen::Matrix<double, 6, 6> B_inv;
};

#endif //ADAPTIVE_DAMPING_CONTROL_H



