#include "Adaptive_damping_controller.h"
#include <cmath>

Adaptive_damping_controller::Adaptive_damping_controller():sigma(0.005),
    F_e_prev(Eigen::Matrix<double, 6, 1>::Zero()),
    F_d_prev(Eigen::Matrix<double, 6, 1>::Zero()),
    Fai_prev(Eigen::Matrix<double, 6, 1>::Zero()),
    de_prev(Eigen::Matrix<double, 6, 1>::Zero()) {
        
    Eigen::Matrix<double, 6, 1> Mv_diag;
    Mv_diag << 3.0, 3.0, 1.0, 1.0, 1.0, 1.0;
    Mv = Mv_diag.asDiagonal(); 

    Eigen::Matrix<double, 6, 1> Bv_diag;
    Bv_diag << 1700.0, 1700.0, 1500.0, 50.0, 50.0, 50.0;
    Bv = Bv_diag.asDiagonal();

    M_B_inv = (Mv + Bv * T).inverse();
    B_inv = Bv.inverse();
}

void Adaptive_damping_controller::cal_adaptive_new_target(const Eigen::Matrix<double, 6, 1> &ActualBaseForce, const Eigen::Matrix<double, 6, 1> &TargetBaseForce, Eigen::Matrix<double, 6, 1> &TargetBaseVelocity_new_delta) {
    F_e = ActualBaseForce;
    F_d = TargetBaseForce;

    de = T * M_B_inv * (F_e - F_d) + M_B_inv * Mv * de_prev - T * M_B_inv * Bv * Fai_prev + sigma * T * M_B_inv * (F_e_prev - F_d_prev);
    
    Fai = Fai_prev - sigma * B_inv * (F_e_prev - F_d_prev);
    
    //double kt;
    //transition_sigmod_cal(F_e, F_d, kt);
    //Fai = kt * Fai;
    
    de_prev = de;
    Fai_prev = Fai;
    F_d_prev = F_d;
    F_e_prev = F_e;

    TargetBaseVelocity_new_delta = de;
}

void Adaptive_damping_controller::transition_sigmod_cal(const Eigen::Matrix<double, 6, 1> &actual_force, const Eigen::Matrix<double, 6, 1> &target_force, double &ktrans)
{
    double fl;
    ktrans = 0.0;
    if(fabs(target_force(2)) >= 5)
    {
        fl = 0.7 * target_force(2);
        // 计算过渡段参数
        ktrans = 1 / (1 + exp(-30/fabs(target_force(2))*(actual_force(2) - fl)));
        //printf("ktrans = %lf\n", ktrans);
    }
}