#ifndef MPC_IMPEDENCE_CONTROLLER_H
#define MPC_IMPEDENCE_CONTROLLER_H

#include <memory>
#include <Eigen/Dense>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntarrayvel.hpp>
#include <kdl/frames.hpp>
#include "shared_memory.h"
#include "compile_switch.h"
#include "Target_Face_Cartesian_Trajectory.h"
#include "forward_smooth_filter.h"
#include "Adaptive_damping_controller.h"
#include "ADRC.h"

class Impedence_Controller {
public:
    Impedence_Controller();
    virtual ~Impedence_Controller();
    void run(const double *ActPosition, const double *ActVelocity, double TargetTorque[6]);
    friend void saveFile(Impedence_Controller &ic);

private:
    long int cyc_count = 0;
    bool init_flag =  true;
    const double dt = 0.002;
    double f_ext_record[6];
    Eigen::Matrix<double, dof, 1> endEffectorPose_eigen;
    Eigen::Matrix<double, dof, 1> endEffectorVelocity_eigen;

    Eigen::Matrix<double, dof, 1> u;
    Eigen::Matrix<double, dof, 1> v;
    Eigen::Matrix<double, dof, 1> tau;
    Eigen::Matrix<double, dof, 1> tau_pd;
    Eigen::Matrix<double, dof, 1> ddp_r;
    Eigen::Matrix<double, dof, 1> ddp_r_new;
    Eigen::Matrix<double, dof, 1> ddp_r_new_delta;
    Eigen::Matrix<double, dof, 1> dp_r;
    Eigen::Matrix<double, dof, 1> de_k;
    Eigen::Matrix<double, dof, 1> de_k_prev;

    Eigen::Matrix<double, dof, 1> tau_f;
    Eigen::Matrix<double, dof, 1> dq_r;
    Eigen::Matrix<double, dof, 1> dq_r_new_delta;
    Eigen::Matrix<double, dof, 1> ddq_r_new_delta;
    
    Eigen::Matrix<double, dof, 1> ddq_r;
    Eigen::Matrix<double, dof, 1> dq_r_new;
    Eigen::Matrix<double, dof, 1> q_r_new;

    Eigen::Matrix<double, dof, 1> dp_r_new_delta;
    Eigen::Matrix<double, dof, 1> dp_r_new_delta_prev;

    Eigen::Matrix<double, dof, dof> Kp;
    Eigen::Matrix<double, dof, dof> Kd;
    Eigen::Matrix<double, dof, dof> S;

    Eigen::Matrix<double, dof, 1> fd_base;

    KDL::Chain Manipulator;

    // 存储末端执行器的笛卡尔位置
    KDL::Frame cartesian_pos;
    
    // 存储末端执行器的笛卡尔速度
    KDL::FrameVel cartesian_vel;

    std::unique_ptr<SharedMemory> force_shm;

    double delta_force[6];
    double force_deadzone[6];

    double endEffectorPose[6];
    double endEffectorVelocity[6];
    
    Adaptive_damping_controller adaptive_damping_controller;

    Target_Face_Cartesian_Trajectory target_face_cartesian_trajectory;

    Forward_Smooth_Filter force_filter;

    ADRC adrc;

    KDL::Chain createSixDOFRobotChain();
    void setJointPositionsAndVelocities(KDL::JntArray& jointpositions, KDL::JntArray& jointvelocities, const double* ActPosition, const double* ActVelocity);
    void calculateEndEffectorPoseAndVelocity(const KDL::Chain& chain, const KDL::JntArray& jointpositions, const KDL::JntArray& jointvelocities, double *end_effector_pose, double *end_effector_velocity);
};

#endif // MPC_IMPEDENCE_CONTROLLER_H
