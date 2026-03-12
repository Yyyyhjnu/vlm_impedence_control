#include <iostream>
#include <cmath>
#include <fstream>
#include "Impedence_Controller.h"
#include "Matrix_C.h"
#include "Matrix_F.h"
#include "Matrix_M.h"
#include "Matrix_G.h"

#define SHM_NAME "force_shm"
#define SHM_SIZE 1000

const double PD_MAX_OUTPUT[6] = {30.0,30.0,30.0,15.0,15.0,30.0};

KDL::Chain Impedence_Controller::createSixDOFRobotChain() {
    KDL::Chain robot_chain;

    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(0.0, M_PI_2, 115.25e-3, 0.0)));
    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(-234e-3, 0.0, 0.0, -M_PI_2)));
    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(-206e-3, 0.0, 0.0, 0.0)));
    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(0.0, M_PI_2, 104e-3, -M_PI_2)));
    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(0.0, -M_PI_2, 100e-3, 0.0)));
    robot_chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ), KDL::Frame::DH(0.0, 0.0, 90.2e-3, 0.0)));

    return robot_chain;
}

void Impedence_Controller::setJointPositionsAndVelocities(KDL::JntArray& jointpositions, KDL::JntArray& jointvelocities, 
    const double* ActPosition, const double* ActVelocity) {
    for (int i = 0; i < 6; i++) {
        jointpositions(i) = ActPosition[i];
        jointvelocities(i) = ActVelocity[i];
    }
}

void Impedence_Controller::calculateEndEffectorPoseAndVelocity(const KDL::Chain& chain, const KDL::JntArray& jointpositions, 
    const KDL::JntArray& jointvelocities, double *end_effector_pose, double *end_effector_velocity) {
    // 正运动学求解器 (位置)
    KDL::ChainFkSolverPos_recursive fksolver_pos(chain);
    
    // 正速度学求解器 (速度)
    KDL::ChainFkSolverVel_recursive fksolver_vel(chain);
    
    // 使用 JntArrayVel 来封装关节角度和速度
    KDL::JntArrayVel jointposvel(jointpositions, jointvelocities);

    // 计算末端执行器的笛卡尔位置
    fksolver_pos.JntToCart(jointpositions, cartesian_pos);
    
    // 计算末端执行器的笛卡尔速度
    fksolver_vel.JntToCart(jointposvel, cartesian_vel);

    double roll, pitch, yaw;
    cartesian_pos.M.GetRPY(roll, pitch, yaw);  // 提取欧拉角

    end_effector_pose[0] = cartesian_pos.p.x();  
    end_effector_pose[1] = cartesian_pos.p.y();  
    end_effector_pose[2] = cartesian_pos.p.z();  
    end_effector_pose[3] = roll;   
    end_effector_pose[4] = pitch;  
    end_effector_pose[5] = yaw;    

    KDL::Twist twist = cartesian_vel.GetTwist();

    end_effector_velocity[0] = twist.vel.x();
    end_effector_velocity[1] = twist.vel.y();
    end_effector_velocity[2] = twist.vel.z();
    end_effector_velocity[3] = twist.rot.x();
    end_effector_velocity[4] = twist.rot.y();
    end_effector_velocity[5] = twist.rot.z();
}

Impedence_Controller::Impedence_Controller():adrc(20, 0.01) {

    force_shm = std::unique_ptr<SharedMemory>(new SharedMemory(SHM_NAME, SHM_SIZE));

    Manipulator = createSixDOFRobotChain();

    force_deadzone[0] = 0.5;
    force_deadzone[1] = 0.5;
    force_deadzone[2] = 0.5;
    force_deadzone[3] = 0.05;
    force_deadzone[4] = 0.05;
    force_deadzone[5] = 0.05;

    fd_base(0, 0) = 0.0;
    fd_base(1, 0) = 0.0;
    fd_base(2, 0) = 10.0;
    fd_base(3, 0) = 0.0;
    fd_base(4, 0) = 0.0;
    fd_base(5, 0) = 0.0;

    // Eigen::Matrix<double, 6, 1> Kp_diag;
    // Kp_diag << 2500, 3000, 5000, 4500, 3500, 2500;
    // Kp = Kp_diag.asDiagonal();

    // Eigen::Matrix<double, 6, 1> Kd_diag;
    // Kd_diag << 50, 60, 60, 60, 70, 70;
    // Kd = Kd_diag.asDiagonal();

    Eigen::Matrix<double, 6, 1> S_diag;
    S_diag << 0, 0, 1, 0, 0, 0;
    S = S_diag.asDiagonal();
}

Impedence_Controller::~Impedence_Controller() = default;

void Impedence_Controller::run(const double *ActPosition, const double *ActVelocity, double TargetTorque[6]) {
    cyc_count++;
    target_face_cartesian_trajectory.getTargetCartesianVelocity(dp_r);
    //target_face_cartesian_trajectory.getTargetCartesianAcceleration(ddp_r);

    KDL::JntArray jointpositions(6); 
    KDL::JntArray jointvelocities(6); 

    setJointPositionsAndVelocities(jointpositions, jointvelocities, ActPosition, ActVelocity);
    calculateEndEffectorPoseAndVelocity(Manipulator, jointpositions, jointvelocities, endEffectorPose, endEffectorVelocity);

    endEffectorPose_eigen = Eigen::Matrix<double, dof, 1>::Map(endEffectorPose, 6);//更新当前状态
    endEffectorVelocity_eigen = Eigen::Matrix<double, dof, 1>::Map(endEffectorVelocity, 6);//更新当前状态

    double f_ext[6] = {0, 0, 0, 0, 0, 0};// f_ext_filt[6] = {0, 0, 0, 0, 0, 0};
    force_shm->readData(f_ext);
    //force_filter.filt(f_ext, f_ext_filt);
    //std::cout << f_ext[2] << std::endl;

    // if (f_ext[2] < -30)
    // {
    //     f_ext[2] = -30;
    // }
    
    // if (f_ext[2] > 5)
    // {
    //     f_ext[2] = 5;
    // }

    if (f_ext[2]  > 1000  || f_ext[2] < -1000)  //安全措施
    {
        f_ext[2] = -10;
    }

    for (size_t i = 0; i < 6; i++)
    {
        delta_force[i] = f_ext[i];
        if (delta_force[i] > force_deadzone[i]) {
            delta_force[i] -= force_deadzone[i];
        } else if (delta_force[i] < -force_deadzone[i]) {
            delta_force[i] += force_deadzone[i];
        } else {
            delta_force[i] = 0;
        }
        f_ext_record[i] = delta_force[i];
    }

    Eigen::Matrix<double, 6, 1> fext_cartesian_eigen = Eigen::Matrix<double, 6, 1>::Map(delta_force, 6);
    Eigen::Matrix<double, 6, 1> ActPosition_eigen = Eigen::Matrix<double, 6, 1>::Map(ActPosition, 6);
    Eigen::Matrix<double, 6, 1> ActVelocity_eigen = Eigen::Matrix<double, 6, 1>::Map(ActVelocity, 6);

    if (init_flag)
    {   
        q_r_new = ActPosition_eigen;
        init_flag = false;
    }

    Eigen::Matrix3d eigen_rotation = Eigen::Map<Eigen::Matrix3d>(cartesian_pos.M.data);
    eigen_rotation.transposeInPlace();
    Eigen::Matrix<double, 6, 1> fext_base_eigen;
    Eigen::Matrix<double, 3, 3> p_base_end;

    p_base_end(0, 0) = 0;
    p_base_end(1, 1) = 0;
    p_base_end(2, 2) = 0;
    p_base_end(0, 1) = -cartesian_pos.p.data[2];
    p_base_end(0, 2) = cartesian_pos.p.data[1];
    p_base_end(1, 0) = cartesian_pos.p.data[2];
    p_base_end(1, 2) = -cartesian_pos.p.data[0];
    p_base_end(2, 0) = -cartesian_pos.p.data[1];
    p_base_end(2, 1) = cartesian_pos.p.data[0];

    fext_base_eigen.head<3>() = eigen_rotation * fext_cartesian_eigen.head<3>();
    fext_base_eigen.tail<3>() = p_base_end * eigen_rotation * fext_cartesian_eigen.head<3>() + eigen_rotation * fext_cartesian_eigen.tail<3>();

    fext_base_eigen = S * fext_base_eigen;

    KDL::ChainJntToJacSolver jac_solver(Manipulator);
    KDL::ChainJntToJacDotSolver jac_dot_solver(Manipulator);
    KDL::Jacobian J(Manipulator.getNrOfJoints());
    jac_solver.JntToJac(jointpositions, J);
    KDL::Jacobian J_dot(Manipulator.getNrOfJoints());
    jac_dot_solver.JntToJacDot(KDL::JntArrayVel(jointpositions, jointvelocities), J_dot);

    //Eigen::Matrix<double, 6, 6> J_inverse, J_inverse_effect;
    Eigen::Matrix<double, 6, 6> J_inverse;
    if (std::abs(J.data.determinant()) > 1e-6) {
        J_inverse = J.data.inverse();
    } else {
        std::cout << "雅克比矩阵不可逆。" << std::endl;
    }

    // if (cyc_count > 19752 && cyc_count <= 20252) {;
    //     fd_base(2, 0) = 0.0;
    // } else if(cyc_count > 20252) {
    //     fd_base(2, 0) = -2.0;
    // }

    adaptive_damping_controller.cal_adaptive_new_target(fext_base_eigen, fd_base, dp_r_new_delta);

    dq_r = J_inverse * dp_r;
    dq_r_new_delta = J_inverse * dp_r_new_delta;
    dq_r_new = dq_r + dq_r_new_delta;
    q_r_new = q_r_new + dq_r_new * dt;

    adrc.run(ActPosition_eigen, ActVelocity_eigen, q_r_new, tau);

    std::memcpy(TargetTorque, tau.data(), 6 * sizeof(double));
}
