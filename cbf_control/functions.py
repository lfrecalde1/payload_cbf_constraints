import casadi as ca
import numpy as np
from dual_quaternion import DualQuaternion
from casadi import Function
from dual_quaternion import Quaternion

# Auxiliar values to create the dual quaternion
qw_1_aux = ca.MX.sym('qw_1_aux', 1, 1)
qx_1_aux = ca.MX.sym('qx_1_aux', 1, 1)
qy_1_aux = ca.MX.sym('qy_1_aux', 1, 1)
qz_1_aux = ca.MX.sym('qz_1_aux', 1, 1)
q_1_aux = ca.vertcat(qw_1_aux, qx_1_aux, qy_1_aux, qz_1_aux)
tx_1_aux = ca.MX.sym("tx_1_aux", 1, 1)
ty_1_aux = ca.MX.sym("ty_1_aux", 1, 1)
tz_1_aux = ca.MX.sym("tz_1_aux", 1, 1)
t_1_aux = ca.vertcat(0.0, tx_1_aux, ty_1_aux, tz_1_aux)

# DualQuaternion from axis and position
Q1_pose =  DualQuaternion.from_pose(quat = q_1_aux, trans = t_1_aux)

qw_1 = ca.MX.sym('qw_1', 1, 1)
qx_1 = ca.MX.sym('qx_1', 1, 1)
qy_1 = ca.MX.sym('qy_1', 1, 1)
qz_1 = ca.MX.sym('qz_1', 1, 1)
q_1 = ca.vertcat(qw_1, qx_1, qy_1, qz_1)
dw_1 = ca.MX.sym("dw_1", 1, 1)
dx_1 = ca.MX.sym("dx_1", 1, 1)
dy_1 = ca.MX.sym("dy_1", 1, 1)
dz_1 = ca.MX.sym("dz_1", 1, 1)
d_1 = ca.vertcat(dw_1, dx_1, dy_1, dz_1)

# Auxiliar Transformation Link2
aux_2_1 = -np.pi/2
n_2_1 = ca.MX([0.0, 0.0, 1.0])
q_2_1 = ca.vertcat(ca.cos(aux_2_1/2), ca.sin(aux_2_1/2)@n_2_1)
t_2_1 = ca.MX([0.0, 0.0, 0.0, -0.2])

aux_2_2 = -np.pi
n_2_2 = ca.MX([1.0, 0.0, 0.0])
q_2_2 = ca.vertcat(ca.cos(aux_2_2/2), ca.sin(aux_2_2/2)@n_2_2)
t_2_2 = ca.MX([0.0, 0.0, 0.0, 0.0])

theta_2 = ca.MX.sym('theta_2', 1, 1)
n_2 = ca.MX([0.0, 0.0, 1.0])
q_2 = ca.vertcat(ca.cos(theta_2/2), ca.sin(theta_2/2)@n_2)
t_2 = ca.MX([0.0, 0.0, 0.0, 0.0])

Q2_1_pose =  DualQuaternion.from_pose(quat = q_2_1, trans = t_2_1)
Q2_2_pose =  DualQuaternion.from_pose(quat = q_2_2, trans = t_2_2)
Q2_pose =  DualQuaternion.from_pose(quat = q_2, trans = t_2)

# Creating states of the current dualquaternion
Q1 = DualQuaternion(q_real= Quaternion(q = q_1), q_dual = Quaternion(q = d_1))


def dualquat_from_pose_casadi():
    # Compute the Pose based of the quaternion and the trasnation
    values = Q1_pose.get[:, 0]
    f_pose = Function('f_pose', [qw_1_aux, qx_1_aux, qy_1_aux, qz_1_aux, tx_1_aux, ty_1_aux, tz_1_aux], [values])
    return f_pose

def forward_kinematics_casadi_camera():
    # Compute the Pose based of the quaternion and the trasnation
    tranformation = Q1 *  Q2_1_pose * Q2_2_pose * Q2_pose
    values = tranformation.get[:, 0]
    f_pose = Function('f_pose', [q_1, d_1, theta_2], [values])
    return f_pose

def casadi_camera():
    # Compute the Pose based of the quaternion and the trasnation
    tranformation =  Q2_1_pose * Q2_2_pose * Q2_pose
    values = tranformation.get[:, 0]
    f_pose = Function('f_pose', [theta_2], [values])
    return f_pose

