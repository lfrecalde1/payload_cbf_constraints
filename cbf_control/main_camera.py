#!/usr/bin/env python3
# Import libraries
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from cbf_control import dualquat_from_pose_casadi
from cbf_control import forward_kinematics_casadi_camera
from cbf_control import casadi_camera
from cbf_control import dualquat_trans_casadi, dualquat_quat_casadi, rotation_casadi, rotation_inverse_casadi, dual_velocity_casadi, dual_quat_casadi, velocities_from_twist_casadi
from cbf_control import create_ocp_solver
from scipy.linalg import block_diag

import threading
import time

from nav_msgs.msg import Odometry
from acados_template import AcadosSimSolver, AcadosOcpSolver

from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs

from dual_quaternion import fancy_plots_3, plot_states_position_b_frame, plot_linear_velocities, plot_angular_velocities
from ament_index_python.packages import get_package_share_path
import os

# Creating Funtions based on Casadi
dualquat_from_pose = dualquat_from_pose_casadi()
get_trans = dualquat_trans_casadi()
get_quat = dualquat_quat_casadi()
dual_twist = dual_velocity_casadi()
velocity_from_twist = velocities_from_twist_casadi()
rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()

# Camera Pose
camera_pose_f = casadi_camera()
forward_camera_f = forward_kinematics_casadi_camera()
class CameraNode(Node):
    def __init__(self):
        super().__init__('CAMERA')
        # Parameters of the sytem
        self.initial = 5
        self.g = 9.81
        self.mQ = (1.0)

        # Inertia Matrix
        self.Jxx = 0.00305587
        self.Jyy = 0.00159695
        self.Jzz = 0.00159687
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

        # Lets define internal variables
        self.ts = 0.01
        self.t_final = 40
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)

        ## Prediction time 
        self.t_N = 0.5
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Initial States dual set zeros
        pos_0 = np.array([0.0, -1, 0.0], dtype=np.double)
        #pos_0 = np.array([0.0, -0.0, 0.0], dtype=np.double)
        theta_0 = np.pi/4
        n_0 = np.array([0.0, 0.0, 1.0])
        quat_0 = np.hstack([np.cos(theta_0 / 2), np.sin(theta_0 / 2) * np.array(n_0)])
        #quat_0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.x_0 = np.hstack((pos_0, quat_0))

        
        ## COmpute Auxiliar vectora
        self.x = np.zeros((7, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.x[:, 0] = self.x_0

        # Dual quaternion internal states
        self.dual_1 = dualquat_from_pose(self.x_0[3], self.x_0[4], self.x_0[5],  self.x_0[6], self.x_0[0], self.x_0[1], self.x_0[2])
        self.X = np.zeros((8, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.X[:, 0] = np.array(self.dual_1).reshape((8, ))

        # Body frame
        self.body_msg = Odometry()
        self.publisher_body_ = self.create_publisher(Odometry, "body", 10)

        self.publisher_point_ = self.create_publisher(PointStamped, 'point_1', 10)
        self.publisher_point_1_ = self.create_publisher(PointStamped, 'point_2', 10)

        # fames
        self.frames = TransformBroadcaster(self)

        # Camera transformation
        self.camera_pose = camera_pose_f(0.0)
        self.forward_camera = forward_camera_f(self.X[0:4, 0], self.X[4:8, 0], 0.0)

        # Buffer and transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        # Define a point in the W frame
        self.point_in_w = PointStamped()
        self.point_in_w.header.frame_id = 'world'
        self.point_in_w.header.stamp = self.get_clock().now().to_msg()
        self.point_in_w.point.x = 1.2
        self.point_in_w.point.y = 0.0
        self.point_in_w.point.z = 0.1

        # Define a point in the W frame
        self.point_in_w1 = PointStamped()
        self.point_in_w1.header.frame_id = 'world'
        self.point_in_w1.header.stamp = self.get_clock().now().to_msg()
        self.point_in_w1.point.x = 1.0
        self.point_in_w1.point.y = 0.2
        self.point_in_w1.point.z = 0.0

        self.x_w = np.zeros((3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.x_w[:, 0] =  np.array([self.point_in_w.point.x, self.point_in_w.point.y, self.point_in_w.point.z])

        self.x_w1 = np.zeros((3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.x_w1[:, 0] =  np.array([self.point_in_w1.point.x, self.point_in_w1.point.y, self.point_in_w1.point.z])

        # Numerical Intgeration of the system
        ocp = create_ocp_solver(self.X[:, 0], self.N_prediction, self.t_N, self.L, self.ts)

        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= False, generate= False)
        self.acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + ocp.model.name + ".json", build= False, generate= False)

        # Save file
        path_image_file_name = "results"
        self.path_image_file = os.path.join(get_package_share_path("cbf_control"), path_image_file_name)

        # Linear velocity inertial frame and angular velocity body frame
        self.u = np.zeros((6, self.t.shape[0] - self.N_prediction), dtype=np.double)

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)

        # Start thread for the simulation
        self.simulation_thread.start()

    def body_frame(self, X):
        h = get_trans(X)
        q = get_quat(X)
        self.body_msg.header.frame_id = "world"
        self.body_msg.header.stamp = self.get_clock().now().to_msg()

        self.body_msg.pose.pose.position.x = float(h[1, 0])
        self.body_msg.pose.pose.position.y = float(h[2, 0])
        self.body_msg.pose.pose.position.z = float(h[3, 0])

        self.body_msg.pose.pose.orientation.x = float(q[1, 0])
        self.body_msg.pose.pose.orientation.y = float(q[2, 0])
        self.body_msg.pose.pose.orientation.z = float(q[3, 0])
        self.body_msg.pose.pose.orientation.w = float(q[0, 0])

        # Send Message
        self.publisher_body_.publish(self.body_msg)
        return None 

    def transform_point(self, frame):
            try:
                # Wait for the transform to become available
                transform = self.tf_buffer.lookup_transform(frame, 'world', rclpy.time.Time())
                # Transform the point from W frame to target frame
                point_in_target_frame = tf2_geometry_msgs.do_transform_point(self.point_in_w, transform)

                vector = np.array([point_in_target_frame.point.x, point_in_target_frame.point.y, point_in_target_frame.point.z])
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                vector = np.array([0.0, 0.0, 0.0])
            return  vector

    def tranform_data(self, origin_frame, child_frame, X):
        # Compute quaternion and translations
        t_d = get_trans(X)
        q_d = get_quat(X)

        t = TransformStamped()
        # Define the static transform

        t.header.stamp =  self.get_clock().now().to_msg()
        t.header.frame_id = origin_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(t_d[1, 0])
        t.transform.translation.y = float(t_d[2, 0])
        t.transform.translation.z = float(t_d[3, 0])
        t.transform.rotation.x = float(q_d[1, 0])
        t.transform.rotation.y = float(q_d[2, 0])
        t.transform.rotation.z = float(q_d[3, 0])
        t.transform.rotation.w = float(q_d[0, 0])
        self.frames.sendTransform(t)

    def transform_point_manual(self, X):
        # Compute quaternion and translations
        # Get translation and Orientation body frame
        t_d = get_trans(X)
        q_d = np.array(get_quat(X)).reshape((4, ))

        # Transform translation to a vector no a pure quaternion anymore
        t_wb = np.array([t_d[1, 0], t_d[2, 0], t_d[3, 0]]).reshape((3, ))


        x_w = np.array([self.point_in_w.point.x,  self.point_in_w.point.y, self.point_in_w.point.z])
        vector = inverse_rot(q_d, x_w - t_wb)
        return np.array(vector).reshape((3, ))

    def systemdynamics(self, x, v):
        xdot = v
        return xdot[:]

    def f_d(self, x, v):
        k1 = self.systemdynamics(x, v)
        k2 = self.systemdynamics(x+(self.ts/2)*k1, v)
        k3 = self.systemdynamics(x+(self.ts/2)*k2, v)
        k4 = self.systemdynamics(x+(self.ts)*k3, v)
        x_k = x + (self.ts/6)*(k1 +2*k2 +2*k3 +k4)
        return x_k

    def transform_point_manual_camera(self, X_b, X_c, X):
        # Compute quaternion and translations of the final frame
        # We need only the rotation part
        t = get_trans(X)
        q = np.array(get_quat(X)).reshape((4, ))

        # Compute orientation and translation body frame respect to the world frame
        t_b = get_trans(X_b)
        q_b = np.array(get_quat(X_b)).reshape((4, ))

        position_b = np.array([t_b[1, 0], t_b[2, 0], t_b[3, 0]]).reshape((3, ))

        # Compute quaternion and translations of the camera respect to body
        t_c = get_trans(X_c)
        position_cb = np.array([t_c[1, 0], t_c[2, 0], t_c[3, 0]]).reshape((3, ))
        q_c = np.array(get_quat(X_c)).reshape((4, ))
        
        # Point in the world frame 
        x_w = np.array([self.point_in_w.point.x,  self.point_in_w.point.y, self.point_in_w.point.z])
        x_w1 = np.array([self.point_in_w1.point.x,  self.point_in_w1.point.y, self.point_in_w1.point.z])

        vector1 = inverse_rot(q, x_w - position_b) - inverse_rot(q_c, position_cb)
        vector2 = inverse_rot(q, x_w1 - position_b) - inverse_rot(q_c, position_cb)
        return np.array(vector1).reshape((3, )), np.array(vector2).reshape((3, ))
        
    def update_point(self, x1, point_in_w, publisher_point_):
        # Define a point in the W frame
        point_in_w.header.frame_id = 'world'
        point_in_w.header.stamp = self.get_clock().now().to_msg()
        point_in_w.point.x = x1[0]
        point_in_w.point.y = x1[1]
        point_in_w.point.z = x1[2]
        # Publish the message
        publisher_point_.publish(point_in_w)
        return None

    def H_plus(self, q):
        # Compute the H_plus operator
        H_plus = np.vstack((
        np.hstack((q[0], -q[1], -q[2], -q[3])),
        np.hstack((q[1], q[0], -q[3], q[2])),
        np.hstack((q[2], q[3], q[0], -q[1])),
        np.hstack((q[3], -q[2], q[1], q[0]))))
        return H_plus


    def H_minus(self, q):
        # Compute the minus operator 
        H_minus = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], q[3], -q[2]],
        [q[2], -q[3], q[0], q[1]],
        [q[3], q[2], -q[1], q[0]]])

        return H_minus

    def transform_point_manual_matrix_form(self, X):
        # Compute quaternion and translations
        # Get translation and Orientation body frame
        t_d = get_trans(X)
        q_d = np.array(get_quat(X)).reshape((4, ))

        # Transform translation to a vector no a pure quaternion anymore
        t_wb = np.array([t_d[1, 0], t_d[2, 0], t_d[3, 0]]).reshape((3, ))

        # Matrix form projection to the body frame
        H1 = self.H_plus(q_d)
        H2 = self.H_minus(q_d)

        t_wb = np.array([t_d[1, 0], t_d[2, 0], t_d[3, 0]]).reshape((3, ))
        x_w = np.array([self.point_in_w.point.x,  self.point_in_w.point.y, self.point_in_w.point.z])

        T = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        vector = H1.T@H2@T@(x_w - t_wb)
        return vector[1:4]
    
    def transform_point_manual_camera_matrix_form(self, X_b, X_c, X):
        # Compute quaternion and translations of the final frame
        c = np.array(get_quat(X)).reshape((4, ))
        H_plus_c = self.H_plus(c)
        H_minus_c = self.H_minus(c)

        # Compute orientation and translation body frame respect to the world frame
        t_b = np.array(get_trans(X_b)).reshape((4, ))
        position_b = np.array([0.0, t_b[1], t_b[2], t_b[3]]).reshape((4, ))

        # Compute quaternion and translations of the camera respect to body
        t_bc = np.array(get_trans(X_c)).reshape((4, ))
        position_bc = np.array([0.0, t_bc[1], t_bc[2], t_bc[3]]).reshape((4, ))
        q_bc = np.array(get_quat(X_c)).reshape((4, ))

        H_plus_q_bc = self.H_plus(q_bc)
        H_minus_q_bc = self.H_minus(q_bc)
        
        # Point in the world frame 
        x_w = np.array([0.0, self.point_in_w.point.x,  self.point_in_w.point.y, self.point_in_w.point.z])
        x_w1 = np.array([0.0, self.point_in_w1.point.x,  self.point_in_w1.point.y, self.point_in_w1.point.z])

        x_c1 = H_plus_c.T@H_minus_c@(x_w - position_b) - H_plus_q_bc.T@H_minus_q_bc@position_bc
        x_c2 = H_plus_c.T@H_minus_c@(x_w1 - position_b) - H_plus_q_bc.T@H_minus_q_bc@position_bc
        return x_c1[1:4], x_c2[1:4]

    def l2_norm(self, x_cd, x_max, y_max):
        # Desired Point camera frame
        x_c = x_cd[0]
        y_c = x_cd[1]

        number_points = 200

        # Create a grid of x, y values
        x = np.linspace(-x_max, x_max, number_points)
        y = np.linspace(-y_max, y_max, number_points)
        X, Y = np.meshgrid(x, y)

        R1_norm = np.sqrt((X - x_c)**2)
        R2_norm = np.sqrt((Y - y_c)**2)

        # Define the smooth exponential function using the L2 norm
        Z_smooth_norm = R1_norm + R2_norm

        return Z_smooth_norm

    def exp_function(self, x_cd, x_max, y_max, n, kx, ky, kz, r_max_x, r_max_y, r_max_z):
        # Desired Point camera frame
        x_c = x_cd[0]
        y_c = x_cd[1]

        number_points = 200

        # Create a grid of x, y values
        x = np.linspace(-x_max, x_max, number_points)
        y = np.linspace(-y_max, y_max, number_points)
        X, Y = np.meshgrid(x, y)

        # Define the parameters of the smooth exponential function
        A = 1.0  # Amplitude

        # Calculate the distance from the center using the L2 norm (Euclidean distance)
        R1 = (X - x_c)**n
        R2 = (Y - y_c)**n
        
        # Define the smooth exponential function using the L2 norm
        Z_smooth = 1 - A * np.exp(-((R1 / r_max_x**kx) + (R2 / r_max_y**ky)))
        return Z_smooth

    def exp(self, x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z):
        # Desired Point camera frame
        x_0 = x_cd[0]
        y_0 = x_cd[1]
        z_0 = x_cd[2]

        R1 = (x_c[0] - x_0)**n
        R2 = (x_c[1] - y_0)**n
        R3 = (x_c[2] - z_0)**n

        A = 1.0  # Amplitude

        #value = A * np.exp(-((R1 / r_max_x**kx) + (R2 / r_max_y**ky) + (R3 / r_max_z**kz)))
        value = A * np.exp(-((R1 / r_max_x**kx) + (R2 / r_max_y**ky)))
        return  value

    def v_function(self, x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z):
        # Desired Point camera frame
        value = 1 - self.exp(x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z)
        return  value

    def exp_jacobian(self, x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z):
        # Desired Point camera frame
        x_0 = x_cd[0]
        y_0 = x_cd[1]
        z_0 = x_cd[2]

        a = ((x_c[0] - x_0)**(n-1))/r_max_x**kx
        b = ((x_c[1] - y_0)**(n-1))/r_max_y**ky
        c = ((x_c[2] - z_0)**(n-1))/r_max_z**kz

        factor_exp = self.exp(x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z)
        aux = factor_exp*n

        #Jaux = np.array([[a, b, c]])
        Jaux = np.array([[a, b]])
        J = aux*Jaux
        return  J
    def Jacobian_camera_frame(self, X_b, X_c, X, x_c):
        c = np.array(get_quat(X)).reshape((4, ))
        H_plus_c = self.H_plus(c)
        H_minus_c = self.H_minus(c)

        # Compute orientation and translation body frame respect to the world frame
        t_b = np.array(get_trans(X_b)).reshape((4, ))

        # Compute quaternion and translations of the camera respect to body
        t_bc = np.array(get_trans(X_c)).reshape((4, ))
        position_bc = np.array([0.0, t_bc[1], t_bc[2], t_bc[3]]).reshape((4, ))
        q_bc = np.array(get_quat(X_c)).reshape((4, ))

        H_plus_q_bc = self.H_plus(q_bc)
        H_minus_q_bc = self.H_minus(q_bc)

        # Point respect to the camera frame  value from  the sensor
        position_c = np.array([0.0, x_c[0], x_c[1], x_c[2]]).reshape((4, ))


        # Jacobian elements
        a = H_plus_q_bc@position_c + H_minus_q_bc@position_bc
        b = H_minus_q_bc.T@position_c + H_plus_q_bc.T@position_bc

        H_minus_a = self.H_minus(a)

        H_plus_b = self.H_plus(b)

        # Compute Jacobians
        T = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        T2 = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        
        # Jacobian first part
        J1 = (1/2)*(H_plus_b@H_minus_q_bc - H_plus_q_bc.T@H_minus_a)@T
        J2 = -H_plus_c.T@H_minus_c@T
        J3 = T2@H_plus_c.T@H_minus_c@T
        J = np.hstack((J1, J2))

        # Reduce Dimensions
        J = T2@J
        return J, J3
        
    def control_law_camera_frame(self, X_b, X_c, X, x_c_1, x_cd_1, x_c1_dot_w, x_c_2, x_cd_2, x_c2_dot_w):
        J1, J31 = self.Jacobian_camera_frame(X_b, X_c, X, x_c_1)
        J2, J32 = self.Jacobian_camera_frame(X_b, X_c, X, x_c_2)
        J = np.vstack((J1, J2))
        J3 = block_diag(J31, J32)

        x_c_dot_w = np.vstack((x_c1_dot_w.reshape((3, 1)), x_c2_dot_w.reshape((3, 1))))

        # Error Control
        x_error1 = x_cd_1 - x_c_1
        x_error2 = x_cd_2 - x_c_2

        x_error1 = x_error1.reshape((3, 1))
        x_error2 = x_error2.reshape((3, 1))

        x_error = np.vstack((x_error1, x_error2))

        # Null space projection
        #W = np.diag([1, 1, 1, 1, 1, 1])
        W = np.diag([300, 300, 1, 1, 1, 1])
        I = np.diag([1, 1, 1, 1, 1, 1])

        # Desired Velocities Null space
        v_d_i = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        w_d_b = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        # General Vector velocities
        v_d = np.vstack((w_d_b, v_d_i))

        J_inverse = np.linalg.inv(W)@J.T@np.linalg.inv(J@np.linalg.inv(W)@J.T)
        K = 0.1*np.diag([1, 1, 1, 1, 1, 1])

        # Control Law
        #u = J_inverse@(K@x_error) + (I + J_inverse@J)@v_d
        #u = J_inverse@(K@x_error) + (I + J_inverse@J)@v_d
        u = np.linalg.pinv(J)@(0.1*x_error) + (I + J_inverse@J)@v_d
        return u[:, 0]

    def control_law_camera_frame_exp(self, X_b, X_c, X, x_c, x_cd, x_dot_w, n, kx, ky, kz, r_max_x, r_max_y, r_max_z):
        # Compute quaternion and translations of the final frame
        c = np.array(get_quat(X)).reshape((4, ))
        H_plus_c = self.H_plus(c)
        H_minus_c = self.H_minus(c)

        # Compute orientation and translation body frame respect to the world frame
        t_b = np.array(get_trans(X_b)).reshape((4, ))

        # Compute quaternion and translations of the camera respect to body
        t_bc = np.array(get_trans(X_c)).reshape((4, ))
        position_bc = np.array([0.0, t_bc[1], t_bc[2], t_bc[3]]).reshape((4, ))
        q_bc = np.array(get_quat(X_c)).reshape((4, ))

        H_plus_q_bc = self.H_plus(q_bc)
        H_minus_q_bc = self.H_minus(q_bc)

        # Point respect to the camera frame  value from  the sensor
        position_c = np.array([0.0, x_c[0], x_c[1], x_c[2]]).reshape((4, ))


        # Jacobian elements
        a = H_plus_q_bc@position_c + H_minus_q_bc@position_bc
        b = H_minus_q_bc.T@position_c + H_plus_q_bc.T@position_bc

        H_minus_a = self.H_minus(a)

        H_plus_b = self.H_plus(b)

        # Compute Jacobians
        T = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        T2 = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        
        # Jacobian first part
        J1 = (1/2)*(H_plus_b@H_minus_q_bc - H_plus_q_bc.T@H_minus_a)@T
        J2 = -H_plus_c.T@H_minus_c@T
        J3 = T2@H_plus_c.T@H_minus_c@T
        J = np.hstack((J1, J2))

        # Reduce Dimensions
        J = T2@J

        # Error Control
        v_error = 0.0 - self.v_function(x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z)
        v_error = np.array([v_error])
        J_exp = self.exp_jacobian(x_c, x_cd, n, kx, ky, kz, r_max_x, r_max_y, r_max_z)

        J = J_exp@J[0:2, :]
        J3 = J_exp@J3[0:2, :]


        # Null space projection
        W = np.diag([1, 1, 1, 1, 1, 1])
        W = np.diag([500, 500, 1, 1, 1, 500])
        I = np.diag([1, 1, 1, 1, 1, 1])

        # Desired Velocities Null space
        v_d_i = np.array([0.0, 0.0, 0.0])
        w_d_b = np.array([0.0, 0.0, 0.0])
        # General Vector velocities
        v_d = np.hstack((v_d_i, w_d_b))

        J_inverse = np.linalg.inv(W)@J.T@np.linalg.inv(J@np.linalg.inv(W)@J.T)
        K = 5

        # Control Law
        #u = J_inverse@(K@x_error - J3@x_dot_w) + (I + J_inverse@J)@v_d
        u = J_inverse@(K*v_error- J3@x_dot_w) + (I + J_inverse@J)@v_d
        #u = J_inverse@(K*v_error) + (I + J_inverse@J)@v_d
        #u = np.linalg.pinv(J)@(v_error)
        return u

    def control_law(self, x_bd, x_b, X, x_dot_w):
        # Desired values 
        #x_bd = np.array([-0.0, -0.0, -1])
        # Get translation and Orientation body frame
        t_d = get_trans(X)
        q_d = np.array(get_quat(X)).reshape((4, ))

        # Transform translation to a vector no a pure quaternion anymore
        x_b_quat = np.array([0.0, x_b[0], x_b[1], x_b[2]])

        T = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        T2 = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # Matrix form projection to the body frame
        H1_q = self.H_plus(q_d)
        H2_q = self.H_minus(q_d)
        J2 = -H1_q.T@H2_q@T

        J3 = H1_q.T@H2_q@T
        J3 = T2@J3

        # Matrix form point respect to the body frame
        H1_xb = self.H_plus(x_b_quat)
        H2_xb = self.H_minus(x_b_quat)
        J1 = (1/2)*(H1_xb - H2_xb)@T

        J = np.hstack((J1, J2))
        J = T2@J
        x_error = x_bd - x_b
        u = np.linalg.pinv(J)@(x_error - J3@x_dot_w)
        #u = np.linalg.pinv(J)@(x_error)
        return u

    def run(self):
        # Control Loop
        #self.u[3, :] = 0.5*np.sin(self.t)
        #self.u[4, :] = 0.5*np.cos(self.t)
        #self.u[2, :] = 1*np.cos(self.t)
        #self.u[5, :] = 1*np.cos(self.t)

        # Velocity Object
        v_w = np.zeros((3, self.t.shape[0]), dtype=np.double)
        #v_w[0, :] = 0.0*np.sin(self.t)
        #v_w[1, :] = 0.1*np.cos(self.t)
        #v_w[2, :] = 0*np.cos(self.t)

        v_w1 = np.zeros((3, self.t.shape[0]), dtype=np.double)
        #v_w1[0, :] = 0.0*np.sin(self.t)
        #v_w1[1, :] = 0.1*np.cos(self.t)

        # Empty vector current values b frame
        x_b_data = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)
        x_c_data = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)

        # Desired Point B frame
        x_bd = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)
        x_bd[0, :] = 0.0
        x_bd[1, :] = -0.1
        x_bd[2, :] = -0.2

        x_cd = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)
        x_cd[0, :] = 0.0
        x_cd[1, :] = -0.05
        x_cd[2, :] = 0.9
        
        x_cd1 = np.zeros((3, self.t.shape[0] - self.N_prediction), dtype=np.double)
        x_cd1[0, :] = -0.2
        x_cd1[1, :] = 0.05
        x_cd1[2, :] = 0.7

        # Map L2 Norm
        x_max = 1.5
        y_max = 1.0
        n = 10
        kx = 4
        ky = 4
        kz = 4
        r_max_x = 1.5  # Maximum radius for the smooth transition
        r_max_y = 0.6  # Maximum radius for the smooth transition
        r_max_z = 2  # Maximum radius for the smooth transition
        Z_smooth_norm = self.l2_norm(x_cd[:, 0], x_max, y_max)
        Z_smooth_exp = self.exp_function(x_cd[:, 0], x_max, y_max, n, kx, ky, kz, r_max_x, r_max_y, r_max_z)

        ## Simulation loop
        for k in range(0, self.t.shape[0]- self.N_prediction):
            # Get model
            tic = time.time()
            # Update point for the projection
            self.update_point(self.x_w[:, k], self.point_in_w, self.publisher_point_)
            self.update_point(self.x_w1[:, k], self.point_in_w1, self.publisher_point_1_)
            # Point Projected to the body frame using tf in the background and manual computation
            x_b = self.transform_point('body')
            x_b_aux = self.transform_point_manual(self.X[:, k])
            x_b_aux_2 = self.transform_point_manual_matrix_form(self.X[:, k])

            # Camera orientation Dualquaternion
            self.forward_camera = forward_camera_f(self.X[0:4, k], self.X[4:8, k], 0.0)

            # Point Projected to the camera frame using tf and manual computation
            x_c = self.transform_point("camera")
            x_c_aux, x_c_2_aux = self.transform_point_manual_camera(self.X[:, k], self.camera_pose, self.forward_camera)
            x_c_aux_2, x_c_2_aux_2 = self.transform_point_manual_camera_matrix_form(self.X[:, k], self.camera_pose, self.forward_camera)
            x_c_data[:, k] = x_c_aux

            # Print values
            print(x_c_aux)
            print(x_c_2_aux)
            print("--------")
            print(x_c_aux_2)
            print(x_c_2_aux_2)
            t_camera_body = get_trans(self.camera_pose)
            q_camera_body = get_quat(self.camera_pose)
            #print(t_camera_body)
            #print(q_camera_body)


            # Control Lawcamera frame
            self.u[:, k] = self.control_law_camera_frame(self.X[:, k], self.camera_pose, self.forward_camera, x_c_aux_2, x_cd[:, k], v_w[:, k], x_c_2_aux_2, x_cd1[:, k], v_w1[:, k])
            self.u[0, k] = 0.0
            self.u[1, k] = 0.0
            self.u[5, k] = 0.0
            #self.u[:, k] = self.control_law_camera_frame_exp(self.X[:, k], self.camera_pose, self.forward_camera, x_c_aux, x_cd[:, k], v_w[:, k], n, kx, ky, kz, r_max_x, r_max_y, r_max_z)

            # Publish TF information body, camera and usign dualquaternion
            self.tranform_data("world", "body", self.X[:, k])
            self.tranform_data("body", "camera", self.camera_pose)
            self.tranform_data("world", "camera_fixed", self.forward_camera)
            self.body_frame(self.X[:, k])

            # Acados integration
            self.acados_integrator.set("x", self.X[:, k])
            self.acados_integrator.set("u", self.u[:, k])
            status_integral = self.acados_integrator.solve()
            xcurrent = self.acados_integrator.get("x")
            self.X[:, k + 1] = xcurrent

            # Update point information
            self.x_w1[:, k + 1] = self.f_d(self.x_w1[:, k], v_w1[:, k])
            self.x_w[:, k + 1] = self.f_d(self.x_w[:, k], v_w[:, k])
            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info("Camera Projections")
        
        fig11, ax11, ax12, ax13 = fancy_plots_3()
        plot_states_position_b_frame(fig11, ax11, ax12, ax13, x_c_data[0:3, :], x_cd[0:3, :], self.t, "Position Results Body Frame Point "+ str(self.initial), self.path_image_file)
        plt.show()
        fig21, ax21, ax22, ax23 = fancy_plots_3()
        plot_linear_velocities(fig21, ax21, ax22, ax23, self.u[3:6, :], self.t, "Linear Velocity Inertial Frame "+ str(self.initial), self.path_image_file)
        plt.show()
        
        fig31, ax31, ax32, ax33 = fancy_plots_3()
        plot_angular_velocities(fig31, ax31, ax32, ax33, self.u[0:3, :], self.t, "Angular Velocity Body Frame "+ str(self.initial), self.path_image_file)
        plt.show()
        #plt.figure(figsize=(12, 6))
        #plt.subplot(1, 2, 1)
        #plt.imshow(Z_smooth_norm, extent=[-x_max, x_max, -y_max, y_max], origin='lower', cmap='viridis')
        #plt.colorbar(label='f(x, y)')
        #plt.title('L2 Norm')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.plot(x_c_data[0, :], x_c_data[1, :], 'rx', markersize=1)  # Red point

        #plt.subplot(1, 2, 2)
        #plt.imshow(Z_smooth_exp, extent=[-x_max, x_max, -y_max, y_max], origin='lower', cmap='viridis')
        #plt.colorbar(label='f(x, y)')
        #plt.title('Exp function')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.plot(x_c_data[0, :], x_c_data[1, :], 'rx', markersize=1)  # Red point

        #plt.tight_layout()
        #plt.show()

def main(args=None):
    rclpy.init(args=args)
    planning_node = CameraNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()