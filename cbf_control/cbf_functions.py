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
        self.t_final = 20
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)

        ## Prediction time 
        self.t_N = 0.5
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)

        # Start thread for the simulation
        self.simulation_thread.start()


    def run(self):
        # Control Loop
        # Define the camera frame dimensions
        W, H = 640, 480  # Example dimensions

        # Define the center of the frame
        x_c, y_c = W / 2, H / 2

        # Create a grid of x, y values
        x = np.linspace(0, W, W)
        y = np.linspace(0, H, H)
        X, Y = np.meshgrid(x, y)

        # Define the parameters of the smooth exponential function
        A = 1.0  # Amplitude
        r_max_1 = 220  # Maximum radius for the smooth transition
        r_max_2 = 150  # Maximum radius for the smooth transition
        k = 4  # Sharpness of the transition

        # Calculate the distance from the center using the L2 norm (Euclidean distance)
        R1 = np.sqrt((X - x_c)**8)
        R2 = np.sqrt((Y - y_c)**8)
        
        R1_norm = np.sqrt((X - x_c)**2)
        R2_norm = np.sqrt((Y - y_c)**2)

        # Define the smooth exponential function using the L2 norm
        #Z_smooth_L2 = A * np.exp(-((R1 / r_max_1)**k + (R2 / r_max_2)**k))
        Z_smooth_L2 = 1 - A * np.exp(-((R1 / r_max_1**k) + (R2 / r_max_2**k)))
        Z_smooth_norm = R1_norm + R2_norm

        plt.figure(figsize=(12, 6))

        # First subplot: Smooth Exponential Function with L2 Norm
        plt.subplot(1, 2, 1)
        plt.imshow(Z_smooth_L2, extent=[0, W, 0, H], origin='lower', cmap='viridis')
        plt.colorbar(label='f(x, y)')
        plt.title('Exponential Function with L Norm')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Second subplot: Norm Function
        plt.subplot(1, 2, 2)
        plt.imshow(Z_smooth_norm, extent=[0, W, 0, H], origin='lower', cmap='viridis')
        plt.colorbar(label='f(x, y)')
        plt.title('L2 Norm')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.show()

        for k in range(0, self.t.shape[0]- self.N_prediction):
            # Get model
            tic = time.time()
            # Update point for the projection
            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info("Camera CBF")
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