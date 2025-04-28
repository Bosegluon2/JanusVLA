"""
name : ur5e_ik.py
description : UR5e Inverse Kinematics
author : 汪子策
date : 2025-4-28
version : 1.0
license : All rights reserved.
Copyright (c) 2025 汪子策. All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------ DH Parameters for UR5e (from your table) ------
DH_PARAMS = [
    (0,          np.pi/2, 89.15, 0),    # (a, alpha, d, theta)
    (-425,       0,        0,    0),
    (-392.25,    0,        0,    0),
    (0,          np.pi/2, 109.15, 0),
    (0,         -np.pi/2, 94.65, 0),
    (0,          0,       82.30, 0)
]

# ------ Function to compute individual transformation matrix ------
def dh_transform(a, alpha, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# ------ Forward Kinematics ------
def forward_kinematics(thetas):
    T = np.eye(4)
    joints = [np.array([0, 0, 0])]  # base point
    for i, (a, alpha, d, _) in enumerate(DH_PARAMS):
        Ti = dh_transform(a, alpha, d, thetas[i])
        T = T @ Ti
        joints.append(T[:3, 3])
    return T, joints

# ------ Numerical Jacobian ------
def numerical_jacobian(thetas, delta=1e-6):
    J = np.zeros((6, 6))
    T0, _ = forward_kinematics(thetas)
    for i in range(6):
        thetas_perturb = np.copy(thetas)
        thetas_perturb[i] += delta
        Ti, _ = forward_kinematics(thetas_perturb)

        dP = (Ti[:3, 3] - T0[:3, 3]) / delta
        dR = (rotation_vector_from_matrix(Ti[:3, :3]) - rotation_vector_from_matrix(T0[:3, :3])) / delta

        J[:, i] = np.concatenate((dP, dR))
    return J

# ------ Rotation Matrix to Rotation Vector ------
def rotation_vector_from_matrix(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        return np.zeros(3)
    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    return theta * np.array([rx, ry, rz])

# ------ Newton-Raphson Inverse Kinematics ------
def inverse_kinematics(target_T, initial_guess=None, tol=1e-3, max_iter=100):
    if initial_guess is None:
        thetas = np.zeros(6)
    else:
        thetas = initial_guess.copy()

    for _ in range(max_iter):
        T_current, _ = forward_kinematics(thetas)
        pos_err = target_T[:3, 3] - T_current[:3, 3]
        rot_err = rotation_vector_from_matrix(target_T[:3, :3]) - rotation_vector_from_matrix(T_current[:3, :3])
        error = np.concatenate((pos_err, rot_err))

        if np.linalg.norm(error) < tol:
            break

        J = numerical_jacobian(thetas)
        dtheta = np.linalg.pinv(J) @ error
        thetas += dtheta

    return thetas

# ------ Plotting ------
def plot_arm(ax, joints):
    joints = np.array(joints)
    ax.clear()
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o', markersize=8)
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 1500])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('UR5e Arm Drawing Circle')

# ------ Main: Draw Circle Animation ------
if __name__ == "__main__":
    # Generate circular trajectory
    r = 200  # radius mm
    z_fixed = 500  # fixed height
    angles = np.linspace(0, 2*np.pi, 100)
    trajectory = np.array([[r*np.cos(a), r*np.sin(a), z_fixed] for a in angles])

    # Precompute IK solutions
    joint_solutions = []
    current_guess = np.zeros(6)
    for point in trajectory:
        target_T = np.eye(4)
        target_T[:3, 3] = point
        solution = inverse_kinematics(target_T, initial_guess=current_guess)
        joint_solutions.append(solution)
        current_guess = solution  # use previous as next initial guess

    # Setup animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        _, joints = forward_kinematics(joint_solutions[frame])
        plot_arm(ax, joints)

    ani = FuncAnimation(fig, update, frames=len(joint_solutions), interval=100)
    plt.show()