import numpy as np
import numpy as np
from dex_robot.utils.robot_wrapper import RobotWrapper
import os
from dex_robot.utils.file_io import rsc_path
import transforms3d
import cv2
import numpy as np

from scipy.linalg import sqrtm
from numpy.linalg import inv
import numpy
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv

A1 = numpy.array([[-0.989992, -0.14112,  0.000, 0],
                 [0.141120 , -0.989992, 0.000, 0],
                 [0.000000 ,  0.00000, 1.000, 0],
                 [0        ,        0,     0, 1]])

B1 = numpy.array([[-0.989992, -0.138307, 0.028036, -26.9559],
                 [0.138307 , -0.911449, 0.387470, -96.1332],
                 [-0.028036 ,  0.387470, 0.921456, 19.4872],
                 [0        ,        0,     0, 1]])

A2 = numpy.array([[0.07073, 0.000000, 0.997495, -400.000],
                [0.000000, 1.000000, 0.000000, 0.000000],
                [-0.997495, 0.000000, 0.070737, 400.000],
                [0, 0, 0,1]])

B2 = numpy.array([[ 0.070737, 0.198172, 0.997612, -309.543],
                [-0.198172, 0.963323, -0.180936, 59.0244],
                [-0.977612, -0.180936, 0.107415, 291.177],
                [0, 0, 0, 1]])


def logR(T):
    R = T[0:3, 0:3]
    theta = numpy.arccos((numpy.trace(R) - 1)/2)
    logr = numpy.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*numpy.sin(theta))
    return logr

def Calibrate(A, B):
    n_data = len(A)
    M = numpy.zeros((3,3))
    C = numpy.zeros((3*n_data, 3))
    d = numpy.zeros((3*n_data, 1))
    A_ = numpy.array([])
    
    for i in range(n_data-1):
        alpha = logR(A[i])
        beta = logR(B[i])
        alpha2 = logR(A[i+1])
        beta2 = logR(B[i+1])
        alpha3 = numpy.cross(alpha, alpha2)
        beta3  = numpy.cross(beta, beta2) 
        
        M1 = numpy.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
        M2 = numpy.dot(beta2.reshape(3,1),alpha2.reshape(3,1).T)
        M3 = numpy.dot(beta3.reshape(3,1),alpha3.reshape(3,1).T)
        M = M1+M2+M3
    
    theta = numpy.dot(sqrtm(inv(numpy.dot(M.T, M))), M.T)

    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        
        C[3*i:3*i+3, :] = numpy.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - numpy.dot(theta, trans_b)
        
    b_x  = numpy.dot(inv(numpy.dot(C.T, C)), numpy.dot(C.T, d))
    return theta, b_x

# X = numpy.eye(4)
# A = [A1, A2]
# B = [B1, B2]
# theta, b_x = Calibrate(A, B)
# X[0:3, 0:3] = theta
# X[0:3, -1] = b_x.flatten()

import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_random_transformation_matrix():
    """ 180도 회전을 제외한 임의의 4x4 변환 행렬 생성 """
    while True:
        # 랜덤 회전 행렬 생성 (SO(3))
        random_rotation = R.random().as_matrix()  # scipy를 이용한 무작위 회전 행렬
        if np.linalg.det(random_rotation) > 0.99:  # det(R) ≈ 1을 만족하는 경우만 사용
            break
    
    # 랜덤 이동 벡터 생성
    translation = np.random.uniform(-10, 10, size=(3,))
    
    # 변환 행렬 구성
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = random_rotation
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

X_init = np.array([[ 0.53522037,  0.62129585,  0.57230291, -5.6590087 ],
 [-0.8120144,   0.5651012,   0.14592206,  7.85438364],
 [-0.2327483,  -0.54281866,  0.80695485, -9.92593868],
 [ 0.,          0.,          0.,          1.        ]])
# A_list와 B_list에 각각 10개의 랜덤한 변환 행렬 저장
A_list = [generate_random_transformation_matrix() for _ in range(9)]
B_list = []
for i in range(len(A_list)):
    B = np.linalg.inv(X_init) @ A_list[i] @ X_init
    B[:3, 3] += np.random.normal(0, 0.01, size=(3))
    B_list.append(B)

X = np.eye(4)
theta, b_x = Calibrate(A_list, B_list)
X[0:3, 0:3] = theta
X[0:3, -1] = b_x.flatten()

print("X: ")
print(X)
print("AX: ")
print(A_list[0] @ X)
print("XB: ")
print(X @ B_list[0])
print("AX-BX: ")

print(A_list[0] @ X - X @ B_list[0])
print(X_init-X)

# robot = RobotWrapper(
#     os.path.join(rsc_path, "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf")
# )

# # 예제 데이터 (A, B 쌍 여러 개 필요)
# A_samples = []
# B_samples = []

# for i in range(9):
#     qpos = np.load(f"data/calibration_pose/pose_{i}.npy")
#     robot.compute_forward_kinematics(qpos)
#     link_index = robot.get_link_index("link5")
#     link_pose_in_robot_frame = robot.get_link_pose(link_index)
#     B_samples.append(link_pose_in_robot_frame)

