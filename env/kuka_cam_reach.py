#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   kuka_reach_env.py
@Time    :   2021/03/20 14:33:24
@Author  :   Yan Wen
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import pybullet as p
import pybullet_data
import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
from numpy import arange
import logging

import math
import cv2
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from termcolor import colored
import torch
import matplotlib.pyplot as plt
from colorama import Fore, init, Back
import sys
import os

init(autoreset=True)  # this lets colorama takes effect only in current line.
# Otherwise, colorama will let the sentences below 'print(Fore.GREEN+'xx')'
# all become green color.

#### 一些变量 ######
#LOGGING_LEVEL=logging.INFO
# is_render=False
# is_good_view=False   #这个的作用是在step时加上time.sleep()，把机械比的动作放慢，看的更清，但是会降低训练速度
#########################

# logging.basicConfig(
#     level=LOGGING_LEVEL,
#     format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     filename='../logs/reach_env.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
#     filemode='w')
# logger = logging.getLogger(__name__)
# env_logger=logging.getLogger('env.py')

# logging模块的使用
# 级别                何时使用
# DEBUG       细节信息，仅当诊断问题时适用。
# INFO        确认程序按预期运行
# WARNING     表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
# ERROR       由于严重的问题，程序的某些功能已经不能正常执行
# CRITICAL    严重的错误，表明程序已不能继续执行

#是用来设置机械臂的摄像头参数，比如分辨率，视场等等。
# 摄像头一般用来获取机械臂和物品的相对位置、距离以及姿态等信息，进而控制机械臂进行抓取。
# 所以在这个环境中，相机参数与机械臂和物品的交互有关。
class KukaCamReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 2000

    # final_image_size=40
    # resize = T.Compose([T.ToPILImage(),
    #                     T.Resize(final_image_size, interpolation=Image.CUBIC),
    #                     T.ToTensor()])

    def __init__(self, is_render=False, is_good_view=False):

        # some camera parameters
        # all the parameters are tested with test/slide_bar_for_camera.py file.
        #定义了一些参数（宽度、高度、势场）这些参数都是用来计算相机视图矩阵和透视投影矩阵的，从而让PyBullet中的虚拟相机呈现出正确的视角。
        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
            [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  #the direction is from the light source position to the origin of the world frame.
        }
        #他这个有问题，两个cpu
        # self.device = torch.device(
        #     "cpu" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.view_matrix=p.computeViewMatrix(
        #     cameraEyePosition=self.camera_parameters['eye_position'],
        #     cameraTargetPosition=self.camera_parameters['target_position'],
        #     cameraUpVector=self.camera_parameters['camera_up_vector']
        # )
        #使用PyBullet函数 计算视角矩阵（view matrix）,视角矩阵是一个4x4的矩阵，用于将3D场景中的物体坐标转换为2D屏幕坐标。
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)
        #计算透视投影矩阵。它将三维世界坐标系中的点投影到二维平面上，用于实现透视效果。在图形学中，通常使用透视投影矩阵来定义摄像机的视角和投影方式，即如何将三维场景投影到二维平面上。
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
            self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs = 0.2
        self.x_high_obs = 0.7
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(#spaces.Box 是用来定义 gym 环境的观测空间的
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)
        # self.observation_space=spaces.Box(low=np.array([self.x_low_obs,self.y_low_obs,self.z_low_obs]),
        #                              high=np.array([self.x_high_obs,self.y_high_obs,self.z_high_obs]),
        #                              dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 84, 84))

        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #p.connect(p.GUI)
        self.step_counter = 0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                               "kuka_iiwa/model.urdf"),
                                  useFixedBase=True)
        table_uid = p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])
        #这段代码是用来改变物体的视觉形状。改变table为白色
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        #object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
        self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                                 "random_urdfs/000/000.urdf"),
                                    basePosition=[
                                        random.uniform(self.x_low_obs,
                                                       self.x_high_obs),
                                        random.uniform(self.y_low_obs,
                                                       self.y_high_obs), 0.01
                                    ])

        self.num_joints = p.getNumJoints(self.kuka_id)

        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()

        # (_,_,self.images,_,_) = p.getCameraImage(
        #     width=self.camera_parameters['width'],
        #     height=self.camera_parameters['height'],
        #     viewMatrix=self.view_matrix,
        #     projectionMatrix=self.projection_matrix,

        #     renderer=p.ER_BULLET_HARDWARE_OPENGL
        # )

        #获取相机拍摄的图像。
        #由于只需要获取rgba，因此使用(_, _, px, _, _)来取得相机拍摄的图像。
        #只需要图像中的颜色信息，所以是rgba
        (_, _, px, _,                        #[R1, G1, B1, A1, R2, G2, B2, A2, ...]
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px   #--原来的，是一个tuple元祖类型
        self.images = np.array(px)                         # -gpt：转化为numpy数组
        self.images = self.images.reshape((960, 960, 4))   # -gpt：数组的形状从 (960, 960, 4) 转换为 (960, 960, 4)
        #这行代码是在为机械臂的最后一个关节开启力和扭矩传感器。在机械臂运动过程中，通过传感器可以得到关节扭矩和力的信息，这些信息可以用来进行控制或者监测机械臂的运动状态。
        p.enableJointForceTorqueSensor(bodyUniqueId=self.kuka_id,
                                       jointIndex=self.num_joints - 1,
                                       enableSensor=True)

        # print(Fore.GREEN+'force_sensor={}'.format(self._get_force_sensor_value()))

        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        #return np.array(self.object_pos).astype(np.float32)
        #return np.array(self.robot_pos_obs).astype(np.float32)
        #从RGBA格式转换为RGB格式。GBA格式是指包含红色、绿色、蓝色和透明度四个通道的图像格式，而RGB格式只包含红色、绿色和蓝色三个通道，因此把第四个通道去掉就可以得到RGB格式的图像。
        #3 表示只选择前三个颜色通道（即红色、绿色和蓝色）。
        self.images = self.images[:, :, :3].astype(np.uint8)  # the 4th channel is alpha channel, we do not need it. .astype(np.uint8)
        #return self._process_image(self.images)

        return self._process_image(self.images) #RGB转换为灰度图像 -> 灰度图像是一种仅包含亮度值而不包含颜色信息的图像。它们通常用于减少图像处理的复杂性和计算量，因为它们只包含一个单独的亮度通道，而不是三个颜色通道。


    #这个函数将RGB图像转换为灰度图像，并将其大小调整为84x84
    #然后将其增加一个通道，并将像素值标准化到[0,1]范围内。
    #如果输入图像为空，则返回一个大小为(1, 84, 84)的全0数组。
    def _process_image(self, image):
        """Convert the RGB pic to gray pic and add a channel 1

        Args:
            image ([type]): [description]
        """

        if image is not None:
            image = np.array(image)
            if image is not None and len(image.shape) == 3 and image.shape[2] == 3:    #3通道的才转换，1通道的就不用转换了
                image = cv2.convertScaleAbs(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     #将RGB图片转换为灰度图，此时图片的通道数变为了1，输入必须是三通道。
                image = cv2.resize(image, (84, 84))[None, :, :] / 255.  #输出是一个三维数组
            return image
        else:
            return np.zeros((1, 84, 84))

        # image=image.transpose((2,0,1))
        # image=np.ascontiguousarray(image,dtype=np.float32)/255.
        # image=torch.from_numpy(image)
        # #self.processed_image=self.resize(image).unsqueeze(0).to(self.device)
        # self.processed_image=self.resize(image).to(self.device)
        # return self.processed_image


    def step(self, action):
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        self.new_robot_pos = [
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        #logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        return self._reward()

    def _reward(self):

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)
        #
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        #用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        #print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        #如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        if terminated:
            reward = -0.1
            self.terminated = True

        #如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.max_steps_one_episode:
            reward = -0.1
            self.terminated = True

        elif self.distance < 0.1:
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        info = {'distance:', self.distance}
        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #Debug，将px转换为三维数组
        px = np.reshape(px, (960, 960, 4))
        px = px[:, :, :3]  # 取前三个通道
        px = np.flipud(px)  # 上下翻转

        self.images = px    #这里的image需要的是三维的RGB图像

        self.processed_image = self._process_image(self.images)
        #self.observation=self.robot_state
        self.observation = self.object_state
        return self.processed_image, reward, self.terminated, info

    def close(self):
        p.disconnect()

    #这个函数是用来调试用的，作用是让机械臂到达一个目标位置，然后获取当前末端的力传感器数值。它可以用来测试机械臂是否能够到达指定的目标位置，并且力传感器是否正常工作。
    # 这对于调试机械臂的运动和感知能力非常有帮助。
    def run_for_debug(self, target_position):
        temp_robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=target_position,
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=temp_robot_joint_positions[i],
            )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        return self._get_force_sensor_value()

    #这个函数用于获取机械臂的力传感器值，具体来说是获取机械臂的最后一个关节的第3个值，即Z轴方向上的力传感器值。这个函数会在机械臂接触物体时被调用，用于判断机械臂是否接触到了物体。如果接触到了物体，就认为这个episode已经完成了。
    def _get_force_sensor_value(self):
        force_sensor_value = p.getJointState(bodyUniqueId=self.kuka_id,
                                             jointIndex=self.num_joints -
                                             1)[2][2]
        # the first 2 stands for jointReactionForces, the second 2 stands for Fz,
        # the pybullet methods' return is a tuple,so can not
        # index it with str like dict. I think it can be improved
        # that return value is a dict rather than tuple.
        return force_sensor_value


class CustomSkipFrame(gym.Wrapper):
    """ Make a 4 frame skip, so the observation space will change to (4,84,84) from (1,84,84)

    Args:
        gym ([type]): [description]
    """
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(skip, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        print("states:!!!!!!!!!!!!!!!")
        print(states)
        states = np.concatenate(states, 0)[None, :, :, :]   #原版
        # states中每个状态是一个三维的数组
        # 将 states 中的所有状态按照第一个维度拼接起来，形成一个 (num_states, height, width) 的数组。
        # 然后 [None, :, :, :] 在第一个维度增加了一个新的维度，变成了 (1, num_states, height, width) 的数组。
        #states = np.array(states)
        #print(states.shape)  # 打印 states 的维度
        #states = np.stack(states, axis=0) #GPT说的
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)],
                                0)[None, :, :, :]
        return states.astype(np.float32)


if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    import matplotlib.pyplot as plt
    env = KukaCamReachEnv(is_good_view=True, is_render=False)
    env = CustomSkipFrame(env)

    obs = env.reset()
    print(obs)
    print(obs.shape)

    img = obs[0][0]
    plt.imshow(img, cmap='gray')
    plt.show()

    # all the below are some debug codes, if you have interests, look through.

    # b=a[:,:,:3]
    # c=b.transpose((2,0,1))
    # #c=b
    # d=np.ascontiguousarray(c,dtype=np.float32)/255
    # e=torch.from_numpy(d)
    # resize=T.Compose([T.ToPILImage(),
    #                   T.Resize(40,interpolation=Image.CUBIC),
    #                     T.ToTensor()])

    # f=resize(e).unsqueeze(0)
    # #print(f)
    # # g=f.unsqueeze(0)
    # # print(g)
    # #f.transpose((2,0,1))

    # plt.imshow(f.cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #        interpolation='none')

    # #plt.imshow(f)
    # plt.show()

    # resize = T.Compose([T.ToPILImage(),
    #                 T.Resize(40, interpolation=Image.CUBIC),
    #                 T.ToTensor()])

# print(env)
# print(env.observation_space.shape)
# print(env.observation_space.sample())

# for i in range(10):
#     a=env.reset()
#     b=a[:,:,:3]
#     """
#     matplotlib.pyplot.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
#     alpha=None, vmin=None, vmax=None, origin=None, extent=None, *, filternorm=True,
#     filterrad=4.0, resample=None, url=None, data=None, **kwargs)

#     Xarray-like or PIL image
#     The image data. Supported array shapes are:

#     (M, N): an image with scalar data. The values are mapped to colors using normalization and a colormap. See parameters norm, cmap, vmin, vmax.
#     (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
#     (M, N, 4): an image with RGBA values (0-1 float or 0-255 int), i.e. including transparency.
#     The first two dimensions (M, N) define the rows and columns of the image.
#     Out-of-range RGB(A) values are clipped.
#     """
#     plt.imshow(b)
#     plt.show()
#     time.sleep(1)

# for i in range(720):
#     for j in range(720):
#         for k in range(3):
#             if not a[i][j][k]==b[i][j][k]:
#                 print(Fore.RED+'there is unequal')
#                 raise ValueError('there is unequal.')
# print('check complete')

#print(a)
#force_sensor=env.run_for_debug([0.6,0.0,0.03])
# print(Fore.RED+'after force sensor={}'.format(force_sensor))
#print(env.action_space.sample())

# sum_reward=0
# for i in range(10):
#     env.reset()
#     for i in range(2000):
#         action=env.action_space.sample()
#         #action=np.array([0,0,0.47-i/1000])
#         obs,reward,done,info=env.step(action)
#       #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
#         print(colored("reward={},info={}".format(reward,info),"cyan"))
#        # print(colored("info={}".format(info),"cyan"))
#         sum_reward+=reward
#         if done:
#             break
#        # time.sleep(0.1)
# print()
# print(sum_reward)
