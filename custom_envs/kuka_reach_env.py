#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   kuka_reach_env.py
@Time    :   2021/03/20 14:33:24
@Author  :   Yan Wen 
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
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


# import ray
# from ray import tune
# from ray.tune import grid_search
# from ray.rllib.env.env_context import EnvContext
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.utils.test_utils import check_learning_achieved

# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()


class KukaReachEnv(gym.Env):
    #在Gym中，每个环境都可以包含一个元数据（metadata）字典，用于描述环境的属性，例如渲染模式和视频帧率等。
    #在这个例子中，metadata字典定义了环境支持的渲染模式和视频帧率。
    #具体来说，这个环境支持两种渲染模式：human（人类可视化）和rgb_array（返回RGB图像数组），视频帧率为每秒50帧。
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, config):

        self.max_steps_one_episode = config["max_steps_one_episode"]
        self.is_render = config["is_render"]
        self.is_good_view = config["is_good_view"]#自己设定的，如果is_good_view机械臂动作会变慢方便观察

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs = -1.
        self.x_high_obs = 1.0
        self.y_low_obs = -1.
        self.y_high_obs = 1.
        self.z_low_obs = -0.1
        self.z_high_obs = 1.

        self.x_low_obs_for_judge = 0.2
        self.x_high_obs_for_judge = 0.7
        self.y_low_obs_for_judge = -0.3
        self.y_high_obs_for_judge = 0.3
        self.z_low_obs_for_judge = 0
        self.z_high_obs_for_judge = 0.55

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        #动作空间   ->   执行动作
        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)
        #观测空间  ->   获取环境状态
        self.observation_space = spaces.Box(
            low=np.array([
                self.x_low_obs, self.y_low_obs, self.z_low_obs
            ]),
            high=np.array([
                self.x_high_obs, self.y_high_obs, self.z_high_obs
            ]),
            dtype=np.float32)

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

        #self.seed(config['worker_index'] * config['num_workers'])
        self.seed()

    # self.reset()

    # def seed(self, seed=None):
    #     random.seed(seed)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# reset函数是较为重要的一个函数，它定义了机械臂初始化时的状态、机械臂的初始化、observation等。
    def reset(self):
        # p.connect(p.GUI)
        self.step_counter = 0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        # 这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(lineFromXYZ=[
            self.x_low_obs_for_judge, self.y_low_obs_for_judge, 0
        ],
                           lineToXYZ=[
                               self.x_low_obs_for_judge,
                               self.y_low_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_low_obs_for_judge, self.y_high_obs_for_judge, 0
        ],
                           lineToXYZ=[
                               self.x_low_obs_for_judge,
                               self.y_high_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_high_obs_for_judge, self.y_low_obs_for_judge, 0
        ],
                           lineToXYZ=[
                               self.x_high_obs_for_judge,
                               self.y_low_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_high_obs_for_judge, self.y_high_obs_for_judge, 0
        ],
                           lineToXYZ=[
                               self.x_high_obs_for_judge,
                               self.y_high_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])

        p.addUserDebugLine(lineFromXYZ=[
            self.x_low_obs_for_judge, self.y_low_obs_for_judge,
            self.z_high_obs_for_judge
        ],
                           lineToXYZ=[
                               self.x_high_obs_for_judge,
                               self.y_low_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_low_obs_for_judge, self.y_high_obs_for_judge,
            self.z_high_obs_for_judge
        ],
                           lineToXYZ=[
                               self.x_high_obs_for_judge,
                               self.y_high_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_low_obs_for_judge, self.y_low_obs_for_judge,
            self.z_high_obs_for_judge
        ],
                           lineToXYZ=[
                               self.x_low_obs_for_judge,
                               self.y_high_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])
        p.addUserDebugLine(lineFromXYZ=[
            self.x_high_obs_for_judge, self.y_low_obs_for_judge,
            self.z_high_obs_for_judge
        ],
                           lineToXYZ=[
                               self.x_high_obs_for_judge,
                               self.y_high_obs_for_judge,
                               self.z_high_obs_for_judge
                           ])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                               "kuka_iiwa/model.urdf"),
                                  useFixedBase=True)
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"),
                   basePosition=[0.5, 0, -0.65])
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        #object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
        self.object_id = p.loadURDF(
            os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
            basePosition=[
                random.uniform(self.x_low_obs_for_judge,
                               self.x_high_obs_for_judge),
                random.uniform(self.y_low_obs_for_judge,
                               self.y_high_obs_for_judge), 0.01
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
        # logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        # self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        # self.robot_pos=p.getLinkState(self.kuka_id,self.num_joints-1)[4]
        # return np.array(self.object_pos+self.robot_pos).astype(np.float32)
        # return np.array(self.robot_pos_obs).astype(np.float32)
        return self._get_observation()

# step函数也是一个非常重要的函数，它主要定义了机械臂根据强化学习算法生成的action，来决定自己怎样进行动作。其中主要涉及到了机械臂的控制、逆解等。
# 这里的策略是把算法生成的action进行很小的分割，避免因为action的输出离散过大，造成机械臂突然过大幅度的运动。然后将分割后的action作为变化值，与之前的位置信息叠加作为最后的位置值。
    def step(self, action):
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        self.new_robot_pos = [                                            # 这里就是把sample出来的动作缩小然后加上去
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        # logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))
        self.robot_joint_positions = p.calculateInverseKinematics(        # PyBullet内置的逆运动学求解器
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

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1
        return self._reward()

    def _get_observation(self):
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        self.robot_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        observation = self.object_pos
        # print(np.array(self.object_pos+self.robot_pos).astype(np.float32))
        return np.array(observation).astype(np.float32)

# 重中之重
    # （1）如果机械臂末端的位置超出了工作空间，给一个-0.1的奖励；
    # （2）如果在达到一个episode允许的最大step还没达到reach的位置，给一个-0.1的奖励；
    # （3）如果机械臂在允许的step、允许的工作空间范围内reach了物体，给一个+1的奖励
    def _reward(self):

        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]  # num_joints - 1表示末端link | [4]表示linkFramePosition，位置xyz
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)
        #
        self.object_state = np.array(                                  #获取指定物体的位置和方向信息，并将其存储在一个 Numpy 数组中。
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        square_dx = (self.robot_state[0] - self.object_state[0])**2  #(tcp - reach)^2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)     #计算距离
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚     (定义的白色空间)
        terminated = bool(x < self.x_low_obs_for_judge
                          or x > self.x_high_obs_for_judge
                          or y < self.y_low_obs_for_judge
                          or y > self.y_high_obs_for_judge
                          or z < self.z_low_obs_for_judge
                          or z > self.z_high_obs_for_judge)

        if terminated:                                              #超出边界  - 0.1
            reward = -0.1
            self.terminated = True

        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.max_steps_one_episode:        #超出步数
            reward = -0.1
            self.terminated = True

        elif self.distance < 0.1:                                   #到达位置
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        info = {}
        # self.observation=self.robot_state
        self.observation = self.object_state
        return self._get_observation(), reward, self.terminated, info

    def close(self):
        p.disconnect()


if __name__ == '__main__':
    env_config = {
        "is_render": False,
        "is_good_view": False,
        "max_steps_one_episode": 1000
    }

    config = EnvContext(env_config=env_config, worker_index=1, num_workers=1)
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数        （baseline是一个强化学习库）
    env = KukaReachEnv(config)
    print(env)
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    print(obs)

    sum_reward = 0
    for i in range(10000):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            # action=np.array([0,0,0.47-i/1000])
            obs, reward, done, info = env.step(action)
            #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            print('obs=', obs)
            #print("reward={},info={}".format(reward, info))
            # print(colored("info={}".format(info),"cyan"))
            sum_reward += reward
            if done:
                break
        # time.sleep(0.1)
    print()
    print(sum_reward)
