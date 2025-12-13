import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    SpacemouseBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.jenga.wrapper import JengaEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://172.16.0.254:5000/"
    REALSENSE_CAMERAS = {
        "side_right": {
            "serial_number": "1944301041E91D1300",
            "dim": (640, 480),
            "exposure": 5000,
        },
        "side_left": {
            "serial_number": "18443010011AAA0F00",
            "dim": (640, 480),
            "exposure": 5000,
        },
        "wrist": {
            "serial_number": "18443010212EE4F400",
            "dim": (640, 480),
            "exposure": 5000,
        },
    }
    IMAGE_CROP = {
        "side_right": lambda img: img[:, 320:],
        "side_left": lambda img: img[:, :],
        "wrist": lambda img: img[:, :],
    }
    TARGET_POSE = np.array([0.5335188982150202,-0.16665013289964375,0.21339164520982049,3.112483273997248,-0.07797066034665501,-0.1941881137648624])
    GRASP_POSE = np.array([0.5335851277669401,0.03242347368110595,0.20985455757034097,3.108454670187885,-0.07586641998740595,-0.12028077382033389])
    RESET_POSE = np.array([0.46519913336014634,-0.01173795084839937,0.22210094593842042,3.1324504022307385,-0.035296294605906775,0.0359773807726573])
    ABS_POSE_LIMIT_LOW = np.array([0.45688505515254313,-0.2588072949730278,0.13016198377475022,3.085728469905128,-0.06678723124371899,-0.4451142400130191])
    ABS_POSE_LIMIT_HIGH = np.array([0.6105772466980874,0.1460277619415316,0.2550206690408382,3.106672914231177,0.2865488397694822,0.298424371180356])

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.015, 0.05, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 150
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_right", "side_left", "wrist"]
    classifier_keys = ["side_right", "side_left", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = JengaEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            env = SpacemouseBinaryRewardClassifierWrapper(env)
        return env