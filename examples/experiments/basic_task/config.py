import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.basic_task.wrapper import BasicEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://172.16.0.254:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "1944301041B4992E00",
            "dim": (1280, 720),
            "exposure": 5000,
        },
        "wrist_2": {
            "serial_number": "1944301041E91D1300", #18443010212EE4F400 1944301041E91D1300
            "dim": (1280, 720),
            "exposure": 5000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:, :],
        "wrist_2": lambda img: img[:, :],
    }
    TARGET_POSE = np.array([0.4,-0.15,0.02, np.pi, 0, np.pi/2])
    GRASP_POSE = np.array([0.6,-0.15,0.02, np.pi, 0, np.pi/2])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.1,0.1,0.005,0,0,0])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3,0.1,0.1,0,0,0])

    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.07, 0.1, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
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
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = BasicEnv(
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
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))

                pred = sigmoid(classifier(obs)).item()
                zpos = float(obs['state'][0, 6])

                print(int(pred > 0.85), pred)

                return int(pred > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env