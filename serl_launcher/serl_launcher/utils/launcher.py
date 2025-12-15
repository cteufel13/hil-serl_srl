# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import (
    batched_random_crop,
    color_transform,
    gaussian_blur,
    random_cutout,
    random_erasing,
    additive_gaussian_noise,
    posterize,
)

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    augmentation_function=None,
):
    # Use default augmentation if none provided
    if augmentation_function is None:
        augmentation_function = make_batch_augmentation_func(image_keys)

    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=augmentation_function,
    )
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch

    return augment_batch


def make_enhanced_augmentation_func(image_keys, augmentation_config=None) -> callable:
    """
    Create an enhanced augmentation function with configurable photometric and spatial augmentations.

    Args:
        image_keys: List of image observation keys to augment
        augmentation_config: Dict with augmentation parameters. If None, uses simple random crop.
            Expected keys:
                - random_crop_padding: int, padding for random crop (default: 4)
                - brightness_delta: float, max brightness adjustment (default: 0.0)
                - brightness_prob: float, probability of applying brightness (default: 0.0)
                - contrast_delta: float, max contrast adjustment (default: 0.0)
                - contrast_prob: float, probability of applying contrast (default: 0.0)
                - saturation_delta: float, max saturation adjustment (default: 0.0)
                - saturation_prob: float, probability of applying saturation (default: 0.0)
                - hue_delta: float, max hue adjustment (default: 0.0)
                - hue_prob: float, probability of applying hue (default: 0.0)
                - blur_prob: float, probability of applying gaussian blur (default: 0.0)
                - blur_sigma_range: tuple, (min, max) sigma for gaussian blur (default: (0.1, 0.5))
                - grayscale_prob: float, probability of converting to grayscale (default: 0.0)
                - cutout_prob: float, probability of applying random cutout (default: 0.0)
                - cutout_size_range: tuple, (min, max) size as fraction of image (default: (0.05, 0.15))
                - cutout_fill_value: float, fill value for cutout region (default: 0.5)
                - erasing_prob: float, probability of applying random erasing (default: 0.0)
                - erasing_size_range: tuple, (min, max) size as fraction of image (default: (0.05, 0.15))
                - gaussian_noise_prob: float, probability of applying additive gaussian noise (default: 0.0)
                - gaussian_noise_std_range: tuple, (min, max) std for gaussian noise (default: (0.0, 0.02))
                - posterize_prob: float, probability of applying posterize (default: 0.0)
                - posterize_bits_range: tuple, (min, max) bits to keep (default: (4, 6))

    Returns:
        Callable augmentation function that takes (batch, rng) and returns augmented batch
    """
    if augmentation_config is None:
        augmentation_config = {}

    # Extract config with defaults
    crop_padding = augmentation_config.get("random_crop_padding", 4)
    brightness_delta = augmentation_config.get("brightness_delta", 0.0)
    brightness_prob = augmentation_config.get("brightness_prob", 0.0)
    contrast_delta = augmentation_config.get("contrast_delta", 0.0)
    contrast_prob = augmentation_config.get("contrast_prob", 0.0)
    saturation_delta = augmentation_config.get("saturation_delta", 0.0)
    saturation_prob = augmentation_config.get("saturation_prob", 0.0)
    hue_delta = augmentation_config.get("hue_delta", 0.0)
    hue_prob = augmentation_config.get("hue_prob", 0.0)
    blur_prob = augmentation_config.get("blur_prob", 0.0)
    blur_sigma_range = augmentation_config.get("blur_sigma_range", (0.1, 0.5))
    grayscale_prob = augmentation_config.get("grayscale_prob", 0.0)

    # New augmentations
    cutout_prob = augmentation_config.get("cutout_prob", 0.0)
    cutout_size_range = augmentation_config.get("cutout_size_range", (0.05, 0.15))
    cutout_fill_value = augmentation_config.get("cutout_fill_value", 0.5)
    erasing_prob = augmentation_config.get("erasing_prob", 0.0)
    erasing_size_range = augmentation_config.get("erasing_size_range", (0.05, 0.15))
    gaussian_noise_prob = augmentation_config.get("gaussian_noise_prob", 0.0)
    gaussian_noise_std_range = augmentation_config.get("gaussian_noise_std_range", (0.0, 0.02))
    posterize_prob = augmentation_config.get("posterize_prob", 0.0)
    posterize_bits_range = augmentation_config.get("posterize_bits_range", (4, 6))

    # Determine if color augmentation is enabled
    color_aug_enabled = (
        brightness_delta > 0 or contrast_delta > 0 or
        saturation_delta > 0 or hue_delta > 0 or grayscale_prob > 0
    )

    def data_augmentation_fn(rng, observations):
        """Apply augmentations to a single observation dict."""
        for pixel_key in image_keys:
            img = observations[pixel_key]

            # Split RNG for different augmentation types
            rng, crop_rng, color_rng, blur_rng = jax.random.split(rng, 4)

            # 1. Random crop (always applied if padding > 0)
            if crop_padding > 0:
                img = batched_random_crop(img, crop_rng, padding=crop_padding, num_batch_dims=2)

            # 2. Color augmentation (if enabled)
            if color_aug_enabled:
                # Normalize to [0, 1] for color transforms
                img_normalized = img / 255.0

                # Apply color transform to each image in batch
                def apply_color_to_single(single_img, single_rng):
                    return color_transform(
                        single_img,
                        single_rng,
                        brightness=brightness_delta,
                        contrast=contrast_delta,
                        saturation=saturation_delta,
                        hue=hue_delta,
                        to_grayscale_prob=grayscale_prob,
                        color_jitter_prob=1.0,  # Apply color jitter with individual probs
                        apply_prob=max(brightness_prob, contrast_prob, saturation_prob, hue_prob),
                        shuffle=True,
                    )

                # Get batch size and flatten batch dimensions
                batch_shape = img_normalized.shape[:2]  # (batch_size, temporal_stack)
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])

                # Split RNG for each image in batch
                color_rngs = jax.random.split(color_rng, flat_batch_size)

                # Apply color transform to each image
                img_colored = jax.vmap(apply_color_to_single)(img_flat, color_rngs)

                # Reshape back and denormalize
                img = img_colored.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # 3. Gaussian blur (if enabled)
            if blur_prob > 0:
                # Normalize to [0, 1] for blur
                img_normalized = img / 255.0

                def apply_blur_to_single(single_img, single_rng):
                    # gaussian_blur expects image in HWC format
                    return gaussian_blur(
                        single_img,
                        single_rng,
                        sigma_min=blur_sigma_range[0],
                        sigma_max=blur_sigma_range[1],
                        apply_prob=blur_prob,
                    )

                # Flatten batch dimensions
                batch_shape = img_normalized.shape[:2]
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])

                # Split RNG for each image
                blur_rngs = jax.random.split(blur_rng, flat_batch_size)

                # Apply blur to each image - vmap over both img and rng
                img_blurred = jax.vmap(apply_blur_to_single, in_axes=(0, 0))(img_flat, blur_rngs)

                # Reshape back and denormalize
                img = img_blurred.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # 4. New augmentations (cutout, erasing, noise, posterize)
            rng, cutout_rng, erasing_rng, noise_rng, posterize_rng = jax.random.split(rng, 5)

            # Random Cutout
            if cutout_prob > 0:
                img_normalized = img / 255.0

                def apply_cutout_to_single(single_img, single_rng):
                    return random_cutout(
                        single_img,
                        single_rng,
                        min_size=cutout_size_range[0],
                        max_size=cutout_size_range[1],
                        fill_value=cutout_fill_value,
                        apply_prob=cutout_prob,
                    )

                batch_shape = img_normalized.shape[:2]
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])
                cutout_rngs = jax.random.split(cutout_rng, flat_batch_size)
                img_cutout = jax.vmap(apply_cutout_to_single, in_axes=(0, 0))(img_flat, cutout_rngs)
                img = img_cutout.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # Random Erasing
            if erasing_prob > 0:
                img_normalized = img / 255.0

                def apply_erasing_to_single(single_img, single_rng):
                    return random_erasing(
                        single_img,
                        single_rng,
                        min_size=erasing_size_range[0],
                        max_size=erasing_size_range[1],
                        apply_prob=erasing_prob,
                    )

                batch_shape = img_normalized.shape[:2]
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])
                erasing_rngs = jax.random.split(erasing_rng, flat_batch_size)
                img_erased = jax.vmap(apply_erasing_to_single, in_axes=(0, 0))(img_flat, erasing_rngs)
                img = img_erased.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # Additive Gaussian Noise
            if gaussian_noise_prob > 0:
                img_normalized = img / 255.0

                def apply_noise_to_single(single_img, single_rng):
                    return additive_gaussian_noise(
                        single_img,
                        single_rng,
                        std_min=gaussian_noise_std_range[0],
                        std_max=gaussian_noise_std_range[1],
                        apply_prob=gaussian_noise_prob,
                    )

                batch_shape = img_normalized.shape[:2]
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])
                noise_rngs = jax.random.split(noise_rng, flat_batch_size)
                img_noisy = jax.vmap(apply_noise_to_single, in_axes=(0, 0))(img_flat, noise_rngs)
                img = img_noisy.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # Posterize
            if posterize_prob > 0:
                img_normalized = img / 255.0

                def apply_posterize_to_single(single_img, single_rng):
                    return posterize(
                        single_img,
                        single_rng,
                        min_bits=posterize_bits_range[0],
                        max_bits=posterize_bits_range[1],
                        apply_prob=posterize_prob,
                    )

                batch_shape = img_normalized.shape[:2]
                flat_batch_size = batch_shape[0] * batch_shape[1]
                img_flat = img_normalized.reshape(flat_batch_size, *img_normalized.shape[2:])
                posterize_rngs = jax.random.split(posterize_rng, flat_batch_size)
                img_posterized = jax.vmap(apply_posterize_to_single, in_axes=(0, 0))(img_flat, posterize_rngs)
                img = img_posterized.reshape(*batch_shape, *img_normalized.shape[2:]) * 255.0

            # Update observations with augmented image
            try:
                # Try FrozenDict API first
                observations = observations.copy(add_or_replace={pixel_key: img})
            except TypeError:
                # Fall back to regular dict mutation
                observations[pixel_key] = img

        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        """Augment both observations and next_observations in batch."""
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])

        # Update batch - handle both dict and FrozenDict
        try:
            # Try FrozenDict API first
            batch = batch.copy(
                add_or_replace={
                    "observations": obs,
                    "next_observations": next_obs,
                }
            )
        except TypeError:
            # Fall back to regular dict
            batch = dict(batch)
            batch["observations"] = obs
            batch["next_observations"] = next_obs
        return batch

    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
