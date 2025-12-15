#!/usr/bin/env python3
"""
Visualization script for data augmentations in the Jenga task.

This script loads sample images from demonstration data and visualizes
the effects of different augmentation profiles.

Usage:
    python visualize_augmentations.py --profile conservative
    python visualize_augmentations.py --profile moderate --save_path my_viz.png
"""

import argparse
import pickle as pkl
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import augmentation config and functions
from augmentation_config import get_augmentation_config, print_config, PROFILES
from serl_launcher.utils.launcher import make_enhanced_augmentation_func


def load_demo_data(demo_path):
    """Load demonstration data from pickle file."""
    print(f"Loading demo data from {demo_path}...")
    with open(demo_path, "rb") as f:
        demo_data = pkl.load(f)
    print(f"Loaded {len(demo_data)} transitions")
    return demo_data


def extract_sample_images(demo_data, image_keys, num_samples=3):
    """
    Extract sample images from demo data.

    Args:
        demo_data: List of transition dictionaries
        image_keys: List of image keys to extract
        num_samples: Number of sample timesteps to extract

    Returns:
        Dictionary mapping image_key -> list of images
    """
    images = {key: [] for key in image_keys}

    # Get total number of transitions
    total_transitions = len(demo_data)

    # Sample evenly spaced indices
    indices = np.linspace(0, total_transitions - 1, num_samples, dtype=int)

    for idx in indices:
        transition = demo_data[idx]
        for key in image_keys:
            img = transition['observations'][key]
            # Handle temporal stacking - take the first frame if stacked
            if len(img.shape) == 4:  # (temporal_stack, H, W, C)
                img = img[0]
            images[key].append(img)

    return images


def create_augmented_batch(images, image_keys):
    """
    Create a batch from sample images in the format expected by augmentation functions.

    Args:
        images: Dict of image_key -> list of images
        image_keys: List of image keys

    Returns:
        Batch dictionary with observations
    """
    # Stack images into batch (num_samples, 1, H, W, C)
    # The 1 is for temporal dimension (obs_horizon=1 in jenga config)
    batch_images = {}
    for key in image_keys:
        imgs = np.stack(images[key], axis=0)  # (num_samples, H, W, C)
        imgs = imgs[:, np.newaxis, :, :, :]  # (num_samples, 1, H, W, C)
        batch_images[key] = jnp.array(imgs)

    # Create batch in the format expected by augmentation function
    batch = {
        "observations": batch_images,
        "next_observations": batch_images.copy(),  # Not used for visualization
    }

    return batch


def apply_augmentations(batch, augmentation_func, num_augmentations=5):
    """
    Apply augmentation multiple times to generate diverse examples.

    Args:
        batch: Batch dictionary
        augmentation_func: Augmentation function
        num_augmentations: Number of augmented versions to generate

    Returns:
        List of augmented batches
    """
    augmented_batches = []

    for i in range(num_augmentations):
        rng = jax.random.PRNGKey(i)  # Different seed for each augmentation
        # Create a deep copy of batch to avoid in-place mutations
        import copy
        batch_copy = copy.deepcopy(batch)
        aug_batch = augmentation_func(batch_copy, rng)
        augmented_batches.append(aug_batch)

    return augmented_batches


def visualize_augmentations(
    original_batch,
    augmented_batches,
    image_keys,
    profile_name,
    save_path=None
):
    """
    Create a visualization grid showing original and augmented images.

    Args:
        original_batch: Original batch without augmentation
        augmented_batches: List of augmented batches
        image_keys: List of image keys to visualize
        profile_name: Name of the augmentation profile
        save_path: Path to save the visualization (if None, displays interactively)
    """
    num_samples = original_batch["observations"][image_keys[0]].shape[0]
    num_augmentations = len(augmented_batches)
    num_cameras = len(image_keys)

    # Create figure with subplots
    # Rows: samples, Columns: (original + augmentations) * num_cameras
    fig, axes = plt.subplots(
        num_samples,
        (num_augmentations + 1) * num_cameras,
        figsize=(4 * (num_augmentations + 1) * num_cameras, 4 * num_samples)
    )

    # Ensure axes is 2D even for single sample
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Data Augmentation Visualization - Profile: {profile_name.upper()}',
        fontsize=16,
        fontweight='bold'
    )

    for sample_idx in range(num_samples):
        for cam_idx, image_key in enumerate(image_keys):
            # Column offset for this camera
            col_offset = cam_idx * (num_augmentations + 1)

            # Original image
            orig_img = np.array(original_batch["observations"][image_key][sample_idx, 0])
            orig_img = np.clip(orig_img / 255.0, 0, 1)  # Normalize to [0, 1]

            ax = axes[sample_idx, col_offset]
            ax.imshow(orig_img)
            ax.axis('off')
            if sample_idx == 0:
                ax.set_title(f'{image_key}\nOriginal', fontsize=12, fontweight='bold')

            # Augmented images
            for aug_idx, aug_batch in enumerate(augmented_batches):
                aug_img = np.array(aug_batch["observations"][image_key][sample_idx, 0])
                aug_img = np.clip(aug_img / 255.0, 0, 1)  # Normalize to [0, 1]

                ax = axes[sample_idx, col_offset + aug_idx + 1]
                ax.imshow(aug_img)
                ax.axis('off')
                if sample_idx == 0:
                    ax.set_title(f'{image_key}\nAug {aug_idx + 1}', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize data augmentations for Jenga task')
    parser.add_argument(
        '--profile',
        type=str,
        default='conservative',
        choices=list(PROFILES.keys()),
        help='Augmentation profile to visualize'
    )
    parser.add_argument(
        '--demo_path',
        type=str,
        default='demo_data/jenga_20_demos_2025-12-12_12-12-59.pkl',
        help='Path to demo data pickle file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of sample images to visualize'
    )
    parser.add_argument(
        '--num_augmentations',
        type=int,
        default=5,
        help='Number of augmented versions to generate per sample'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='augmentation_preview.png',
        help='Path to save visualization (set to empty string to display instead)'
    )

    args = parser.parse_args()

    # Image keys for Jenga task
    image_keys = ["side_right", "side_left", "wrist"]

    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    demo_path = script_dir / args.demo_path

    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        print(f"Please check the path or download demo data first.")
        return

    # Load demo data
    demo_data = load_demo_data(demo_path)

    # Extract sample images
    print(f"\nExtracting {args.num_samples} sample images...")
    sample_images = extract_sample_images(demo_data, image_keys, args.num_samples)

    # Get augmentation configuration
    aug_config = get_augmentation_config(args.profile)
    print_config(aug_config)

    # Create augmentation function
    if args.profile == 'none':
        print("No augmentation will be applied (profile='none')")
        augmentation_func = None
    else:
        augmentation_func = make_enhanced_augmentation_func(image_keys, aug_config)

    # Create batch from sample images
    original_batch = create_augmented_batch(sample_images, image_keys)

    if augmentation_func is None:
        # Just visualize original images
        print("\nVisualizing original images (no augmentation)...")
        visualize_augmentations(
            original_batch,
            [],  # No augmented versions
            image_keys,
            args.profile,
            args.save_path if args.save_path else None
        )
    else:
        # Apply augmentations
        print(f"\nGenerating {args.num_augmentations} augmented versions...")
        augmented_batches = apply_augmentations(
            original_batch,
            augmentation_func,
            args.num_augmentations
        )

        # Visualize
        print("Creating visualization...")
        save_path = script_dir / args.save_path if args.save_path else None
        visualize_augmentations(
            original_batch,
            augmented_batches,
            image_keys,
            args.profile,
            save_path
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
