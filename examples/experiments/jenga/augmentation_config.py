"""
Augmentation configuration profiles for Jenga task.

This module provides different augmentation profiles optimized for the Jenga manipulation task.
The profiles balance between increasing visual diversity and preserving fine-grained features
needed for precise block extraction.
"""

# Conservative Profile - Recommended starting point
# Modest augmentations that preserve critical visual features while adding robustness
CONSERVATIVE = {
    "random_crop_padding": 4,
    "brightness_delta": 0.1,
    "brightness_prob": 0.7,
    "contrast_delta": 0.15,
    "contrast_prob": 0.7,
    "saturation_delta": 0.1,
    "saturation_prob": 0.5,
    "hue_delta": 0.03,
    "hue_prob": 0.3,
    "blur_prob": 0.05,
    "blur_sigma_range": (0.1, 0.3),
    "grayscale_prob": 0.02,
}

# Moderate Profile - Use if conservative works well and you want more diversity
# Increased augmentation strength while still being task-appropriate
MODERATE = {
    "random_crop_padding": 6,
    "brightness_delta": 0.15,
    "brightness_prob": 0.8,
    "contrast_delta": 0.2,
    "contrast_prob": 0.8,
    "saturation_delta": 0.15,
    "saturation_prob": 0.6,
    "hue_delta": 0.05,
    "hue_prob": 0.4,
    "blur_prob": 0.1,
    "blur_sigma_range": (0.1, 0.5),
    "grayscale_prob": 0.05,
}

# Aggressive Profile - Maximum augmentation (use with caution)
# Strong augmentations that may degrade fine-grained visual features
AGGRESSIVE = {
    "random_crop_padding": 8,
    "brightness_delta": 0.2,
    "brightness_prob": 0.9,
    "contrast_delta": 0.3,
    "contrast_prob": 0.9,
    "saturation_delta": 0.2,
    "saturation_prob": 0.7,
    "hue_delta": 0.1,
    "hue_prob": 0.5,
    "blur_prob": 0.2,
    "blur_sigma_range": (0.1, 1.0),
    "grayscale_prob": 0.1,
}

# Crop Only - Just random crop (same as original default)
# Use this to match the original augmentation behavior
CROP_ONLY = {
    "random_crop_padding": 4,
    "brightness_delta": 0.0,
    "brightness_prob": 0.0,
    "contrast_delta": 0.0,
    "contrast_prob": 0.0,
    "saturation_delta": 0.0,
    "saturation_prob": 0.0,
    "hue_delta": 0.0,
    "hue_prob": 0.0,
    "blur_prob": 0.0,
    "blur_sigma_range": (0.1, 0.5),
    "grayscale_prob": 0.0,
}

# No Augmentation - Disable all augmentations
NONE = {
    "random_crop_padding": 0,
    "brightness_delta": 0.0,
    "brightness_prob": 0.0,
    "contrast_delta": 0.0,
    "contrast_prob": 0.0,
    "saturation_delta": 0.0,
    "saturation_prob": 0.0,
    "hue_delta": 0.0,
    "hue_prob": 0.0,
    "blur_prob": 0.0,
    "blur_sigma_range": (0.1, 0.5),
    "grayscale_prob": 0.0,
}


# Test Profile - Very visible augmentations for visualization testing
# Use this to verify augmentations are working
TEST = {
    "random_crop_padding": 8,
    "brightness_delta": 0.3,
    "brightness_prob": 1.0,
    "contrast_delta": 0.4,
    "contrast_prob": 1.0,
    "saturation_delta": 0.3,
    "saturation_prob": 1.0,
    "hue_delta": 0.15,
    "hue_prob": 1.0,
    "blur_prob": 0.5,
    "blur_sigma_range": (0.5, 1.5),
    "grayscale_prob": 0.0,
}

# Profile registry
PROFILES = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
    "crop_only": CROP_ONLY,
    "none": NONE,
    "test": TEST,
}


def get_augmentation_config(profile_name="conservative"):
    """
    Get augmentation configuration by profile name.

    Args:
        profile_name: One of "conservative", "moderate", "aggressive", "crop_only", "none"

    Returns:
        Dictionary with augmentation parameters

    Raises:
        ValueError: If profile_name is not recognized
    """
    if profile_name is None:
        return NONE

    profile_name = profile_name.lower()
    if profile_name not in PROFILES:
        raise ValueError(
            f"Unknown augmentation profile: {profile_name}. "
            f"Available profiles: {list(PROFILES.keys())}"
        )

    return PROFILES[profile_name].copy()


def print_config(config):
    """Pretty print an augmentation configuration."""
    print("\n" + "=" * 60)
    print("Augmentation Configuration")
    print("=" * 60)

    print("\nSpatial Augmentations:")
    print(f"  Random Crop Padding: {config['random_crop_padding']}")

    print("\nPhotometric Augmentations:")
    print(f"  Brightness: delta=±{config['brightness_delta']}, prob={config['brightness_prob']}")
    print(f"  Contrast:   delta=±{config['contrast_delta']}, prob={config['contrast_prob']}")
    print(f"  Saturation: delta=±{config['saturation_delta']}, prob={config['saturation_prob']}")
    print(f"  Hue:        delta=±{config['hue_delta']}, prob={config['hue_prob']}")

    print("\nOther Augmentations:")
    print(f"  Gaussian Blur: prob={config['blur_prob']}, sigma={config['blur_sigma_range']}")
    print(f"  Grayscale:     prob={config['grayscale_prob']}")

    print("=" * 60 + "\n")
