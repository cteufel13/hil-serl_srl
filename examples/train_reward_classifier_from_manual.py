import glob
import os
import pickle as pkl
import math
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 300, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_string(
    "checkpoint_path", None, "Path to checkpoint folder with buffer and demo_buffer"
)
flags.DEFINE_boolean(
    "use_demo_as_positive", True, "Use demo_buffer transitions as positive examples"
)
flags.DEFINE_float("test_split", 0.1, "Fraction of data to use for testing")
flags.DEFINE_float(
    "max_neg_ratio",
    20.0,
    "Maximum negative-to-positive ratio to keep in memory (per split)",
)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    assert FLAGS.checkpoint_path is not None, "checkpoint_path must be provided"

    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
    demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")

    assert os.path.exists(buffer_path), f"Buffer path not found: {buffer_path}"
    assert os.path.exists(
        demo_buffer_path
    ), f"Demo buffer path not found: {demo_buffer_path}"

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Temporary buffers to collect all data
    all_pos_data = []
    all_neg_data = []

    # Load buffer transitions and label based on reward
    # reward > 0 means success, reward == 0 means failure
    print("Loading buffer (online experience, labeled by reward)...")
    buffer_files = glob.glob(os.path.join(buffer_path, "*.pkl"))
    for path in buffer_files:
        try:
            with open(path, "rb") as f:
                buffer_data = pkl.load(f)
            for trans in buffer_data:
                if "images" in trans.get("observations", {}).keys():
                    continue
                # Use reward field to determine label
                reward = trans.get("rewards", 0)
                if reward > 0:
                    all_pos_data.append(trans)
                else:
                    all_neg_data.append(trans)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # Load demo_buffer transitions and label based on reward
    print("Loading demo_buffer (intervention data, labeled by reward)...")
    demo_files = glob.glob(os.path.join(demo_buffer_path, "*.pkl"))
    for path in demo_files:
        try:
            with open(path, "rb") as f:
                demo_data = pkl.load(f)
            for trans in demo_data:
                if "images" in trans.get("observations", {}).keys():
                    continue
                # Use reward field to determine label
                reward = trans.get("rewards", 0)
                if reward > 0:
                    all_pos_data.append(trans)
                else:
                    all_neg_data.append(trans)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    print(f"Total positive samples: {len(all_pos_data)}")
    print(f"Total negative samples: {len(all_neg_data)}")

    # Split into train and test sets (90/10)
    rng_split = np.random.RandomState(42)
    rng_split.shuffle(all_pos_data)
    rng_split.shuffle(all_neg_data)
    split_idx_pos = int(len(all_pos_data) * (1 - FLAGS.test_split))
    split_idx_neg = int(len(all_neg_data) * (1 - FLAGS.test_split))

    pos_train = all_pos_data[:split_idx_pos]
    pos_test = all_pos_data[split_idx_pos:]
    neg_train_full = all_neg_data[:split_idx_neg]
    neg_test_full = all_neg_data[split_idx_neg:]

    # Downsample negatives to cap ratio for memory and balance
    max_neg_train = int(len(pos_train) * FLAGS.max_neg_ratio)
    max_neg_test = int(len(pos_test) * FLAGS.max_neg_ratio)
    neg_train = neg_train_full[: max(max_neg_train, 1)]
    neg_test = neg_test_full[: max(max_neg_test, 1)]

    print(f"\nTrain set (capped): {len(pos_train)} positive, {len(neg_train)} negative")
    print(f"Test set (capped): {len(pos_test)} positive, {len(neg_test)} negative")

    # Cache counts before freeing lists
    train_pos_count = len(pos_train)
    train_neg_count = len(neg_train)
    test_pos_count = len(pos_test)
    test_neg_count = len(neg_test)

    # Create training buffers sized to the data to avoid OOM
    pos_buffer_train = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=max(train_pos_count, 1),
        include_label=True,
    )
    neg_buffer_train = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=max(train_neg_count, 1),
        include_label=True,
    )

    for trans in pos_train:
        trans["labels"] = 1
        pos_buffer_train.insert(trans)
    for trans in neg_train:
        trans["labels"] = 0
        neg_buffer_train.insert(trans)

    # Create test buffers sized to the data to avoid OOM
    pos_buffer_test = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=max(test_pos_count, 1),
        include_label=True,
    )
    neg_buffer_test = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=max(test_neg_count, 1),
        include_label=True,
    )

    for trans in pos_test:
        trans["labels"] = 1
        pos_buffer_test.insert(trans)
    for trans in neg_test:
        trans["labels"] = 0
        neg_buffer_test.insert(trans)

    # Free raw lists to save memory
    del all_pos_data, all_neg_data, pos_train, neg_train, pos_test, neg_test

    pos_iterator_train = pos_buffer_train.get_iterator(
        sample_args={
            "batch_size": min(FLAGS.batch_size // 2, train_pos_count),
        },
        device=None,
    )

    neg_iterator_train = neg_buffer_train.get_iterator(
        sample_args={
            "batch_size": min(FLAGS.batch_size // 2, train_neg_count),
        },
        device=None,
    )

    # For test, create small-batch iterators to avoid OOM
    test_pos_bs = max(1, min(8, test_pos_count))
    test_neg_bs = max(1, min(16, test_neg_count))
    test_pos_iterator = pos_buffer_test.get_iterator(
        sample_args={
            "batch_size": test_pos_bs,
        },
        device=None,
    )
    test_neg_iterator = neg_buffer_test.get_iterator(
        sample_args={
            "batch_size": test_neg_bs,
        },
        device=None,
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator_train)
    neg_sample = next(neg_iterator_train)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(
        key,
        sample["observations"],
        config.classifier_keys,
    )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        labels = batch["labels"].squeeze(-1)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["observations"],
                rngs={"dropout": key},
                train=True,
            )
            # Unweighted binary cross entropy
            bce_loss = optax.sigmoid_binary_cross_entropy(logits, labels[..., None])
            return bce_loss.mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)

        logits = state.apply_fn(
            {"params": state.params},
            batch["observations"],
            train=False,
            rngs={"dropout": key},
        )
        preds = nn.sigmoid(logits).squeeze(-1)

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        pos_total = jnp.maximum(jnp.sum(pos_mask), 1.0)
        neg_total = jnp.maximum(jnp.sum(neg_mask), 1.0)

        pos_correct = jnp.sum((preds >= 0.5) * pos_mask)
        neg_correct = jnp.sum((preds < 0.5) * neg_mask)

        pos_acc = pos_correct / pos_total
        neg_acc = neg_correct / neg_total
        balanced_accuracy = (pos_acc + neg_acc) / 2.0

        return state.apply_gradients(grads=grads), loss, balanced_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator_train)
        neg_sample = next(neg_iterator_train)
        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_balanced_acc = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Balanced Acc: {train_balanced_acc:.4f}"
        )

    # Evaluate on test set in micro-batches to avoid OOM
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    num_eval_batches = max(
        1,
        math.ceil(test_pos_count / test_pos_bs),
        math.ceil(test_neg_count / test_neg_bs),
    )

    total_pos = jnp.array(0.0)
    total_neg = jnp.array(0.0)
    correct_pos = jnp.array(0.0)
    correct_neg = jnp.array(0.0)
    total_all = jnp.array(0.0)
    correct_all = jnp.array(0.0)

    for _ in range(num_eval_batches):
        test_pos_batch = next(test_pos_iterator)
        test_neg_batch = next(test_neg_iterator)
        test_batch = concat_batches(test_pos_batch, test_neg_batch, axis=0)

        labels_test = test_batch["labels"]
        if labels_test.ndim > 1:
            labels_test = labels_test.squeeze(-1)
        else:
            labels_test = labels_test.reshape(-1)

        test_logits = classifier.apply_fn(
            {"params": classifier.params},
            test_batch["observations"],
            train=False,
        )
        test_preds = nn.sigmoid(test_logits).squeeze(-1)

        test_pos_mask = labels_test > 0.5
        test_neg_mask = ~test_pos_mask

        pos_total_batch = jnp.sum(test_pos_mask)
        neg_total_batch = jnp.sum(test_neg_mask)

        correct_pos_batch = jnp.sum((test_preds >= 0.85) * test_pos_mask)
        correct_neg_batch = jnp.sum((test_preds < 0.85) * test_neg_mask)

        total_pos += pos_total_batch
        total_neg += neg_total_batch
        correct_pos += correct_pos_batch
        correct_neg += correct_neg_batch

        correct_all += jnp.sum((test_preds >= 0.5) == labels_test)
        total_all += labels_test.shape[0]

    total_pos = jnp.maximum(total_pos, 1.0)
    total_neg = jnp.maximum(total_neg, 1.0)

    test_pos_acc = correct_pos / total_pos
    test_neg_acc = correct_neg / total_neg
    test_balanced_acc = (test_pos_acc + test_neg_acc) / 2.0
    test_overall_acc = correct_all / jnp.maximum(total_all, 1.0)

    print(f"Test Overall Accuracy: {test_overall_acc:.4f}")
    print(f"Test Positive Class Accuracy: {test_pos_acc:.4f}")
    print(f"Test Negative Class Accuracy: {test_neg_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
    print("=" * 60 + "\n")

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(
        f"Classifier checkpoint saved to {os.path.join(FLAGS.checkpoint_path, 'classifier_ckpt/')}"
    )


if __name__ == "__main__":
    app.run(main)
