from typing import Iterator, Tuple, Any, Dict

import glob
import h5py
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image
import json
import argparse

DEFAULT_CONFIG_PATH = "/home/wangxianhao/data/project/reasoning/rlds_dataset_builder/tabletop_dataset/dataset_configs.json"

class PrimitiveDatasetV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Tabletop-Lift-Book dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = DEFAULT_CONFIG_PATH
        print("Using config path:", self.config_path)

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            self.dataset_configs = self.config['datasets']
            print(f"Loaded {len(self.dataset_configs)} dataset configs")

        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Base camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),
                        'base_front_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Base front camera RGB observation.',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint positions (qpos).',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, includes joint positions and tcp pose.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of joint velocities and gripper control.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        # Cache for resized images to avoid repeated computation
        image_cache = {}

        def _parse_example(episode_path, instruction, language_embedding):
            try:
                print(f"Processing episode: {episode_path}")
                # Load HDF5 file
                with h5py.File(episode_path, 'r') as f:
                    # Get actions and determine number of steps
                    actions = np.array(f['traj_0/actions'], dtype=np.float32)
                    action_shape = actions.shape[1]
                    actions = np.vstack([actions, np.zeros((1, action_shape), dtype=np.float32)])
                    num_steps = len(actions)

                    # Get RGB images
                    base_rgb = np.array(f['traj_0/obs/sensor_data/base_camera/rgb'])
                    base_front_rgb = np.array(f['traj_0/obs/sensor_data/base_front_camera/rgb'])
                    hand_rgb = np.array(f['traj_0/obs/sensor_data/hand_camera/rgb'])

                    # Get robot state
                    qpos = np.array(f['traj_0/obs/agent/qpos'])
                    tcp_pose = np.array(f['traj_0/obs/extra/tcp_pose']) if 'traj_0/obs/extra/tcp_pose' in f else np.zeros((num_steps + 1, 7))

                    # Get rewards, success, terminated, truncated
                    rewards = np.array(f['traj_0/rewards']) if 'traj_0/rewards' in f else np.zeros(num_steps)
                    success = np.array(f['traj_0/success']) if 'traj_0/success' in f else np.zeros(num_steps, dtype=bool)
                    terminated = np.array(f['traj_0/terminated']) if 'traj_0/terminated' in f else np.zeros(num_steps, dtype=bool)
                    truncated = np.array(f['traj_0/truncated']) if 'traj_0/truncated' in f else np.zeros(num_steps, dtype=bool)

                    # Assemble episode
                    episode = []
                    for i in range(num_steps):
                        # Resize images to 224x224 using TensorFlow for GPU acceleration
                        base_img = tf.image.resize(base_rgb[i], [224, 224], method='lanczos3').numpy().astype(np.uint8) 
                        base_front_img = tf.image.resize(base_front_rgb[i], [224, 224], method='lanczos3').numpy().astype(np.uint8)
                        hand_img = tf.image.resize(hand_rgb[i], [224, 224], method='lanczos3').numpy().astype(np.uint8)

                        # Extract joint state (first 7 dimensions of qpos)
                        joint_state = qpos[i][:7] if qpos[i].shape[0] >= 7 else np.pad(qpos[i], (0, 7 - qpos[i].shape[0]))

                        # Create state vector (8-dimensional)
                        state = qpos[i][:8] if qpos[i].shape[0] >= 8 else np.pad(qpos[i], (0, 8 - qpos[i].shape[0]))

                        episode.append({
                            'observation': {
                                'image': base_img,
                                'wrist_image': hand_img,
                                'base_front_image': base_front_img,
                                'joint_state': joint_state,
                                'state': state,
                            },
                            'action': actions[i],
                            'discount': 1.0,
                            'reward': float(rewards[i]) if i < len(rewards) else 0.0,
                            'is_first': i == 0,
                            'is_last': i == (num_steps - 1),
                            'is_terminal': bool(terminated[i]) if i < len(terminated) else (i == num_steps - 1),
                            'language_instruction': instruction,
                            'language_embedding': language_embedding,
                        })

                    # Create output data sample
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path
                        }
                    }
                    print(f"Finished processing episode: {episode_path}")
                    return episode_path, sample
            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                return None

        # Iterate over dataset configs and generate examples
        for dataset_config in self.dataset_configs:
            hdf5_dir = dataset_config['hdf5_dir']
            instruction = dataset_config['instruction']
            print(f"Processing dataset: {dataset_config.get('name', hdf5_dir)} with instruction: {instruction}")
            # Pre-compute language embedding once per instruction
            language_embedding = self._embed([instruction])[0].numpy()

            episode_paths = glob.glob(os.path.join(hdf5_dir, '**/*.h5'), recursive=True)
            print(f"Found {len(episode_paths)} HDF5 files in {hdf5_dir}")
            for episode_path in episode_paths:
                result = _parse_example(episode_path, instruction, language_embedding)
                if result is not None:
                    yield result

def main():
    parser = argparse.ArgumentParser(description='Build RLDS dataset from HDF5 files')
    parser.add_argument('--config_path', type=str, default='dataset_configs.json',
                        help='Path to config JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to save the dataset')
    
    args = parser.parse_args()
    
    builder = PrimitiveDatasetV1(config_path=args.config_path, data_dir=args.data_dir)
    builder.download_and_prepare()
    builder.as_dataset()

if __name__ == '__main__':
    main()