from typing import Iterator, Tuple, Any

import glob
import h5py
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image


class TabletopDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Tabletop-PickPlace dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the language embedding model
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
                            doc='Base front camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
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
        # You can modify this to create train/val splits if needed
        return {
            'train': self._generate_examples(path='/media/raid/workspace/wangxianhao/project/reasoning/Datasets/datasets/Tabletop-Lift-Book-v1/ar_teleop'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load HDF5 file
            with h5py.File(episode_path, 'r') as f:
                # Get actions and determine number of steps
                actions = np.array(f['traj_0/actions'])
                num_steps = len(actions)
                
                # Get RGB images
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
                
                # Create a default language instruction (modify as needed)
                language_instruction = "Lift the book"
                language_embedding = self._embed([language_instruction])[0].numpy()
                
                # Assemble episode
                episode = []
                for i in range(num_steps):
                    # Resize images to 224x224
                    base_img = Image.fromarray(base_front_rgb[i]).resize((224, 224), Image.Resampling.LANCZOS)
                    hand_img = Image.fromarray(hand_rgb[i]).resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Extract joint state (first 7 dimensions of qpos) 
                    joint_state = qpos[i][:7] if qpos[i].shape[0] >= 7 else np.pad(qpos[i], (0, 7 - qpos[i].shape[0]))
                    
                    # Create state vector (8-dimensional)
                    state = qpos[i][:8]
                    
                    episode.append({
                        'observation': {
                            'image': np.array(base_img),
                            'wrist_image': np.array(hand_img),
                            'joint_state': joint_state,
                            'state': state,
                        },
                        'action': actions[i],
                        'discount': 1.0,
                        'reward': rewards[i] if i < len(rewards) else 0.0,
                        'is_first': i == 0,
                        'is_last': i == (num_steps - 1),
                        'is_terminal': terminated[i] if i < len(terminated) else (i == num_steps - 1),
                        'language_instruction': language_instruction,
                        'language_embedding': language_embedding,
                    })

                # Create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path
                    }
                }

                return episode_path, sample

        # Find all HDF5 files in the directory
        episode_paths = glob.glob(os.path.join(path, '**/*.h5'), recursive=True)
        
        # For smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # For large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )