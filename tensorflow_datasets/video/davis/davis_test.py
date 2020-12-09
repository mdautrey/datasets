# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Davis dataset test."""

import numpy as np
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.video.davis import davis


class DavisTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for davis dataset."""
  # TODO(davis):
  DATASET_CLASS = davis.Davis
  SPLITS = {
      'train_480p': 1,  # Number of fake train examples.
      'val_480p': 1,
      'train_full_resolution': 1,
      'val_full_resolution': 1,
  }

  def test_dataset_shapes(self):
    builder = self._make_builder()
    self._download_and_prepare_as_dataset(builder)
    splits = builder.as_dataset()

    low_res_example = list(splits['train_480p'])[0]
    high_res_example = list(splits['train_full_resolution'])[0]
    low_res_num_frames = low_res_example['metadata']['num_frames'].numpy()
    high_res_num_frames = high_res_example['metadata']['num_frames'].numpy()

    # Check that the shapes of the dataset examples is correct.
    self.assertEqual(low_res_num_frames, 4)
    self.assertEqual(high_res_num_frames, 4)
    self.assertEqual(low_res_example['video']['frames'].numpy().shape,
                     (4, 480, 854, 3))
    self.assertEqual(low_res_example['video']['segmentations'].numpy().shape,
                     (4, 480, 854, 1))
    self.assertEqual(high_res_example['video']['frames'].numpy().shape,
                     (4, 1080, 1920, 3))
    self.assertEqual(high_res_example['video']['segmentations'].numpy().shape,
                     (4, 1080, 1920, 1))

  def test_annotations_classes(self):
    builder = self._make_builder()
    self._download_and_prepare_as_dataset(builder)
    splits = builder.as_dataset()

    low_res_train_example = list(splits['train_480p'])[0]
    high_res_train_example = list(splits['train_full_resolution'])[0]
    low_res_val_example = list(splits['val_480p'])[0]
    high_res_val_example = list(splits['val_full_resolution'])[0]

    # Check that the dataset examples contain the correct number of classes.
    self.assertLen(
        np.unique(low_res_train_example['video']['segmentations'].numpy()), 2)
    self.assertLen(
        np.unique(high_res_train_example['video']['segmentations'].numpy()), 2)
    self.assertLen(
        np.unique(low_res_val_example['video']['segmentations'].numpy()), 3)
    self.assertLen(
        np.unique(high_res_val_example['video']['segmentations'].numpy()), 3)


if __name__ == '__main__':
  tfds.testing.test_main()
