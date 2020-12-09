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

"""DAVIS 2017 dataset for video object segmentation."""

import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """
The DAVIS 2017 video object segmentation dataset.
"""

_CITATION = """\
@article{DBLP:journals/corr/Pont-TusetPCASG17,
  author    = {Jordi Pont{-}Tuset and
               Federico Perazzi and
               Sergi Caelles and
               Pablo Arbelaez and
               Alexander Sorkine{-}Hornung and
               Luc Van Gool},
  title     = {The 2017 {DAVIS} Challenge on Video Object Segmentation},
  journal   = {CoRR},
  volume    = {abs/1704.00675},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.00675},
  archivePrefix = {arXiv},
  eprint    = {1704.00675},
  timestamp = {Mon, 13 Aug 2018 16:48:55 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/Pont-TusetPCASG17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


class Davis(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for davis dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download the DAVIS 2017 full resolution and 480p datasets manually from
  https://davischallenge.org/, unzip both of them, and place both at
  ~/tensorflow_datasets/downloads/manual/. This will produce some name
  collisions as both the full resolution and 480p downloads will extract into a
  folder called DAVIS and both include some of the same files (such as the
  train.txt in ImageAnnotations). Since these files are the same in both
  downloads, you can ignore these collisions.
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'video':
                tfds.features.Sequence({
                    'frames': tfds.features.Image(shape=(None, None, 3)),
                    'segmentations': tfds.features.Image(shape=(None, None, 1)),
                }),
            'metadata': {
                'num_frames': tf.int64,
                'video_segment': tf.string,
            },
        }),
        supervised_keys=None,
        homepage='https://davischallenge.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    manual_path = dl_manager.manual_dir
    train_files = os.path.join(manual_path, 'DAVIS', 'ImageSets', '2017',
                               'train.txt')
    val_files = os.path.join(manual_path, 'DAVIS', 'ImageSets', '2017',
                             'val.txt')
    return {
        'train_480p':
            self._generate_examples(train_files, '480p'),
        'val_480p':
            self._generate_examples(val_files, '480p'),
        'train_full_resolution':
            self._generate_examples(train_files, 'Full-Resolution'),
        'val_full_resolution':
            self._generate_examples(val_files, 'Full-Resolution'),
    }

  def _generate_examples(self, path, resolution):
    """Yields examples in the form of key, dataset dictionary."""

    with tf.io.gfile.GFile(path, 'r') as file:
      videos_to_include = file.readlines()
    videos_to_include = [name.strip('\n') for name in videos_to_include]
    root_path = path
    for _ in range(3):  # Move up three directories.
      root_path = os.path.dirname(root_path)
    for video in videos_to_include:
      images_path = os.path.join(root_path, 'JPEGImages', resolution, video)
      annotations_path = os.path.join(root_path, 'Annotations', resolution,
                                      video)
      seq_len = len(tf.io.gfile.listdir(images_path))
      images = []
      annotations = []
      for i in range(seq_len):
        image_path = os.path.join(images_path, '{0:05d}.jpg'.format(i))
        annotation_path = os.path.join(annotations_path,
                                       '{0:05d}.png'.format(i))
        image = tf.io.gfile.GFile(image_path, 'rb')
        annotation = tf.io.gfile.GFile(annotation_path, 'rb')
        images.append(image)
        annotations.append(annotation)

      video_dict = []
      for img, seg in zip(images, annotations):
        video_dict.append({'frames': img, 'segmentations': seg})
      metadata = {'num_frames': seq_len, 'video_segment': video}
      key = video + resolution
      yield key, {'video': video_dict, 'metadata': metadata}

