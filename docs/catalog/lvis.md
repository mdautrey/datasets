<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="lvis" />
  <meta itemprop="description" content="LVIS: A dataset for large vocabulary instance segmentation.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;lvis&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/lvis" />
  <meta itemprop="sameAs" content="https://www.lvisdataset.org/" />
  <meta itemprop="citation" content="@inproceedings{gupta2019lvis,&#10;  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},&#10;  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},&#10;  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},&#10;  year={2019}&#10;}" />
</div>

# `lvis`

Note: This dataset was added recently and is only available in our
`tfds-nightly` package
<span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>.

*   **Description**:

LVIS: A dataset for large vocabulary instance segmentation.

*   **Homepage**: [https://www.lvisdataset.org/](https://www.lvisdataset.org/)

*   **Source code**:
    [`tfds.object_detection.lvis.Lvis`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/object_detection/lvis/lvis.py)

*   **Versions**:

    *   **`1.0.0`** (default): Initial release. Test split has dummy
        annotations.

*   **Download size**: `25.35 GiB`

*   **Dataset size**: `22.28 GiB`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    No

*   **Splits**:

Split          | Examples
:------------- | -------:
`'test'`       | 19,822
`'train'`      | 100,170
`'validation'` | 19,809

*   **Features**:

```python
FeaturesDict({
    'image': Image(shape=(None, None, 3), dtype=tf.uint8),
    'image/id': tf.int64,
    'objects': Sequence({
        'area': tf.int64,
        'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
        'id': tf.int64,
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=1203),
        'segmentation': Image(shape=(None, None, 1), dtype=tf.uint8),
    }),
})
```

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `None`

*   **Citation**:

```
@inproceedings{gupta2019lvis,
  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):
    Missing.
