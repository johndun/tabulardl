import numpy as np
import torch
from pytest import raises

from tabularencoder.features import (
    DataType, Feature, NumericFeature, FeatureTransformBeforeFitError, CategoricalFeature,
    CategoricalArrayFeature, ValueNotFoundError
)


def test_feature():
    with raises(NotImplementedError):
        feature = Feature('id', 'key', DataType.NUMERIC)
        feature._fit_data_transformer(None)

    with raises(NotImplementedError):
        feature = Feature('id', 'key', DataType.NUMERIC)
        feature._transform_raw_data(None)

    with raises(FeatureTransformBeforeFitError):
        feature = Feature('id', 'key', DataType.NUMERIC)
        _ = feature.transform_raw_data([1.,])


def test_numeric_array():
    data = [[0., 0.], [0., 1.]]
    feature = NumericFeature('id', 'key')
    feature.fit_data_transformer(data)
    assert (feature.mean == [0., 0.5]).all()
    assert (feature.std == [0., 0.5]).all()
    assert (feature.clip_values == [[0., 0.], [0., 1.]]).all()
    transformed = feature.transform_raw_data(data)
    assert (transformed.numpy() == data).all()
    feature.center = True
    transformed = feature.transform_raw_data(data)
    assert (transformed.numpy() == [[0., -0.5], [0., 0.5]]).all()
    feature.scale = True
    transformed = feature.transform_raw_data(data)
    assert (transformed.numpy() == [[0., -1.], [0., 1]]).all()
    data = [[0., 0.], [0., 0.5], [0., 1.]]
    feature = NumericFeature('id', 'key', clip_percentiles=[None, 0.5])
    feature.fit_data_transformer(data)
    transformed = feature.transform_raw_data(data)
    assert (transformed.numpy() == [[0., 0.], [0., 0.5], [0., 0.5]]).all()


def test_categorical_feature():
    # low_count_threshold, different unknown and missing values
    feature = CategoricalFeature('id', 'key', low_count_threshold=2)
    data = [1, 1, 2]
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, '<UNKNOWN>': 1, 1: 2}
    assert feature.dictionary_size == 3
    # max_vocab_size
    feature = CategoricalFeature('id', 'key', max_vocab_size=3)
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, '<UNKNOWN>': 1, 1: 2}
    # same unknown and missing values
    feature = CategoricalFeature('id', 'key', unknown_value='<MISSING>', low_count_threshold=2)
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, 1: 1}
    # transform single
    transformed = feature.transform_raw_data('a').numpy()
    assert (transformed == [0]).all()
    # data type
    assert isinstance(feature.transform_raw_data('a'), torch.LongTensor)
    # transform list
    transformed = feature.transform_raw_data([1, 'a', None]).numpy()
    assert (transformed == [1, 0, 0]).all()
    # pre existing dictionary
    feature = CategoricalFeature(
        'id', 'key', unknown_value='<MISSING>', value_map={'<MISSING>': 0, 1: 1}
    )
    transformed = feature.transform_raw_data('a').numpy()
    assert (transformed == [0]).all()
    # missing key in pre existing dictionary exception
    with raises(ValueNotFoundError):
        _ = CategoricalFeature('id', 'key', value_map={1: 1})


def test_numeric_feature():
    feature = NumericFeature('id', 'key', missing_value=0.)
    data = [1., None]
    # fit method
    feature.fit_data_transformer(data)
    assert feature.mean == 0.5
    assert feature.std == 0.5
    assert (feature.clip_values == [0., 1.]).all()
    # torch.FloatTensor
    assert isinstance(feature.transform_raw_data(data), torch.FloatTensor)
    # transform an array
    transformed = feature.transform_raw_data(data).numpy()[:, 0]
    assert (transformed == [1., 0.]).all()
    feature.center = True
    transformed = feature.transform_raw_data(data).numpy()[:, 0]
    assert (transformed == [0.5, -0.5]).all()
    feature.scale = True
    transformed = feature.transform_raw_data(data).numpy()[:, 0]
    assert (transformed == [1., -1.]).all()
    # transform a singleton
    data = 0.
    transformed = feature.transform_raw_data(data).numpy()[:, 0]
    assert (transformed == [-1]).all()
    # clipping
    data = [0., 1., 0., 1.]
    feature = NumericFeature('id', 'key', clip_percentiles=[None, 0.75])
    feature.fit_data_transformer(data)
    assert feature.clip_values[1] == 1.
    transformed = feature.transform_raw_data(2.).numpy()[:, 0]
    assert (transformed == [1.]).all()


def test_categorical_array_feature():
    feature = CategoricalArrayFeature('id', 'key', max_len=3)
    data = [[1, ], [1, None]]
    # fit method
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<PAD>': 0, '<MISSING>': 1, '<UNKNOWN>': 2, 1: 3}
    transformed = feature.transform_raw_data(data).numpy()
    assert transformed.shape == (2, 3)
    assert (transformed[0] == [3, 0, 0]).all()
    assert (transformed[1] == [3, 1, 0]).all()
