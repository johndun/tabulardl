import numpy as np
import torch
from pytest import raises

from tabulardl import (
    DataType, Feature, NumericFeature, FeatureTransformBeforeFitError, CategoricalFeature,
    CategoricalArrayFeature
)
from tabulardl.data.features.categorical import ValueNotFoundError


def test_feature():
    with raises(NotImplementedError):
        feature = Feature(DataType.NUMERIC)
        feature._fit_data_transformer(None)

    with raises(NotImplementedError):
        feature = Feature(DataType.NUMERIC)
        feature._transform_raw_data(None)

    with raises(FeatureTransformBeforeFitError):
        feature = Feature(DataType.NUMERIC)
        _ = feature.transform_raw_data([1.,])


def test_numeric_array():
    data = [[0., 0.], [0., 1.]]
    feature = NumericFeature()
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
    feature = NumericFeature(clip_percentiles=[None, 0.5])
    feature.fit_data_transformer(data)
    transformed = feature.transform_raw_data(data)
    assert (transformed.numpy() == [[0., 0.], [0., 0.5], [0., 0.5]]).all()


def test_categorical_feature():
    # low_count_threshold, different unknown and missing values
    feature = CategoricalFeature(low_count_threshold=2)
    data = [1, 1, 2]
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, '<UNKNOWN>': 1, 1: 2}
    assert feature.dictionary_size == 3
    # max_vocab_size
    feature = CategoricalFeature(max_vocab_size=3)
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, '<UNKNOWN>': 1, 1: 2}
    # same unknown and missing values
    feature = CategoricalFeature(unknown_value='<MISSING>', low_count_threshold=2)
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
    feature = CategoricalFeature(unknown_value='<MISSING>', value_map={'<MISSING>': 0, 1: 1})
    transformed = feature.transform_raw_data('a').numpy()
    assert (transformed == [0]).all()
    # missing key in pre existing dictionary exception
    with raises(ValueNotFoundError):
        _ = CategoricalFeature(value_map={1: 1})


def test_numeric_feature():
    feature = NumericFeature(missing_value=0.)
    data = [1., None]
    # fit method
    feature.fit_data_transformer(data)
    assert feature.mean == 0.5
    assert feature.std == 0.5
    assert (feature.clip_values == [0., 1.]).all()
    # torch.FloatTensor
    assert isinstance(feature.transform_raw_data(data), torch.FloatTensor)
    # transform an array
    transformed = feature.transform_raw_data(data).numpy()
    assert (transformed == [1., 0.]).all()
    feature.center = True
    transformed = feature.transform_raw_data(data).numpy()
    assert (transformed == [0.5, -0.5]).all()
    feature.scale = True
    transformed = feature.transform_raw_data(data).numpy()
    assert (transformed == [1., -1.]).all()
    # transform a singleton
    data = 0.
    transformed = feature.transform_raw_data(data).numpy()
    assert (transformed == [-1]).all()
    # clipping
    data = [0., 1., 0., 1.]
    feature = NumericFeature(clip_percentiles=[None, 0.75])
    feature.fit_data_transformer(data)
    assert feature.clip_values[1] == 1.
    transformed = feature.transform_raw_data(2.).numpy()
    assert (transformed == [1.]).all()


def test_categorical_array_feature():
    feature = CategoricalArrayFeature(max_len=3)
    data = [[1,], [1, None]]
    # fit method
    feature.fit_data_transformer(data)
    assert feature.value_map == {'<MISSING>': 0, '<UNKNOWN>': 1, '<PAD>': 2, 1: 3}
    transformed = feature.transform_raw_data(data).numpy()
    assert transformed.shape == (2, 3)
    assert (transformed[0] == [3, 2, 2]).all()
    assert (transformed[1] == [3, 1, 2]).all()
