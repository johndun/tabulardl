from itertools import chain

import numpy as np
import torch


def device_move_nested(batch, device):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)
        else:
            batch[key] = device_move_nested(val, device)
    return batch


def get_nested_key(data, key):
    if isinstance(key, str):
        return data[key]
    dat = data.get(key[0], {})
    for key in key[1:-1]:
        dat = dat.get(key, {})
    return dat.get(key[-1], None)


def transpose_json_data(data, base_key=None):
    metadata = {}
    base_key = base_key or []
    # Collect complete set of top-level keys
    keys = set(chain(*[(x or {}).keys() for x in data]))
    # Top-level transpose
    flipped = {key: [(x or {}).get(key, None) for x in data] for key in keys}
    for key in keys:
        data_col = flipped[key]
        data_ex = data_col[0]
        # Look for nested keys; recurse
        if isinstance(data_ex, dict):
            flipped[key], nested_metadata = transpose_json_data(data_col, base_key + [key])
            for key, val in nested_metadata.items():
                flattened_key = '__'.join(
                    base_key + val['key'] if isinstance(val['key'], list) else [val['key']]
                )
                metadata[flattened_key] = val
            continue
        metadata[key] = infer_datatype(
            data_col, {'key': key if not base_key else base_key + [key]}
        )
    return flipped, metadata


def infer_datatype(data, metadata):
    # Get the 1st non-missing example
    n_samples = len(data)
    xnn = [x for x in data if x is not None]
    example = xnn[0] if xnn else None
    if example is None:
        metadata['type'] = 'Empty'
        return metadata
    if isinstance(example, list):
        metadata = infer_datatype(list(chain(*xnn)), metadata)
        metadata['type'] += 'Array'
        lens = [len(x) if x is not None else 0 for x in data]
        metadata['is_equal_size'] = all([x == lens[0] for x in lens])
        metadata['prop_emptylist'] = sum([x is None or x == 0 for x in lens]) / n_samples
        metadata['max_len'] = max(lens)
        metadata['mean_len'] = sum(lens) / len(lens)
    else:
        metadata['examples'] = xnn[:5]
    if isinstance(example, (int, float)):
        metadata['type'] = 'Numeric'
        metadata['is_all_ints'] = all([isinstance(x, int) for x in data if x is not None])
        metadata['prop_nonnull'] = len(xnn) / n_samples
        metadata['distinct_vals'] = len(set(data))
        metadata['mean'] = np.mean(xnn)
        metadata['std'] = np.std(xnn)
        metadata['min'] = np.min(xnn)
        metadata['max'] = np.max(xnn)
    if isinstance(example, str):
        metadata['type'] = 'Categorical'
        metadata['prop_nonnull'] = len([x for x in data if x is not None]) / n_samples
        metadata['distinct_vals'] = len(set(data))
    return metadata


def infer_features(feature_metadata):
    for key, meta in feature_metadata.items():
        if meta['prop_nonnull'] < 0.1:
            continue
        if meta.get('prop_emptylist', 0.) > 0.9:
            continue
        print(f"    {meta['type'] if meta['type'] != 'NumericArray' else 'Numeric'}Feature(")
        print(8 * ' ' + f"id='{key}',")
        print(8 * ' ' + 'key=' + (
            f"{meta['key']}" if isinstance(meta['key'], list) else "'" + meta['key'] + "'"
        ) + ',')
        if 'max_len' in meta and meta['type'] != 'NumericArray':
            print(8 * ' ' + f'max_len={meta["max_len"]},')
        for field in meta.keys():
            if field in field in ('type', 'key', 'examples'):
                continue
            if field == 'max_len' and meta['type'] != 'NumericArray':
                continue
            print(8 * ' ' + f'# {field}={meta[field]},')
        print(8 * ' ' + f'# examples={meta["examples"]}')
        print('    ),')
