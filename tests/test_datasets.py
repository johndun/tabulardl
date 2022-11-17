import torch

from tabulardl import TabularDataset


def test_tabular_dataset():
    dataset = TabularDataset({'x': torch.FloatTensor(list(range(10)))})
    assert len(dataset) == 10  # Length function
    assert dataset[3]['x'].item() == 3  # Single item indexing
    assert dataset[:5]['x'].shape == (5,)  # Range indexing

    data, more_data = [
        TabularDataset({
            'actions': torch.zeros(10),
            'states': TabularDataset({'x': torch.zeros(10), 'y': torch.zeros(10)}),
            'next_states': TabularDataset({'x': torch.zeros(10), 'y': torch.zeros(10)})
        })
        for _ in range(2)
    ]
    assert set(data[0].keys()) == {'actions', 'states', 'next_states'}  # Sample keys
    assert set(data[0]['states'].keys()) == {'x', 'y'}  # Nested keys
    assert data[:2]['states']['x'].shape == (2,)  # Range indexing

    combined_data = data.join(more_data)
    assert len(combined_data) == 20  # Joins
    assert combined_data.data['actions'].shape == (20,)

    for idx, batch in enumerate(combined_data.loader(batch_size=2)):
        continue

    assert batch['states']['x'].shape == (2,)  # Loader batch shape
    assert idx == 9  # Number of batches
