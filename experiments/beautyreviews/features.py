from tabularencoder import CategoricalFeature, NumericFeature, CategoricalArrayFeature


INTERACTION_FEATURE_SCHEMA = {
    'asin': CategoricalFeature(
        id='asin',
        key='asin',
        max_vocab_size=100000,
    ),
    'reviewerID': CategoricalFeature(
        id='reviewerID',
        key='reviewerID',
        max_vocab_size=100000,
    ),
    'helpful': NumericFeature(
        id='helpful',
        key='helpful',
        missing_value=0,
        unit_scale=True,
    ),
    'review_month': CategoricalFeature(
        id='review_month',
        key='review_month',
    ),
    'review_day': CategoricalFeature(
        id='review_day',
        key='review_day',
    ),
    'review_year': CategoricalFeature(
        id='review_year',
        key='review_year',
    ),
    'overall': NumericFeature(
        id='overall',
        key='overall',
        center=True,
        scale=True,
    ),
    'reviewText': NumericFeature(
        id='reviewText',
        key='reviewText',
    ),
    'summary': NumericFeature(
        id='summary',
        key='summary',
    ),
}

INTERACTION_FEATURE_SCHEMA_MINIMAL = {
    key: val for key, val in INTERACTION_FEATURE_SCHEMA.items()
    if key in ('reviewerID', )
}

INTERACTION_FEATURE_SCHEMA_NO_TEXT = {
    key: val for key, val in INTERACTION_FEATURE_SCHEMA.items()
    if key in ('reviewerID', 'review_month', 'review_year', 'review_day', 'overall', 'helpful')
}

INTERACTION_FEATURE_SCHEMA_TEXT = {
    key: val for key, val in INTERACTION_FEATURE_SCHEMA.items()
    if key in (
        'reviewerID', 'review_month', 'review_year', 'review_day', 'overall', 'helpful',
        'reviewText', 'summary'
    )
}

ITEM_FEATURE_SCHEMA = {
    'asin': CategoricalFeature(
        id='asin',
        key='asin',
        max_vocab_size=100000,
    ),
    'price': NumericFeature(
        id='price',
        key='price',
        center=True,
        scale=True,
        clip_percentiles=[0, 0.99]
    ),
    'salesRank__Beauty': NumericFeature(
        id='salesRank__Beauty',
        key=['salesRank', 'Beauty'],
        unit_scale=True,
    ),
    'categories': CategoricalArrayFeature(
        id='categories',
        key='categories',
        max_len=6,
        low_count_threshold=30,
    ),
    'related__bought_together': CategoricalArrayFeature(
        id='related__bought_together',
        key=['related', 'bought_together'],
        max_len=4,
        low_count_threshold=30,
    ),
    'related__also_bought': CategoricalArrayFeature(
        id='related__also_bought',
        key=['related', 'also_bought'],
        max_len=100,
        low_count_threshold=30,
    ),
    'related__also_viewed': CategoricalArrayFeature(
        id='related__also_viewed',
        key=['related', 'also_viewed'],
        max_len=60,
        low_count_threshold=30,
    ),
    'brand': CategoricalFeature(
        id='brand',
        key='brand',
        low_count_threshold=30,
    ),
    'title': NumericFeature(
        id='title',
        key='title',
    ),
    'description': NumericFeature(
        id='description',
        key='description',
    ),
    'imUrl': CategoricalFeature(
        id='imUrl',
        key='imUrl',
    ),
}

ITEM_FEATURE_SCHEMA_MINIMAL = {
    key: val for key, val in ITEM_FEATURE_SCHEMA.items()
    if key in ('asin',)
}

ITEM_FEATURE_SCHEMA_NO_TEXT = {
    key: val for key, val in ITEM_FEATURE_SCHEMA.items()
    if key in (
        'asin', 'price', 'brand', 'categories',
        'salesRank__Beauty',
        'related__bought_together', 'related__also_bought', 'related__also_viewed',
    )
}

ITEM_FEATURE_SCHEMA_TEXT = {
    key: val for key, val in ITEM_FEATURE_SCHEMA.items()
    if key in (
        'asin', 'price', 'brand', 'categories',
        'salesRank__Beauty',
        'related__bought_together', 'related__also_bought', 'related__also_viewed',
        'title', 'description'
    )
}
