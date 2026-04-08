# Data Splits

Split assay-labeled data into train, validation, and test sets for predictive model training and evaluation.

!!! note "Coming soon"
    This module is under development. The strategies below describe the intended functionality.

## Why splitting matters

Random splits can be misleading for protein fitness prediction because nearby mutations are correlated. A model that memorizes local neighborhoods will look good on a random split but fail on distant variants — which are exactly what you want to design.

## Split strategies

### By mutational distance

Group variants by their Hamming distance from wildtype. Train on singles/doubles, evaluate on higher-order mutants. This tests whether the predictor can extrapolate to more distant parts of sequence space.

### By activity range

Split so that the test set covers the high-activity tail — the regime you care about for design. If the predictor's ranking in this tail is poor, guidance will be unreliable.

### By position

Hold out all variants at certain positions. Tests whether the predictor can generalize to unseen positions — important when your library design targets positions not in the training data.

## Sensitivity analysis

Train predictors on each split and compare their rankings on the held-out set. If rankings are unstable across splits, the predictor is unreliable for guidance — consider collecting more data or using a simpler model.

<!-- TODO: implement split functions: mutational_distance_split, activity_range_split, position_split -->
<!-- TODO: implement sensitivity_analysis(predictors, splits, sequences) -->
