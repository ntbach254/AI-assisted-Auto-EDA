# california_housing — Pipeline Summary

## Overview

- **Total hypotheses tested**: 15
- **Final accepted**: 11
- **Acceptance rate**: 73.3%

## Accepted Hypotheses

### H1
**Claim**: In the California housing data, median house value is jointly positively associated with median income and average number of rooms per household.

- Test: `linear_regression` | Effect: `0.4355` | p-value: `0.0000e+00` | Direction: association_without_direction

### H3
**Claim**: Population size is jointly negatively related to house age and positively related to average occupancy.

- Test: `linear_regression` | Effect: `-26.9549` | p-value: `6.7721e-135` | Direction: association_without_direction

### H5
**Claim**: Average number of rooms per household rises with median income, even after accounting for the negative association between median income and average bedrooms.

- Test: `linear_regression` | Effect: `0.5001` | p-value: `0.0000e+00` | Direction: association_without_direction

### H6
**Claim**: Median house value is positively associated with median income and negatively associated with average occupancy when both factors are considered together.

- Test: `linear_regression` | Effect: `0.4145` | p-value: `0.0000e+00` | Direction: association_without_direction

### H9
**Claim**: Latitude and longitude are strongly negatively linearly correlated.

- Test: `pearsonr` | Effect: `-0.9276` | p-value: `0.0000e+00` | Direction: negative

### H10
**Claim**: Median income is positively monotonic associated with median house value.

- Test: `spearmanr` | Effect: `0.6774` | p-value: `0.0000e+00` | Direction: positive

### H11
**Claim**: Median income is positively monotonic associated with average number of rooms per household.

- Test: `spearmanr` | Effect: `0.6516` | p-value: `0.0000e+00` | Direction: positive

### H12
**Claim**: House age is negatively monotonic associated with population size.

- Test: `spearmanr` | Effect: `-0.2826` | p-value: `4.6974e-114` | Direction: negative

### H13
**Claim**: Average occupancy is negatively monotonic associated with median house value.

- Test: `spearmanr` | Effect: `-0.2559` | p-value: `3.7432e-93` | Direction: negative

### H14
**Claim**: Average number of rooms per household is positively monotonic associated with median house value.

- Test: `spearmanr` | Effect: `0.2798` | p-value: `1.0231e-111` | Direction: positive

### H15
**Claim**: Median income is negatively monotonic associated with average number of bedrooms per household.

- Test: `spearmanr` | Effect: `-0.2495` | p-value: `1.5413e-88` | Direction: negative

## By Relationship Type

- **association**: 7 hypotheses
- **multi_edge_pattern**: 4 hypotheses