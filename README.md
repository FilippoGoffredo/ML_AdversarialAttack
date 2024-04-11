# ML_AdversarialAttack
 In order to correctly preprocess the dataset, is important also check the distribution of label class. We noticed that:
 - The proportion of individuals which are good creditors is 0.7, while the proportion of individuals which are bad creditors is 0.3.
This finding is crucial because we can affirm that our dataset is balanced, in order to avoid critical cases of bias.

Then we proceed with one-hot-encoding: one hot-encoding works with k levels (k as number of categorical features). drop_first parameter is used to drop the first level of each categorical feature (if it is set to true).
We set true because we need only k−1 dummy variables to represent all the information about that variable.


As an important part of the preprocessing, we perfomed on data already splitted the normalization: we normalized in this step in order to avoid data leakage. So, we applied MinMax normalization mapping between 0 and 1 the training set.