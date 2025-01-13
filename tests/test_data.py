import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from torch.utils.data import Dataset

from mlops_final_exercise.data import corrupt_mnist


# @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
# I am testing the wrong function here.
def test_data():
    n_train = 30000
    n_test = 5000

    train_set, test_set = corrupt_mnist()
    assert len(train_set) == n_train, "Expected 30000 samples in the training set"
    assert len(test_set) == n_test, "Expected 5000 samples in the test set"
    for i in range(n_train):
        assert train_set[i][0].shape in [(1, 28, 28), (784,)], "Expected each sample to have shape [1, 28, 28]"
        assert train_set[i][1] in range(10), "Expected target to be in range [0, 9]"
    for i in range(n_test):
        assert test_set[i][0].shape in [(1, 28, 28), (784,)], "Expected each sample to have shape [1, 28, 28]"
        assert test_set[i][1] in range(10), "Expected target to be in range [0, 9]"
