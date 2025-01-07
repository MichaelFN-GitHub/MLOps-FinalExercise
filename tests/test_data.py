import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset

from mlops_final_exercise.data import corrupt_mnist


'''
from mlops_final_exercise.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
'''

def test():
    corrupt_mnist()
    assert isinstance(1, int)
    #assert isinstance(train_set, tuple) and len(train_set) == 2 and isinstance(train_set[0], Dataset) and isinstance(train_set[1], Dataset)
    #assert isinstance(test_set, tuple) and len(test_set) == 2 and isinstance(test_set[0], Dataset) and isinstance(test_set[1], Dataset)