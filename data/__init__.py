from .brats import BraTS
from .dataset_example import BraTS_new

datasets = {
    'brats': BraTS,
    'new': BraTS_new
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)