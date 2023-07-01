from albumentations import (
    Compose,
    HorizontalFlip,
    ShiftScaleRotate,
    CoarseDropout
)
from albumentations.pytorch import ToTensor

transform_train = Compose([
    HorizontalFlip(),
    ShiftScaleRotate(),
    CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(mean of your dataset), mask_fill_value=None),
    ToTensor(),
])