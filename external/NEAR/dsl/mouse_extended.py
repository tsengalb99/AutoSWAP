import torch
from .library_functions import AffineFeatureSelectionFunction

MOUSE_OFFSET = 108
MOUSE_FEATURE_SUBSETS = {
    # positional selects
    'positional1': torch.LongTensor(list(range(0, 14))),
    'centroid1': torch.LongTensor(list(range(14, 22))),
    'angle1': torch.LongTensor(list(range(22, 27)) + [51, 52]),
    'shape1': torch.LongTensor(list(range(27, 34)) + [49, 50]),
    'speed1': torch.LongTensor(list(range(34, 40)) + [53, 54] + list(range(90, 108))),
    'distance1': torch.LongTensor(list(range(40, 49)) + list(range(55, 90))),
    'positional2': torch.LongTensor(list(range(MOUSE_OFFSET + 0, MOUSE_OFFSET + 14))),
    'centroid2': torch.LongTensor(list(range(MOUSE_OFFSET + 14, MOUSE_OFFSET + 22))),
    'angle2': torch.LongTensor(list(range(MOUSE_OFFSET + 22, MOUSE_OFFSET + 27)) + \
                               [MOUSE_OFFSET + 51, MOUSE_OFFSET + 52]),
    'shape2': torch.LongTensor(list(range(MOUSE_OFFSET + 27, MOUSE_OFFSET + 34)) + \
                               [MOUSE_OFFSET + 49, MOUSE_OFFSET + 50]),
    'speed2': torch.LongTensor(list(range(MOUSE_OFFSET + 34, MOUSE_OFFSET + 40)) + \
                               [MOUSE_OFFSET + 53, MOUSE_OFFSET + 54] + \
                               list(range(MOUSE_OFFSET + 90, MOUSE_OFFSET + 108))),
    'distance2': torch.LongTensor(list(range(MOUSE_OFFSET + 40, MOUSE_OFFSET + 49)) + \
                                  list(range(MOUSE_OFFSET + 55, MOUSE_OFFSET + 90))),
    'mouse1': torch.LongTensor(list(range(0, 108))),
    'mouse2': torch.LongTensor(list(range(108, 216))),
}

MOUSE_FULL_FEATURE_DIM = 216


class Mouse1ExtendedPositionalSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["positional1"]
        super().__init__(input_size, output_size, num_units, name="positional1")


class Mouse1ExtendedCentroidSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["centroid1"]
        super().__init__(input_size, output_size, num_units, name="centroid1")


class Mouse1ExtendedAngleSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["angle1"]
        super().__init__(input_size, output_size, num_units, name="angle1")


class Mouse1ExtendedShapeSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["shape1"]
        super().__init__(input_size, output_size, num_units, name="shape1")


class Mouse1ExtendedSpeedSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["speed1"]
        super().__init__(input_size, output_size, num_units, name="speed1")


class Mouse1ExtendedDistanceSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["distance1"]
        super().__init__(input_size, output_size, num_units, name="distance1")


class Mouse1ExtendedFullSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["mouse1"]
        super().__init__(input_size, output_size, num_units, name="mouse1")


class Mouse2ExtendedPositionalSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["positional2"]
        super().__init__(input_size, output_size, num_units, name="positional2")


class Mouse2ExtendedCentroidSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["centroid2"]
        super().__init__(input_size, output_size, num_units, name="centroid2")


class Mouse2ExtendedAngleSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["angle2"]
        super().__init__(input_size, output_size, num_units, name="angle2")


class Mouse2ExtendedShapeSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["shape2"]
        super().__init__(input_size, output_size, num_units, name="shape2")


class Mouse2ExtendedSpeedSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["speed2"]
        super().__init__(input_size, output_size, num_units, name="speed2")


class Mouse2ExtendedDistanceSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["distance2"]
        super().__init__(input_size, output_size, num_units, name="distance2")


class Mouse2ExtendedFullSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MOUSE_FULL_FEATURE_DIM
        self.feature_tensor = MOUSE_FEATURE_SUBSETS["mouse2"]
        super().__init__(input_size, output_size, num_units, name="mouse2")
