import torch
from .library_functions import AffineFeatureSelectionFunction

BBALL_FEATURE_SUBSETS = {
    # positional selects
    'ball_accel': torch.LongTensor(list(range(0, 2))),
    'ball_speed': torch.LongTensor(list(range(2, 3))),
    #'ball_coord': torch.LongTensor(list(range(3, 5))),
    'ball_vel': torch.LongTensor(list(range(5, 7))) - 2,
    'p1_stats': torch.LongTensor([7, 8, 17, 22, 23, 32, 33]) - 2,
    'p2_stats': torch.LongTensor([9, 10, 18, 24, 25, 34, 35]) - 2,
    'p3_stats': torch.LongTensor([11, 12, 19, 26, 27, 36, 37]) - 2,
    'p4_stats': torch.LongTensor([13, 14, 20, 28, 29, 38, 39]) - 2,
    'p5_stats': torch.LongTensor([15, 16, 21, 30, 31, 40, 41]) - 2,
    'player_accel': torch.LongTensor(list(range(7, 17))) - 2,
    'player_speed': torch.LongTensor(list(range(17, 22))) - 2,
    'player_coord': torch.LongTensor(list(range(22, 32))) - 2,
    'player_vel': torch.LongTensor(list(range(32, 42))) - 2,
}

BBALL_FULL_FEATURE_DIM = 42


class BBallBallAccelSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["ball_accel"]
        super().__init__(input_size, output_size, num_units, name="ball_accel")


class BBallBallSpeedSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["ball_speed"]
        super().__init__(input_size, output_size, num_units, name="ball_speed")


class BBallBallCoordSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["ball_coord"]
        super().__init__(input_size, output_size, num_units, name="ball_coord")


class BBallBallVelSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["ball_vel"]
        super().__init__(input_size, output_size, num_units, name="ball_vel")


class BBallP1StatsSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["p1_stats"]
        super().__init__(input_size, output_size, num_units, name="p1_stats")


class BBallP2StatsSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["p2_stats"]
        super().__init__(input_size, output_size, num_units, name="p2_stats")


class BBallP3StatsSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["p3_stats"]
        super().__init__(input_size, output_size, num_units, name="p3_stats")


class BBallP4StatsSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["p4_stats"]
        super().__init__(input_size, output_size, num_units, name="p4_stats")


class BBallP5StatsSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["p5_stats"]
        super().__init__(input_size, output_size, num_units, name="p5_stats")


class BBallPlayerAccelSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["player_accel"]
        super().__init__(input_size, output_size, num_units, name="player_accel")


class BBallPlayerSpeedSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["player_speed"]
        super().__init__(input_size, output_size, num_units, name="player_speed")


class BBallPlayerCoordSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["player_coord"]
        super().__init__(input_size, output_size, num_units, name="player_coord")


class BBallPlayerVelSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = BBALL_FEATURE_SUBSETS["player_vel"]
        super().__init__(input_size, output_size, num_units, name="player_vel")
