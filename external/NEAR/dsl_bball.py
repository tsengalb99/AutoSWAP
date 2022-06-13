import dsl

DSL_DICT = {
    ('list', 'list'): [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom'): [dsl.FoldFunction, dsl.SimpleITE],
    ('atom', 'atom'): [
        dsl.AddFunction,
        dsl.MultiplyFunction,
        dsl.SimpleITE,
        dsl.AppendFunction,
        dsl.OuterMultiplyFunction,
        dsl.BBallBallAccelSelection,
        dsl.BBallBallSpeedSelection,
        dsl.BBallBallVelSelection,
        dsl.BBallPlayerAccelSelection,
        dsl.BBallPlayerSpeedSelection,
        dsl.BBallPlayerCoordSelection,
        dsl.BBallPlayerVelSelection,
    ]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list'): {
        dsl.MapFunction: 0.0,
        dsl.MapPrefixesFunction: 0.0,
        dsl.SimpleITE: 0.0,
    },
    ('list', 'atom'): {
        dsl.FoldFunction: 0.0,
        dsl.SimpleITE: 0.0,
    },
    ('atom', 'atom'): {
        dsl.AppendFunction: 0.0,
        dsl.OuterMultiplyFunction: 0.0,
        dsl.AddFunction: 0.0,
        dsl.MultiplyFunction: 0.0,
        dsl.SimpleITE: 0.0,
    },
}
