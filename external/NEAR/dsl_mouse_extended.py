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
        dsl.Mouse1ExtendedPositionalSelection,
        dsl.Mouse1ExtendedCentroidSelection,
        dsl.Mouse1ExtendedAngleSelection,
        dsl.Mouse1ExtendedShapeSelection,
        dsl.Mouse1ExtendedSpeedSelection,
        dsl.Mouse1ExtendedDistanceSelection,
        dsl.Mouse1ExtendedFullSelection,
        dsl.Mouse2ExtendedPositionalSelection,
        dsl.Mouse2ExtendedCentroidSelection,
        dsl.Mouse2ExtendedAngleSelection,
        dsl.Mouse2ExtendedShapeSelection,
        dsl.Mouse2ExtendedSpeedSelection,
        dsl.Mouse2ExtendedDistanceSelection,
        dsl.Mouse2ExtendedFullSelection,
    ]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list'): {},
    ('list', 'atom'): {},
    ('atom', 'atom'): {
        dsl.AppendFunction: 0.0,
        dsl.OuterMultiplyFunction: 0.0,
        dsl.AddFunction: 0.0,
        dsl.MultiplyFunction: 0.0,
        dsl.SimpleITE: 0.0
    },
}
