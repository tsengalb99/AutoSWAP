import dsl

DSL_DICT = {
    ('list', 'list'): [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom'): [dsl.FoldFunction, dsl.SimpleITE],
    ('atom', 'atom'): [
        dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE, dsl.AppendFunction,
        dsl.OuterMultiplyFunction, dsl.FruitFlyWingSelection, dsl.FruitFlyRatioSelection,
        dsl.FruitFlyPositionalSelection, dsl.FruitFlyAngularSelection, dsl.FruitFlyLinearSelection
    ]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list'): {},
    ('list', 'atom'): {},
    ('atom', 'atom'): {
        dsl.AddFunction: 0.0,
        dsl.MultiplyFunction: 0.0,
        dsl.SimpleITE: 0.0,
        dsl.AppendFunction: 0.0,
        dsl.OuterMultiplyFunction: 0.0
    }
}
