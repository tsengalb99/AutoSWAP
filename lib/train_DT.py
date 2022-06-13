import numpy as np
from sklearn import tree
import zss


def create_zss_graph(dt, node):
    node_name = f'{dt.feature[node]}_{dt.threshold[node]}'
    cur_node = zss.Node(node_name)
    if dt.children_left[node] == -1 and dt.children_right[node] == -1:
        return cur_node
    else:
        assert dt.children_left[node] != -1
        cur_node.addkid(create_zss_graph(dt, dt.children_left[node]))
        assert dt.children_right[node] != -1
        cur_node.addkid(create_zss_graph(dt, dt.children_right[node]))
    return cur_node


def train_DT(
    train_data,
    valid_data,
    test_data,
    ap_score,
    sel_inds,
):
    # only works on nonsequential data
    features, Y = train_data
    valid_features, _ = valid_data
    test_features, test_Y = test_data

    features = np.squeeze(features)
    sel_features = features[sel_inds]
    sel_Y = Y[sel_inds]

    valid_features = np.squeeze(valid_features)
    test_features = np.squeeze(test_features)

    clf = tree.DecisionTreeClassifier(max_depth=5).fit(sel_features, sel_Y)

    zss_graph = create_zss_graph(clf.tree_, 0)

    preds = []
    for f in [features, valid_features, test_features]:
        preds.append(np.clip(clf.predict_proba(f), 1e-5, 1 - (1e-5)))

    best_ap = ap_score(test_Y, preds[-1])
    return preds, best_ap, zss_graph
