"""Custom AP metrics"""
from sklearn.metrics import average_precision_score as ap_score


def fly_ap_score(gt, pred):
    # 2nd class is true prediction
    return ap_score(gt, pred[:, 1])


def mouse_ap_score(gt, pred):
    total_ap = []
    for i in range(3):
        gt_copy = (gt == i).astype(int)
        total_ap.append(ap_score(gt_copy, pred[:, i]))
    print(total_ap)
    return sum(total_ap) / len(total_ap)


def bball_ap_score(gt, pred):
    total_ap = []
    for i in range(5):
        gt_copy = (gt == i).astype(int)
        total_ap.append(ap_score(gt_copy, pred[:, i]))
    print(total_ap)
    return sum(total_ap) / len(total_ap)


def bball_screen_ap_score(gt, pred):
    # 2nd class is true prediction
    return ap_score(gt, pred[:, 1])
