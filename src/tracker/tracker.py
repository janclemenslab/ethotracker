import numpy as np
from scipy import spatial
from .munkres import Munkres
import cv2
cv2.setNumThreads(1)

m = Munkres()


def match(old_centers, new_centers):
    D = spatial.distance.cdist(old_centers, new_centers)
    pathLabels = m.compute(D)
    new_labels = np.zeros((len(pathLabels),), dtype=np.uintp)
    new_centers_ordered = np.zeros((len(pathLabels), 2))
    for ii in range(new_centers.shape[0]):
        new_labels[ii] = pathLabels[ii][1]
        new_centers_ordered[ii] = new_centers[new_labels[ii]]
    return new_labels, new_centers_ordered


def fit_line(points, line_len=10):
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # transform open cv line format ([xlen, ylen, x0, y0]?) to [x0,x1] pairs
    x1 = (line[3] + line[1] * line_len, line[2] + line[0] * line_len)
    x2 = (line[3] - line[1] * line_len, line[2] - line[0] * line_len)
    return [[(line[3] + line[1] * line_len)[0], (line[2] + line[0] * line_len)[0]], [(line[3] - line[1] * line_len)[0], (line[2] - line[0] * line_len)[0]]], line


def fix_flips(old_lines, new_lines):
    D = np.sum(np.abs(old_lines - new_lines), axis=1)  # distance between old head and new head/tail
    is_flipped = np.argmin(D) != 0
    if is_flipped:  # if new head is not closer to old head than new tail
        new_lines = new_lines[[1, 0], :]  # -> swap head/tail
    return new_lines, is_flipped, D


def test():
    a = np.array([[1, 1], [2, 2], [3, 3]])
    b = np.array([[3, 3], [2, 2], [1, 1]])
    new_labels, _ = match(a, a)
    print(new_labels)
    new_labels, new_centers_ordered = match(a, b)
    print(new_labels)
    print(new_centers_ordered)

if __name__ == "__main__":
    test()
