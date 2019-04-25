def test():
    frame = cv2.imread("test/frame.png")
    background = cv2.imread("test/background.png")

    foreground = (background.astype(np.float32) - frame) / 255.0
    show(foreground, time_out=100)

    foreground = erode(foreground, 6)
    show(foreground, time_out=100)

    foreground = dilate(foreground, 6)
    show(foreground, time_out=100)

    foreground_thres = threshold(foreground, 0.4)
    foreground_thres = foreground_thres[:, :, 0]
    # np.save('fg.npy', foreground_thres)
    show(foreground_thres, time_out=100)

    centers, labels, _, _, area, labeled_frame = segment_connected_components(foreground_thres)
    print(centers)
    print(area)
    show(labeled_frame, time_out=1000)
    show(annotate(frame / 255, centers), time_out=2000)

    centers = segment_cluster(foreground_thres, 12)[0]
    print(centers)

    center_of_mass, _, _, _, _ = segment_center_of_mass(foreground_thres)
    print(center_of_mass)
    show(annotate(frame / 255, centers), time_out=2000)

    labeled_frame = get_chambers(background)
    bounding_box = get_bounding_box(labeled_frame)
    show(labeled_frame / np.max(labeled_frame), time_out=1000)
    show(labeled_frame == 5, time_out=1000)  # pick chamber #5

    show(crop(frame, [10, 550, 100, -1]) / 255, time_out=1000)

    labeled_frame = get_chambers(background)
    labeled_frame[labeled_frame == 10] = 0
    labeled_frame[labeled_frame == 5] = 20

    show(labeled_frame / np.max(labeled_frame), time_out=1000)
    new_labels = np.unique(labeled_frame)
    new_labels[-1] = 0  # delete last chamber (#20)
    # delete last chamber only
    print(np.unique(clean_labels(labeled_frame, new_labels)[0]))
    # remap to cont labels
    print(np.unique(clean_labels(labeled_frame, new_labels, force_cont=True)[0]))
    show(labeled_frame / np.max(labeled_frame), time_out=1000)

