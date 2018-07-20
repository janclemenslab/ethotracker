import numpy as np
import cv2
import scipy.ndimage as sci
import skimage.segmentation


def circular_kernel(kernel_size=3):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def dilate(frame, kernel_size=3):
    return cv2.dilate(frame.astype(np.uint8), circular_kernel(kernel_size))


def erode(frame, kernel_size=3):
    return cv2.erode(frame.astype(np.uint8), circular_kernel(kernel_size))


def close(frame, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def threshold(frame, threshold):
    return frame > threshold


def crop(frame, coords):
    """crop x,y"""
    if len(frame.shape) == 3:
        return frame[coords[1]:coords[3], coords[0]:coords[2], :]
    elif len(frame.shape) == 2:
        return frame[coords[1]:coords[3], coords[0]:coords[2]]


def clean_labels(labeled_frame, new_labels=None, force_cont=False):
    """maps unique values in `labeled_frame` to new values from `new_labels`
    `force_cont` ensures that new labels will be continuous, starting at 0
    source: http://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
    """
    old_labels = np.unique(labeled_frame)
    # new_labels gives the new values you wish old_labels to be mapped to.
    if new_labels is None:
        new_labels = np.array(range(old_labels.shape[0]))

    # ensure labels are sorted - do this always since it should be cheap
    sort_idx = np.argsort(old_labels)
    old_labels = old_labels[sort_idx]
    new_labels = new_labels[sort_idx]
    # remap
    index = np.digitize(labeled_frame, old_labels, right=True)
    labeled_frame = new_labels[index]
    if force_cont:
        labeled_frame, new_labels, _ = clean_labels(labeled_frame)
    return labeled_frame, new_labels, old_labels


def getpoints(frame):
    """Get x,y indices of foreground pixels."""
    # points = np.nonzero(frame)  # returns 2 lists - one for row, one for columns indices
    # return np.vstack(points).astype(np.float32).T  # convert to Nx2 array
    points = np.unravel_index(np.flatnonzero(frame), frame.shape)
    return np.vstack(points).astype(np.float32).T


def samplepoints(frame, nval=5000):
    frame = frame/np.sum(frame)  # normalize to make PDF
    maxval = frame.shape[0] * frame.shape[1]
    linear_indices = np.random.choice(maxval, size=(nval, 1), replace=True, p=np.reshape(frame, (maxval,)))
    points = np.unravel_index(linear_indices, frame.shape[:2])
    return np.column_stack(points).astype(np.float32)


def segment_connected_components(frame, minimal_size=None):
    """Get properties of all connected components."""
    labeled_frame, nlbl = sci.label(frame)

    if minimal_size is not None:
        size = sci.labeled_comprehension(frame, labeled_frame, range(1, nlbl + 1), np.alen,
                                         out_dtype=np.uint, default=0, pass_positions=False)

        # import ipdb; ipdb.set_trace()
        too_small = np.where(size < minimal_size)
        if np.any(too_small):
            for lbl in too_small:
                labeled_frame[labeled_frame==lbl+1] = 0  # remove
        tmp = labeled_frame.copy()
        for cnt, lbl in enumerate(np.unique(labeled_frame)):
            tmp[labeled_frame==lbl] = cnt
        labeled_frame = tmp
        nlbl = np.unique(labeled_frame).shape[0]-1
        frame[labeled_frame==0] = 0  # also remove from frame so it does not contribute to `points`

    points = getpoints(frame)
    # get labels for points
    columns = np.array(points[:, 1], dtype=np.intp)
    rows = np.array(points[:, 0], dtype=np.intp)
    labels = labeled_frame[rows, columns]

    # get centers etc
    centers = np.array(sci.center_of_mass(frame, labeled_frame, range(1, nlbl + 1)),
                       dtype=np.uint)  # convert to list from tuple
    size = sci.labeled_comprehension(frame, labeled_frame, range(1, nlbl + 1), np.alen,
                                     out_dtype=np.uint, default=0, pass_positions=False)
    std = sci.labeled_comprehension(frame, labeled_frame, range(1, nlbl + 1), np.std,
                                    out_dtype=np.uint, default=0, pass_positions=False)
    labels = np.reshape(labels, (labels.shape[0], 1))  # make (n,1), not (n,) for compatibility downstream
    return centers, labels, points, std, size, labeled_frame


def segment_center_of_mass(frame):
    """Get center of mass (median position) of all points.

    Usually very robust for single-fly tracking since it ignores small specks.
    """
    points = getpoints(frame)
    size = points.shape[0]
    if size == 0:  # if frame is empty (no super-threshold points)
        centers = None
        labels = None
        std = None
    else:
        centers = np.nanmedian(points, axis=0)
        std = np.std(points, axis=0)
        labels = np.zeros((size, 1), dtype=np.uint)  # for compatibility with multi-object methods
    return centers, labels, points, std, size


def segment_cluster(frame, num_clusters=1, term_crit=(cv2.TERM_CRITERIA_EPS, 100, 0.01), init_method=cv2.KMEANS_PP_CENTERS):
    """Cluster points to get fly positions."""
    points = getpoints(frame)
    # points = samplepoints(frame)
    cluster_compactness, labels, centers = cv2.kmeans(points, num_clusters, None, criteria=term_crit, attempts=100, flags=init_method)
    return centers, labels, points


def segment_gmm(frame, num_clusters=1, em_cov=cv2.ml.EM_COV_MAT_DIAGONAL, initial_means=None, n_samples=1000):
    """Cluster by fitting GMM."""
    points = samplepoints(frame, n_samples)
    em = cv2.ml.EM_create()
    em.setClustersNumber(num_clusters)
    em.setCovarianceMatrixType(em_cov)
    if initial_means is None:
        em.trainEM(points)
    else:
        em.trainE(points, means0=initial_means)
        _, probs = em.predict(points)
        em.trainM(points, probs)
    centers = em.getMeans()
    _, probs = em.predict(points)
    labels = np.argmax(probs, axis=1)
    return centers, labels, points


def segment_watershed(frame, marker_positions, frame_threshold=180, frame_dilation=3, marker_dilation=15, post_ws_mask=None):
    """Segment by applying watershed transform."""
    # markers provide "seeds" for the watershed transform - same shape as frame with pixels of different integer values for different segments
    if np.all(marker_positions.shape == frame.shape):
        markers = marker_positions
    else:
        markers = np.zeros(frame.shape, dtype=np.uint8)
        for cnt, marker_position in enumerate(marker_positions):
            markers[int(np.floor(marker_position[0])), int(np.floor(marker_position[1]))] = int(cnt + 1)  # +1 since 0=background
        # make segmentation more robust - dilation factor determines tolerance - bigger values allow catching the fly even if it moves quite a bit
        markers = dilate(markers, marker_dilation)
    bg_mask = frame < frame_threshold  # everything outside of mask is ignored - need to test robustness of threshold
    labeled_frame = skimage.segmentation.watershed(dilate(frame, frame_dilation), markers, mask=bg_mask, watershed_line=True)
    # process segmentation
    if post_ws_mask is not None:
        labeled_frame[post_ws_mask == 0] = 0

    labels = labeled_frame[labeled_frame > 0]  # get all foreground labels

    points = getpoints(labeled_frame > 0)  # get position of all foreground pixels
    number_of_segments = np.unique(labeled_frame)[1:].shape[0]+1  # +1 since 0 is background
    centers = np.zeros((number_of_segments, 2))
    std = np.zeros((number_of_segments, 2))
    size = np.zeros((number_of_segments, 1))
    # get stats of segments
    for ii in range(number_of_segments):  # [1:] since first unique values=0=background
        this_points = points[labels == ii, :].astype(np.uintp)
        # centers[ii, :] = this_points[np.argmin(frame[this_points[:,0], this_points[:,1]]),:]
        centers[ii, :] = np.median(this_points, axis=0)
        std[ii, :] = np.std(this_points, axis=0)
        size[ii, 0] = np.sum(labels == ii)

    labels = np.reshape(labels, (labels.shape[0], 1))  # make (n,1), not (n,) for compatibility downstream
    return centers, labels, points, std, size, labeled_frame


def find_flies_in_conn_comps(labeled_frame, positions, max_repeats=5, initial_dilation_factor=10, repeat_dilation_factor=5):
    """Assign positions to conn comps.

    fly_conncomps - conn comp fro each flycnt
    flycnt - number of flies per conn comp
    flybins - bins for cnt histogram
    cnt - n-dilation rounds
    """
    # labeled frame for assigning flies to conn comps - dilate if missing flies - not used splitting multifly comps
    labeled_frame_id = labeled_frame.copy()
    labeled_frame_id = dilate(labeled_frame_id, kernel_size=initial_dilation_factor)  # grow by 10 to connect nearby comps
    nflies = positions.shape[0]
    # if any flies outside of any component - grow blobs and repeat conn comps
    cnt = 0  # n-repeat of conn comp
    flycnt = [nflies]  # init with all flies outside of conn comps
    while flycnt[0] > 0 and cnt < max_repeats:  # flies outside of conn comps
        if cnt is not 0:  # do not display on first pass
            # print(f"{flycnt[0]} outside of the conn comps - growing blobs")
            if flycnt[0] == nflies:
                print('   something is wrong')
            labeled_frame_id = dilate(labeled_frame_id, kernel_size=repeat_dilation_factor)
        # get conn comp each fly is in using previous position
        fly_conncomps = labeled_frame_id[np.uintp(positions[:, 0]), np.uintp(positions[:, 1])]
        # count number of flies per conn comp
        flycnt, flybins = np.histogram(fly_conncomps, bins=-0.5 + np.arange(np.max(labeled_frame+2)))
        cnt += 1
    flybins = np.uintp(flybins + 0.5)  # make bins indices/labels
    return fly_conncomps, flycnt, flybins, cnt


def split_connected_components_cluster(flybins, flycnt, this_labels, labeled_frame, points, nflies, do_erode=False):
    """Split conn compts containing more than one fly."""
    labels = this_labels.copy()  # copy for new labels
    # split conn compts with multiple flies using clustering
    for con in np.uintp(flybins[flycnt > 1]):
        # cluster points for current conn comp

        con_frame = labeled_frame == con
        # erode to increase separation between flies in a blob
        if do_erode:
            con_frame = erode(con_frame, kernel_size=5)
        con_centers, con_labels, con_points = segment_cluster(con_frame, num_clusters=flycnt[con])
        if do_erode:  # with erosion:
            # delete old labels and points - if we erode we will have fewer points
            points = points[labels[:, 0] != con, :]
            labels = labels[labels[:, 0] != con]
            # append new labels and points
            labels = np.append(labels, np.max(labels) + 10 + con_labels, axis=0)
            points = np.append(points, con_points, axis=0)
        else:  # w/o erosion:
            labels[this_labels == con] = np.max(labels) + 10 + con_labels[:, 0]

    # make labels consecutive numbers again
    new_labels = np.zeros_like(labels)
    for cnt, label in enumerate(np.unique(labels)):
        new_labels[labels == label] = cnt
    labels = new_labels.copy()
    # if np.unique(labels).shape[0]>nflies:
    # import ipdb; ipdb.set_trace()
    # plt.imshow(labeled_frame);plt.plot(old_centers[ii-1,:,1], old_centers[ii-1,:,0], '.r')
    # plt.scatter(points[:,1], points[:,0], c=labels[:,0])
    # calculate center values from new labels
    centers = np.zeros((nflies, 2))
    for label in np.unique(labels):
        centers[label, :] = np.median(points[labels[:, 0] == label, :], axis=0)
    return centers, labels, points


def split_connected_components_watershed(flybins, flycnt, this_labels, labeled_frame, points, nflies, marker_positions, frame, do_erode=False):
    """Split conn compts containing more than one fly."""
    labels = this_labels.copy()  # copy for new labels
    # split conn compts with multiple flies using clustering
    for con in np.uintp(flybins[flycnt > 1]):
        # cluster points for current conn comp
        con_frame = labeled_frame == con
        # erode to increase separation between flies in a blob
        if do_erode:
            con_frame = erode(con_frame, kernel_size=5)
        con_centers, con_labels, con_points = segment_cluster(con_frame, num_clusters=flycnt[con])
        if do_erode:  # with erosion:
            # delete old labels and points - if we erode we will have fewer points
            points = points[labels[:, 0] != con,:]
            labels = labels[labels[:, 0] != con]
            # append new labels and points
            labels = np.append(labels, np.max(labels) + 10 + con_labels, axis=0)
            points = np.append(points, con_points, axis=0)
        else:  # w/o erosion:
            labels[this_labels == con] = np.max(labels) + 10 + con_labels[:, 0]

    # make labels consecutive numbers again
    new_labels = np.zeros_like(labels)
    for cnt, label in enumerate(np.unique(labels)):
        new_labels[labels == label] = cnt
    labels = new_labels.copy()
    # if np.unique(labels).shape[0]>nflies:
    # import ipdb; ipdb.set_trace()
    # plt.imshow(labeled_frame);plt.plot(old_centers[ii-1,:,1], old_centers[ii-1,:,0], '.r')
    # plt.scatter(points[:,1], points[:,0], c=labels[:,0])
    # calculate center values from new labels
    centers = np.zeros((nflies, 2))
    for label in np.unique(labels):
        centers[label, :] = np.median(points[labels[:, 0] == label, :], axis=0)
    return centers, labels, points


def detect_led(frame, channel=-1):
    """Detect LED as darkest spot in channel(-1) in frame corner."""
    # vertical and horizontal size of corners
    vsize = 80
    hsize = 200
    v = frame.shape[0]
    h = frame.shape[1]

    corner_brightness = list()
    # define slices for all four corners of the frame
    corner_slices = ((slice(0, vsize), slice(0, hsize)),
                     (slice(0, vsize), slice(h-hsize, h)),
                     (slice(v-vsize, v), slice(0, hsize)),
                     (slice(v-vsize, v), slice(h-hsize, h)))

    # calculate brightness for each corner
    for corner_slice in corner_slices:
        corner = frame[corner_slice[0], corner_slice[1], channel]
        corner_brightness.append(np.mean(corner))

    # darkest corner cotains led
    led_corner = corner_slices[np.argmin(corner_brightness)]
    # extract start stop indices in correct order for `Foreground.crop` - weird order to accomodate fg.crop
    led_coords = [led_corner[1].start, led_corner[0].start, led_corner[1].stop, led_corner[0].stop]
    return led_coords


def get_chambers(background, chamber_threshold=0.6, min_size=40000, max_size=50000, kernel_size=11):
    """Detect (bright) chambers in background."""
    if len(background.shape) > 2:
        background = background[:, :, 0]
    background_thres = np.double(threshold(background, chamber_threshold * np.mean(background)))
    background_thres = close(background_thres, kernel_size=kernel_size)
    background_thres = dilate(background_thres, kernel_size=kernel_size)
    # initial guess of chambers
    _, _, _, _, area, labeled_frame = segment_connected_components(background_thres)
    # add dummy valye for the area of the background, which is not returned by `segment_connected_components`
    area = np.insert(area, 0, 0)
    # weed out too small chambers based on area
    unique_labels = np.unique(labeled_frame)
    condlist = np.any([area < min_size, area > max_size], axis=0)
    unique_labels[condlist] = 0
    labeled_frame, _, _ = clean_labels(labeled_frame, unique_labels, force_cont=True)
    return labeled_frame


def get_chambers_chaining(background):
    """Find circular chamber."""
    # find circlular chamber
    circles = None  # init
    p1 = 200  # initial parameter
    while circles is None:  # as long as there is no chamber
        circles = cv2.HoughCircles(background.astype(np.uint8),cv2.HOUGH_GRADIENT,1,100, param1=p1,param2=40,minRadius=int(background.shape[0]/3),maxRadius=int(background.shape[0]/2))
        p1 = p1-10  # slowly decrease param
    circles = np.uint16(np.around(circles))

    # create binary mask
    mask = np.zeros(background.shape, dtype=np.uint8)
    for i in circles[0,:]:
        cv2.circle(mask,(i[0],i[1]),i[2]+30,255,-1)
    mask = mask > 0  # make binary
    return mask, circles


def get_bounding_box(labeled_frame):
    """Get bounding boxes of all components."""
    uni_labels = np.unique(labeled_frame)
    bounding_box = np.ndarray((np.max(uni_labels) + 1, 2, 2), dtype=np.int)
    for ii in range(uni_labels.shape[0]):
        points = getpoints(labeled_frame == uni_labels[ii])
        bounding_box[uni_labels[ii], :, :] = np.vstack((np.min(points, axis=0), np.max(points, axis=0)))
    return bounding_box


def annotate(frame, centers=None, lines=None):
    """annotate frame"""
    if centers is not None or lines is not None:
        try:
            nflies = centers.shape[0]
        except Exception as e:
            nflies = lines.shape[0]

        colors = np.zeros((1, nflies, 3), np.uint8)
        colors[0, :] = 220
        colors[0, :, 0] = np.arange(0, 180, 180.0/nflies)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.float32) / 255.0
        colors = [list(map(float, thisColor)) for thisColor in colors]

    if centers is not None:
        for idx, center in enumerate(centers):
            cv2.circle(frame, (center[1], center[0]), radius=6, color=colors[idx], thickness=1)
    if lines is not None:
        for idx, line in enumerate(lines):
            cv2.line(frame, tuple(line[0]), tuple(line[1]), color=colors[idx], thickness=1)
    return frame


def show(frame, window_name="frame", time_out=1, autoscale=False):
    """display frame"""
    if autoscale:
        if len(frame.shape) == 3:
            maxval = np.max(frame)  # not tested
            frame = frame / maxval
        else:
            frame = frame / np.max(frame)
    cv2.imshow(window_name, np.float32(frame))
    cv2.waitKey(time_out)


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


if __name__ == "__main__":
    test()
