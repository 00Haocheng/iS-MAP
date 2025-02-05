import os
import glob
import numpy as np
import cv2
from skimage.color import label2rgb

class ScanNet():
    def __init__(self, scanpath
                 ):
        super(ScanNet, self).__init__()
        self.input_folder=scanpath
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/depth/*.png'), key=lambda x: int(os.path.basename(x)[:-4]))
        self.n_img = len(self.depth_paths)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/color/frame*.jpg'), key=lambda x: int(os.path.basename(x)[5:-4]))

def extract_lineseg(filename,thread):
    CROP = 16
    image = cv2.imread(filename, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    corp_image = image[CROP:-CROP, CROP:-CROP]

    #lsd = cv2.createLineSegmentDetector(0, _scale=1)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(corp_image)[0]
    lines = np.squeeze(lines, 1)
    lengths = np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2)
    arr1inds = lengths.argsort()[::-1]
    lengths = lengths[arr1inds[::-1]]
    lines = lines[arr1inds[::-1]]
    lines = lines[lengths > np.sqrt(corp_image.shape[0] ** 2 + corp_image.shape[1] ** 2) / thread]

    lines = lines[:min(lines.shape[0], 255)]

    line_seg = np.zeros([h, w], dtype=np.int)

    n = 1
    for k in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[k]

        xmin = max(0, int(np.floor(min(x1, x2))))
        xmax = min(int(np.ceil(max(x1, x2))), w - 2 * CROP)

        ymin = max(0, int(np.floor(min(y1, y2))))
        ymax = min(int(np.floor(max(y1, y2))), h - 2 * CROP)

        points = []
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                p = np.array([i, j])
                vec1 = lines[k, :2] - p
                vec2 = lines[k, 2:] - p
                distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(lines[k, :2] - lines[k, 2:])
                if distance < 1:
                    points.append([CROP + j, CROP + i])

        if len(points) < 3:
            continue
        else:
            for p in points:
                line_seg[p[0], p[1]] = n
            n += 1

    return line_seg


#Choose dataset type
dataset="replica"
# dataset="scannet"

if dataset=="replica":
    re_dir = "Dataset/Replica/room0/results"#modify the data path here
    re_dir_store = "Dataset/Replica/room0/line_seg"#modify the output path here
    re_search = re_dir + "/frame*.jpg"
    re_files = sorted(glob.glob(re_search))
    thread=10
elif dataset=="scannet":
    scannet_path = "Dataset/ScanNet/scans/scene_0000_00" #modify the data path here
    re_dir = scannet_path + "/color"
    scan_data = ScanNet(scannet_path)
    re_files = scan_data.color_paths
    re_dir_store = "Dataset/ScanNet/scans/scene_0000_00/line_seg" #modify the output path here
    thread=8
num=0
if not os.path.exists(re_dir_store):
    os.makedirs(re_dir_store)
for filename in re_files:
    if dataset == "replica":
        index = filename.split('/')[-1].split('e')[1]
        index = int(index.split(".")[0])
    elif dataset == "scannet":
        index = filename.split('/')[-1]
        index=index.split('.')[0]
        index = index.split('e')[1]
        index = int(index)
    if num % 1 == 0:
        segment =extract_lineseg(filename,thread)
        cv2.imwrite(os.path.join(re_dir_store, "{:05d}_seg.png".format(index)),
                    segment.astype(np.uint8))  # write seg_image
        # # For visualize
        # color = label2rgb(segment, bg_label=0)  # transfer seg_image to vis_seg_image
        # color = (255 * color).astype(np.uint8)
        #
        # cv2.imwrite(os.path.join(re_dir_store, "{:d}_seg.jpg".format(index)), color)  # write vis_seg_image
        num=num+1


