import os
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class RedLightDetector:

    def __init__(self):
        self.filters = [f for i, f in enumerate(self.load_filters()) if i in [0, 1, 2]]
        self.filter_thresholds = [0.9, 0.95, 0.9]

    def load_filters(self):
        filters_path = 'filters/'
        filter_names = [f for f in sorted(os.listdir(filters_path)) if '.jpg' in f]
        filters = []

        for i in range(len(filter_names)):
            f = Image.open(os.path.join(filters_path, filter_names[i]))
            f_img = np.asarray(f)
            filters.append(f_img.copy().astype(float))

        return filters


    def detect_red_light(self, I, display_image=False):
        '''
        This function takes a numpy array <I> and returns a list <bounding_boxes>.
        The list <bounding_boxes> should have one element for each red light in the
        image. Each element of <bounding_boxes> should itself be a list, containing
        four integers that specify a bounding box: the row and column index of the
        top left corner and the row and column index of the bottom right corner (in
        that order). See the code below for an example.

        Note that PIL loads images in RGB order, so:
        I[:,:,0] is the red channel
        I[:,:,1] is the green channel
        I[:,:,2] is the blue channel
        '''

        bounding_boxes = []
        I_orig = I

        I = I.copy().astype(float)

        norm_means = []
        norm_stds = []

        # normalize image around [-1, 1] ish
        for i in range(3):
            mean = np.mean(I[:, :, i])
            std = np.std(I[:, :, i])
            norm_means.append(mean)
            norm_stds.append(std)

            I[:, :, i] = (I[:, :, i] - mean) / (2 * std)

        for filt_num, filt in enumerate(self.filters):

            # normalize filter with the same transformation we applied to the image
            for i in range(3):
                filt[:, :, i] = (filt[:, :, i] - norm_means[i]) / norm_stds[i]

            filt_width = filt.shape[0]
            filt_height = filt.shape[1]

            n_horizontal_prods = I.shape[0] - filt_width + 1
            n_vertical_prods = I.shape[1] - filt_height + 1

            result = np.empty((n_horizontal_prods, n_vertical_prods))

            filt = filt.flatten()

            # perform convolutions, skipping every other pixel
            for x in range(0, n_horizontal_prods, 2):
                for y in range(0, n_vertical_prods, 2):
                    section = I[x:x + filt_width, y:y + filt_height, :].copy().astype(float)
                    section = section.flatten()
                    section = section / np.linalg.norm(section)
                    result[x, y] = np.dot(filt / np.linalg.norm(filt), section)

            result[result < self.filter_thresholds[filt_num]] = 0

            for x, y in zip(*result.nonzero()):
                x = x
                y = y
                x2 = x + filt_width
                y2 = y + filt_height
                merged = False
                # check for mergeable bounding boxes and merge them if we can. this isn't perfect, especially when one
                # box surrounds the other.
                for other_rect in bounding_boxes:
                    for xc, yc in [(x, y), (x2, y), (x, y2), (x2, y2)]:
                        if other_rect[0] <= xc <= other_rect[2] and other_rect[1] <= yc <= other_rect[3]:
                            other_rect[0] = min(int(x), other_rect[0])
                            other_rect[1] = min(int(y), other_rect[1])
                            other_rect[2] = max(int(x2), other_rect[2])
                            other_rect[3] = max(int(y2), other_rect[3])
                            merged = True

                if not merged:
                    bounding_boxes.append([int(k) for k in [x, y, x2, y2]])

        if display_image:
            fig, ax = plt.subplots()
            ax.imshow(I_orig.astype(int))

            for box in bounding_boxes:
                rect = patches.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.show()

        for i in range(len(bounding_boxes)):
            assert len(bounding_boxes[i]) == 4
        # print(bounding_boxes)

        return bounding_boxes


    def downsample_image(self, I, scale=0.5):
        n_rows = int(np.ceil(I.shape[0] * scale))
        n_cols = int(np.ceil(I.shape[1] * scale))

        # First, do bilinear interpolation on rows
        row_result = np.empty((n_rows, I.shape[1], 3))
        for j in range(n_rows):
            i = I.shape[0] * j / n_rows
            i0 = int(np.floor(i))
            i1 = int(np.ceil(i))
            if i == i0:
                i1 = i0 + 1
            row_result[j, :, :] = (i - i0) * I[i0, :, :] + (i1 - i) * I[i1, :, :]

        result = np.empty((n_rows, n_cols, 3))

        # Then, do the same to columns
        for j in range(n_cols):
            i = I.shape[1] * j / n_cols
            i0 = int(np.floor(i))
            i1 = int(np.ceil(i))
            if i == i0:
                i1 = i0 + 1
            result[:, j, :] = (i - i0) * row_result[:, i0, :] + (i1 - i) * row_result[:, i1, :]

        return result.astype(int)
