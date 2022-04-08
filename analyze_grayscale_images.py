"""Get features from a grayscale image.
Sorry, the features from the NGTDM are not calculated.
"""

import os
import sys

import numpy as np


def main():
    """Main function."""

    # Load an image as grayscale.
    img = load_img()
    assert img.ndim==2, 'The image is not 2-dimentional ndarray.'

    # Statistics.
    feats_stats = get_stats(img)
    # Peak to peak and differences.
    h_pp, v_pp = get_pp(img)
    h_diff, v_diff = get_diff(img)
    # GLCM.
    glcm = get_glcm(img)
    feats_glcm = analyze_glcm(glcm)
    # NGTDM.
    ngtdm = get_ngtdm(img)

    all_features = feats_stats.copy()
    all_features.update(h_pp=h_pp,
                        v_pp=v_pp,
                        h_diff=h_diff,
                        v_diff=v_diff,
                        )
    all_features.update(feats_glcm)

    for key in all_features:
        print(key, all_features[key])


def load_img():
    """Load an image."""

    img = np.array([[10,10,10,10,10],
                    [20,20,20,20,20],
                    [25,25,10,25,25],
                    [20,20,20,20,20],
                    [25,25,10,25,25],
                    ])

    return img

def get_stats(img):
    """Calculate statistics as features."""

    img_min = np.amin(img)
    img_max = np.amax(img)
    img_mean = np.mean(img)
    img_med = np.median(img)
    img_std = np.std(img, ddof=0)

    feats = {}
    feats['min'] = img_min
    feats['max'] = img_max
    feats['mean'] = img_mean
    feats['median'] = img_med
    feats['std'] = img_std

    return feats


def get_pp(img):
    """Calculate the peak to peak to each direction.
    Averaged by each direction lenth.
    """

    # Horizontal direction.
    h_pp = np.amax(img, axis=1) - np.amin(img, axis=1)
    h_pp = np.sum(h_pp)
    base = img.shape[0]  # Height.
    h_pp = h_pp / base
    # Vertical direction.
    v_pp = np.amax(img, axis=0) - np.amin(img, axis=0)
    v_pp = np.sum(v_pp)
    base = img.shape[1]  # width.
    v_pp = v_pp / base

    return h_pp, v_pp


def get_diff(img):
    """Calculate differences to each direction.
    Averaged by following bases.
    Horizontal base: (width - 1) * height
    Vertical base: (height - 1) * width
    """

    # Horizontal direction.
    h_diff = img[:, :-1] - img[:, 1:]
    h_diff = np.abs(h_diff)
    h_diff = np.sum(h_diff)
    base = (img.shape[1] - 1) * img.shape[0]
    h_diff = h_diff / base
    # Vertical direction.
    v_diff = img[:-1] - img[1:]
    v_diff = np.abs(v_diff)
    v_diff = np.sum(v_diff)
    base = (img.shape[0] - 1) * img.shape[1]
    v_diff = v_diff / base

    return h_diff, v_diff


def get_glcm(img):
    """Calculate the GLCM (Glay-Level Co-occurrence Matrix).
    Used img has the gradation of 0-255.
    In order to calculate easier, converted img to 1-64.
    """

    # Convet the image to 6 bit.
    BIT = 6
    step = 2 ** BIT
    img = img / 255 * (step-1)
    img = img + 1
    # Truncate numbers after the dicimal point.
    img = img.astype(np.uint8)

    # Calculate the GLCM.
    glcm = np.zeros((step, step))
    img_pad = np.pad(img, 1)
    # Slect the direction with the following kernel.
    kernel = np.array([[1, 1, 1],
                       [0, 0, 1],
                       [0, 0, 0],
                       ])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # A target pixel value.
            target = img[i, j]
            # Get values arround the target pixel.
            filtered = img_pad[i:i+3, j:j+3]
            filtered = filtered * kernel
            filtered = filtered.flatten()
            # Remove elements that equal to 0.
            idx_list = np.nonzero(filtered)
            filtered = filtered[idx_list]
            # Count up the GLCM.
            for k in filtered:
                glcm[target-1, k-1] += 1

    glcm = glcm / np.sum(glcm)

    return glcm


def analyze_glcm(glcm):
    """Calculate features from the GLCM matrix."""

    # Anglar second moment (energy, uniformity).
    asm = np.sum(glcm ** 2)
    # Entropy.
    entropy = glcm[glcm>0] * np.log(glcm[glcm>0])
    entropy = -np.sum(entropy)
    # Dissimilarity.
    matrix_i = np.arange(glcm.shape[1])
    matrix_j = np.arange(glcm.shape[0])
    matrix_i, matrix_j = np.meshgrid(matrix_i, matrix_j)
    dissimilarity = np.abs(matrix_i - matrix_j)
    dissimilarity = glcm * dissimilarity
    dissimilarity = np.sum(dissimilarity)
    # Contrast.
    contrast = (matrix_i - matrix_j) ** 2
    contrast = glcm * contrast
    contrast = np.sum(contrast)
    # Homogeneity.
    homogeneity = (matrix_i - matrix_j) ** 2
    homogeneity = glcm / (1 + homogeneity)
    homogeneity = np.sum(homogeneity)
    # Maximum probability.
    mp = np.amax(glcm)

    feats = {}
    feats['asm'] = asm
    feats['entropy'] = entropy
    feats['dissimilarity'] = dissimilarity
    feats['contrast'] = contrast
    feats['mp'] = mp

    return feats


def get_ngtdm(img):
    """Calculate the NGTDM (Neighbourhood Gray-Tone-Difference Matrix).
    Used img has the gradation of 0-255.
    """

    # Convet the image to 6 bit.
    BIT = 6
    step = 2 ** BIT
    img = img / 255 * (step-1)
    # Truncate numbers after the dicimal point.
    img = img.astype(np.uint8)

    # Calculate the average value matrix.
    height = img.shape[0] - 2
    width = img.shape[1] - 2
    ave_matrix = np.zeros((height, width))
    tgt_matrix = np.zeros((height, width))
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       ])
    for i in range(height):
        for j in range(width):
            # A target pixel value.
            tgt_matrix[i, j] = img[i+1, j+1]
            # Get the average value arround the target pixel.
            filtered = img[i:i+3, j:j+3]
            filtered = filtered * kernel
            filtered = np.sum(filtered) / 8
            ave_matrix[i, j] = filtered

    # Generate the NGTDM from the average value matrix.
    ave_matrix = tgt_matrix - ave_matrix
    ave_matrix = np.abs(ave_matrix)
    ngtdm = np.zeros(step)
    for i in range(step):
        val = ave_matrix[tgt_matrix==i]
        val = np.sum(val)
        ngtdm[i] = val

    return ngtdm


if __name__ == '__main__':
    main()
