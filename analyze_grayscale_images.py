"""Get features from a grayscale image.
Features of NGTDM are uncertain.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Main function."""

    # Load an image as grayscale.
    img = load_img()
    assert img.ndim == 2, 'The image is not 2-dimentional ndarray.'

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
    feats_ngtdm = analyze_ngtdm(ngtdm)

    # Show as graphs.
    graph(glcm, ngtdm)

    all_features = feats_stats.copy()
    all_features.update(h_pp=h_pp,
                        v_pp=v_pp,
                        h_diff=h_diff,
                        v_diff=v_diff,
                        )
    all_features.update(feats_glcm)
    all_features.update(feats_ngtdm)

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
    img = img * 10
    img = img / img * 200
    print(img)

    return img


def get_stats(img):
    """Calculate statistics as features."""

    img_min = np.amin(img)
    img_max = np.amax(img)
    img_mean = np.mean(img)
    img_med = np.median(img)
    img_std = np.std(img, ddof=0)

    feats_dict = {}
    feats_dict['min'] = img_min
    feats_dict['max'] = img_max
    feats_dict['mean'] = img_mean
    feats_dict['median'] = img_med
    feats_dict['std'] = img_std

    return feats_dict


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
    base = (img.shape[1]-1) * img.shape[0]
    h_diff = h_diff / base
    # Vertical direction.
    v_diff = img[:-1] - img[1:]
    v_diff = np.abs(v_diff)
    v_diff = np.sum(v_diff)
    base = (img.shape[0]-1) * img.shape[1]
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
            # Remove elements that equal to 0.
            filtered = filtered[filtered>0]
            # Count up the GLCM.
            for k in filtered:
                glcm[target-1, k-1] += 1

    glcm = glcm / np.sum(glcm)

    return glcm


def analyze_glcm(glcm):
    """Calculate features from the GLCM."""

    # Anglar second moment (energy, uniformity).
    asm = np.sum(glcm ** 2)
    # Entropy.
    entropy = glcm[glcm>0] * np.log2(glcm[glcm>0])
    entropy = -np.sum(entropy)
    # Dissimilarity.
    mat_i = np.arange(glcm.shape[1])
    mat_j = np.arange(glcm.shape[0])
    mat_i, mat_j = np.meshgrid(mat_i, mat_j)
    dissimilarity = glcm * np.abs(mat_i-mat_j)
    dissimilarity = np.sum(dissimilarity)
    # Contrast.
    contrast = glcm * (mat_i-mat_j)**2
    contrast = np.sum(contrast)
    # Homogeneity.
    homogeneity = glcm / (1+np.abs(mat_i-mat_j))
    homogeneity = np.sum(homogeneity)
    # Maximum probability.
    mp = np.amax(glcm)
    # Correlation.
    mean_i = mat_i * glcm
    mean_i = np.sum(mean_i)
    mean_j = mat_j * glcm
    mean_j = np.sum(mean_j)
    var_i = (mat_i-mean_i) ** 2
    var_i = glcm * var_i
    var_i = np.sum(var_i)
    var_j = (mat_j-mean_j) ** 2
    var_j = glcm * var_j
    var_j = np.sum(var_j)
    if (var_i==0) or (var_j==0):
        correlation = 1
        print('Devided by 0 when calculating the correlation.')
    else:
        correlation = (mat_i-mean_i) * (mat_j-mean_j)
        correlation = correlation / np.sqrt(var_i*var_j)
        correlation = glcm * correlation
        correlation = np.sum(correlation)

    feats_dict = {}
    feats_dict['asm'] = asm
    feats_dict['entropy'] = entropy
    feats_dict['dissimilarity'] = dissimilarity
    feats_dict['glcm_contrast'] = contrast
    feats_dict['mp'] = mp
    feats_dict['correlation'] = correlation

    return feats_dict


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
            filtered = np.sum(filtered) / np.sum(kernel)
            ave_matrix[i, j] = filtered

    # Generate the NGTDM parameters.
    # ni: The number of pixels in each gray level.
    img = img[1:-1, 1:-1]
    ni = np.arange(step)
    for i in ni:
        ni[i] = np.count_nonzero(img==i)
    # pi: Probability.
    pi = ni / np.sum(ni)
    # si: The sum of absolute differences.
    ave_matrix = tgt_matrix - ave_matrix
    ave_matrix = np.abs(ave_matrix)
    si = np.zeros(step)
    for i in range(step):
        val = ave_matrix[tgt_matrix==i]
        val = np.sum(val)
        si[i] = val

    ngtdm = {}
    ngtdm['ni'] = ni
    ngtdm['pi'] = pi
    ngtdm['si'] = si

    return ngtdm


def analyze_ngtdm(ngtdm):
    """Calculate features from the NGTDM."""

    ni = ngtdm['ni']
    pi = ngtdm['pi']
    si = ngtdm['si']

    # ngp: The number of gray levels where pi > 0.
    ngp = np.count_nonzero(pi)
    # nvp: The total number of pixels.
    nvp = np.sum(ni)

    # Coarseness.
    coarseness = pi * si
    coarseness = np.sum(coarseness)
    if coarseness == 0:
        coarseness = 1e6
        print('Devided by 0 when calculating the coarseness.')
    else:
        coarseness = 1 / coarseness
    # Contrast.
    mat_pi, mat_pj = np.meshgrid(pi, pi)
    mat_i = np.arange(len(pi))
    mat_i, mat_j = np.meshgrid(mat_i, mat_i)
    if ngp == 1:
        contrast = 0
        print('Devided by 0 when calculating the contrast.')
    else:
        contrast = 1 / (ngp*(ngp-1))
        contrast = contrast * np.sum(mat_pi*mat_pj*(mat_i-mat_j)**2)
        contrast = contrast / nvp * np.sum(si)
    # Busyness.
    if ngp == 1:
        busyness = 0
        print('Devided by 0 when calculating the busyness.')
    else:
        busyness = np.sum(pi*si)
        busyness = busyness / np.sum(np.abs(mat_i*mat_pi-mat_j*mat_pj))
    # Complexity.
    mask_pi = mat_pi > 0
    mask_pj = mat_pj > 0
    mask = np.logical_and(mask_pi, mask_pj)
    mat_si, mat_sj = np.meshgrid(si, si)
    complexity = np.abs(mat_i-mat_j) * (mat_pi*mat_si+mat_pj*mat_sj)
    complexity = complexity[mask] / (mat_pi+mat_pj)[mask]
    complexity = 1 / nvp * np.sum(complexity)
    # Strenth.
    strenth = (mat_pi+mat_pj) * (mat_i-mat_j)**2
    strenth = np.sum(strenth)
    if np.sum(si) == 0:
        strenth = 0
        print('Devided by 0 when calculating the strenth.')
    else:
        strenth = strenth / np.sum(si)

    feats_dict = {}
    feats_dict['coarseness'] = coarseness
    feats_dict['ngtdm_contrast'] = contrast
    feats_dict['busyness'] = busyness
    feats_dict['complexity'] = complexity
    feats_dict['strenth'] = strenth

    return feats_dict


def graph(glcm, ngtdm):
    """Show datas as graphs."""

    w = glcm.shape[1]
    ngtdm = ngtdm['si']
    ngtdm = np.reshape(ngtdm, (-1, 1))
    ngtdm = np.tile(ngtdm, (1, w))
    fig = plt.figure(figsize=(12, 4))
    # GLCM.
    plt.subplot(1, 2, 1)
    plt.imshow(glcm, vmin=0, vmax=1)
    plt.title('GLCM')
    plt.tick_params(which='both', direction='in')
    plt.colorbar()
    # NGTDM.
    plt.subplot(1, 2, 2)
    plt.imshow(ngtdm, vmin=0, vmax=10)
    plt.xticks([])
    plt.title('NGTDM')
    plt.colorbar()
    plt.show(block=False)
    input('Input any keys to close.')
    plt.close()
    sys.exit()


if __name__ == '__main__':
    main()
