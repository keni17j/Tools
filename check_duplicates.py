"""Check for duplicates of datas.
(1) one-directional array.
(2) two-directional array.
True means no dupulication.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance


def main():
    """Main function."""

    # Get the test data.
    df = get_datas()
    # Check for duplicates.
    print('\n''Check for duplicates between one-dimentional datas.')
    dup_1d(df)
    print('\n''Check for duplicates between two-dimentional datas.')
    dup_2d(df)


def get_datas():
    """Get datas or create dummy datas."""

    col_list = ['group','val1', 'val2']
    group = np.empty(30, dtype=object)
    group[:10] = 'A'
    group[10:20] = 'B'
    group[20:] = 'C'
    val = np.hstack((np.random.randint(0, 10, (2, 10)),
                     np.random.randint(10, 20, (2, 10)),
                     np.random.randint(20, 30, (2, 10)),
                     ))
    val = np.vstack((group, val))
    val = np.transpose(val)
    df = pd.DataFrame(val, columns=col_list)

    return df

def dup_1d(df):
    """Check duplicates in one-dimentional datas.
    Using min-max values.
    """

    target = 'val1'
    # Extract the target column datas and divide to each group.
    group_list = df['group'].unique()
    data_list = []
    for i in group_list:
        data_temp = df[df['group']==i]
        data_temp = data_temp[target]
        data_temp = data_temp.to_numpy()
        data_list.append(data_temp)
    data_list = np.array(data_list)

    for i, j in zip(group_list, data_list):
        print(i, j)

    # Check for duplicates.
    min_list = np.amin(data_list, axis=1)
    max_list = np.amax(data_list, axis=1)
    min_x, min_y = np.meshgrid(min_list, min_list)
    max_x, max_y = np.meshgrid(max_list, max_list)
    flag_a = min_y > max_x
    flag_b = max_y < min_x
    flag = np.logical_or(flag_a, flag_b)
    df = pd.DataFrame(flag, columns=group_list, index=group_list)
    print(df)


def dup_2d(df):
    """Check duplicates in two-dimentional datas.
    Using the Mahalanobis distance.
    """

    target = ['val1', 'val2']
    # Extract target columns datas and divide to each group.
    group_list = df['group'].unique()
    data_list = []
    for i in group_list:
        data_temp = df[df['group']==i]
        data_temp = data_temp[target]
        data_temp = data_temp.to_numpy()
        data_list.append(data_temp)

    for i, j in zip(group_list, data_list):
        print(i, j)

    # Check for duplicates.
    flag_list = []
    for data in data_list:
        d_list, mean, cov_inv = mahalanobis(data)
        d_max = np.amax(d_list)
        temp_list = []
        for data in data_list:
            d_list, _, _ = mahalanobis(data, mean, cov_inv)
            flag = np.amin(d_list) > d_max
            temp_list.append(flag)
        flag_list.append(temp_list)

    df = pd.DataFrame(flag_list, columns=group_list, index=group_list)
    print(df)
    graph_maha(data_list, group_list)


def mahalanobis(data, mean=None, cov_inv=None):
    """Calculate the Mahalanobis distance with unbiased variance.
    To avoid getting negative values,
    the number of datas must be greater than the number of variables.
    In order to check for duplicates, return mean and cov_inv.
    """

    assert data.ndim == 2, 'The data must be two-dimensional ndarray.'

    # Get two parameters that used in the following calculation.
    if mean is None:
        mean = np.mean(data, axis=0)
    if cov_inv is None:
        x = data[:, 0]
        y = data[:, 1]
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        var_xy = np.sum((x-np.mean(x)) * (y-np.mean(y))) / (len(x)-1)
        cov = np.array([[var_x, var_xy],
                        [var_xy, var_y],
                        ])
        # Another way to calculate the covariance matrix.
        #data = data.astype(float)
        #cov = np.cov(data.T, ddof=1)
        cov_inv = np.linalg.inv(cov)

    # Calculate the Mahalanobis distance.
    d_list = []
    for i in data:
        d = np.dot((i-mean), cov_inv)
        d = np.dot(d, (i-mean))
        d = np.sqrt(d)
        # Another way to calculate the distance.
        #d = distance.mahalanobis(i, mean, np.linalg.inv(cov))
        d_list.append(d)
    d_list = np.array(d_list)

    return d_list, mean, cov_inv


def graph_maha(data_list, group_list):
    """Draw a graph.
    (1) Plot datas of each group.
    (2) Draw contour lines for each group.
    """

    fig = plt.figure()

    # Plot each datas.
    for data in data_list:
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y)
    plt.legend(group_list)
    plt.tick_params(which='both', direction='in')

    # Draw the Mahalanobis distance.
    x_min, x_max = plt.xlim()
    x_tick = np.linspace(x_min, x_max, 50)
    y_min, y_max = plt.ylim()
    y_tick = np.linspace(y_min, y_max, 50)
    x_tick, y_tick = np.meshgrid(x_tick, y_tick)
    for data in data_list:
        d_list, mean, cov_inv = mahalanobis(data)
        d_max = np.amax(d_list)
        # Get distances for each point in the grid (x_tick, y_tick).
        z = []
        for i, j in zip(x_tick, y_tick):
            z_temp = np.vstack((i, j)).T
            z_temp, _, _ = mahalanobis(z_temp, mean, cov_inv)
            z.append(z_temp)
        # Draw the line.
        plt.contour(x_tick, y_tick, z, levels=[d_max,])

    plt.show()


if __name__ == '__main__':
    main()
