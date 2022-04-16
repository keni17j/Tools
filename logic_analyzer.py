"""Analyze datas that measured by the logic analyzer.
Extract datas with following conditions.
(1) The mask signal = '1'.
(2) Rising edges and falling edges of the clock signal.
(3) After the ID.
"""

import os
import sys

import numpy as np
import pandas as pd


def main():
    """Main function."""

    # Read the csv file.
    file_path = input('Drag and Drop a csv file --> ')
    header = 0
    enc = 'utf-8'
    df = pd.read_csv(file_path, header=header, encoding=enc)
    print('\n''Signal names', df.columns.to_numpy())

    # Sort.
    #df = df.sort_index(ascending=False)

    # Get indices of the period that the mask signal = 1.
    col_name = input('\n''The mask signal name --> ')
    mask = df[col_name].to_numpy()
    mask_rise, mask_fall = get_edge(mask)
    print('Rising edges of the mask', mask_rise)
    print('Falling edges of the mask', mask_fall)
    print('The number of masks:', len(mask_rise))

    # Split datas using the mask.
    # Get clock edges and corresponded datas to them.
    col_name = input('\n''The clock signal name --> ')
    df_list = []
    for i, j in zip(mask_rise, mask_fall):
        # Extract datas when the mask signal = 1.
        df_temp = df.iloc[i:j]
        # Get clock edges.
        df_clk = df_temp[col_name].to_numpy()
        clk_rige, clk_fall = get_edge(df_clk)
        clk_edge = np.hstack((clk_rige, clk_fall))
        clk_edge = np.sort(clk_edge)
        n = len(clk_edge)
        print('Indices %d to %d, the number of clock edges is %d' % (i, j, n))
        # Extract datas at the edge.
        df_temp = df_temp.iloc[clk_edge]
        df_list.append(df_temp)

    # Select the data signal.
    # Search the ID data.
    col_name = input('\n''The data signal name --> ')
    id = input('\n''The ID (8 bit) --> ')
    data_list = []
    for df in df_list:
        data = df[col_name].to_numpy()
        data = ext_id(data, id)
        print('The data is %d bit.' % len(data))
        data_list.append(data)

    # Convert datas to the format.
    print('\n''Output following datas.')
    for i, data in enumerate(data_list):
        data_list[i] = set_format(data)
        print(data_list[i])

    # Save as a csv file.
    file_path = os.path.abspath(file_path)
    file_path = os.path.dirname(file_path)
    file_path = os.path.join(file_path, 'output.csv')
    data_list = pd.DataFrame(data_list)
    data_list.to_csv(file_path, encoding=enc)


def get_edge(sig):
    """Get edges of a signal.
    There are two rules.
    (1) The first edge is rising edge.
    (2) The final edge is falling edge.
    """

    edge_rise = []
    edge_fall = []
    for i in range(len(sig)-1):
        if sig[i] == 0 and sig[i+1] == 1:
            edge_rise.append(i+1)
        elif sig[i] == 1 and sig[i+1] == 0:
            edge_fall.append(i+1)
    if edge_rise[0] > edge_fall[0]:
        edge_fall = edge_fall[1:]
    if edge_rise[-1] > edge_fall[-1]:
        edge_rise = edge_rise[:-1]

    edge_rise = np.array(edge_rise)
    edge_fall = np.array(edge_fall)

    return edge_rise, edge_fall


def ext_id(data, id):
    """Convert each element to str.
    Extract datas with the ID."""

    data = [str(i) for i in data]
    id = list(id)
    num_id = len(id)
    num_data = len(data)
    for i in range(num_data-num_id):
        flg = data[i:i+num_id] == id
        if np.all(flg):
            data = data[i:]
            break

    if num_data == len(data):
        print('No ID in this data.')

    return data


def set_format(data):
    """Set the data in the requested format.
    (1) Devide into 8 bits.
    (2) Convert to the format 'XXXX XXXX'.
    """

    n = len(data)
    data_list = []
    for i in range(0, n, 8):
        data_temp = ''.join(data[i:i+4]) + ' '
        data_temp = data_temp + ''.join(data[i+4:i+8])
        data_list.append(data_temp)

    return data_list


if __name__ == '__main__':
    main()
