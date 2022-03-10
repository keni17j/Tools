"""Comments."""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Main function."""

    TkGUI()

class TkGUI():
    """
    Create a Tk GUI window.
    Set graph parameters.
    Draw a graph.
    """

    def __init__(self):
        """Do in the first time."""

        # Make a dummy data.
        self.dummy_data = np.arange(1500).reshape(-1, 5)
        self.dummy_column = ['dummy signal'+str(i) for i in range(5)]
        self.df = pd.DataFrame(self.dummy_data, columns=self.dummy_column)
        self.df_col = self.df.columns.values

        # Create the GUI window.
        self.root = tk.Tk()
        self.root.title('LTspice waveform viewer')
        self.parent = tk.Frame(self.root,
                               bg='#dddddd',
                               )
        self.parent.pack()
        self.frames(self.parent) # Create main frames.
        self.root.mainloop()

    def frames(self, parent):
        """Create main frames and arrange them."""

        self.make_frame00(parent, 0, 0)
        self.make_frame01(parent, 1, 0)
        self.make_frame02(parent, 2, 0)
        # Create a button.
        self.btn_draw = tk.Button(parent,
                                  text='Draw a graph',
                                  fg='#000000',
                                  bg='#dddddd',
                                  )
        self.btn_draw.bind('<Button-1>', self.btn_draw_click)
        self.btn_draw.grid(row=3, column=0, pady=10)


    def make_frame00(self, parent, row, col):
        """Create a frame and generate widgets.
        Select a file path.
        Load the file.
        """

        # Create a frame.
        self.frame00 = tk.Frame(parent,
                                width=300,
                                height=300,
                                relief='solid',
                                #borderwidth=1,
                                bg=parent.cget('bg'),
                                padx=10,
                                pady=10,
                                )
        self.frame00.grid(row=row, column=col, sticky=tk.E+tk.W)

        self.inner_frame00 = tk.LabelFrame(self.frame00,
                                           text='Select a file',
                                           padx=10,
                                           pady=10,
                                           fg='#000000',
                                           bg=self.frame00.cget('bg'),
                                           )
        self.inner_frame00.pack(side=tk.LEFT)

        # Create an entry field and buttons to get the file path.
        self.entry_path = tk.Entry(self.inner_frame00,
                                   width=20,
                                   fg='#000000',
                                   bg='#ffffff',
                                   )
        self.entry_path.insert(0, 'file path')

        self.btn_browse = tk.Button(self.inner_frame00,
                                    text='Browse...',
                                    fg='#000000',
                                    bg='#dddddd',
                                    )
        self.btn_browse.bind('<Button-1>', self.btn_browse_click)

        self.btn_load = tk.Button(self.inner_frame00,
                                    text='Load the file',
                                    fg='#000000',
                                    bg='#dddddd',
                                    )
        self.btn_load.bind('<Button-1>', self.btn_load_click)

        # Arrange each widget.
        self.entry_path.grid(row=0, column=0, sticky=tk.W)
        self.btn_browse.grid(row=0, column=1, sticky=tk.W, pady=10)
        self.btn_load.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)

    def make_frame01(self, parent, row, col):
        """Create a frame and generate widgets.
        Select signals, line width, line colors, and markers.
        """

        # Create a frame.
        self.frame01 = tk.Frame(parent,
                                width=300,
                                height=300,
                                relief='solid',
                                #borderwidth=1,
                                bg=parent.cget('bg'),
                                padx=10,
                                pady=10,
                                )
        self.frame01.grid(row=row, column=col, sticky=tk.E+tk.W)

        self.inner_frame01 = tk.LabelFrame(self.frame01,
                                           text='Select signals',
                                           padx=10,
                                           pady=10,
                                           fg='#000000',
                                           bg=self.frame01.cget('bg'),
                                           )
        self.inner_frame01.pack(side=tk.LEFT)

        # Create check buttons and comboboxes.
        self.num = 5  # The number of signals is 5 or less.
        self.color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.marker_list = ['', 'x', 'o', '^', 'D', 's', 'v', '*']
        self.line_list = ['', '-', '--', '-.', ':']
        self.signal_list = self.df_col
        self.signal_val_list = [tk.BooleanVar(value=False) for i in range(self.num)]
        self.signal_wid_list = []
        self.color_val_list = [tk.StringVar(value=self.color_list[i]) for i in range(self.num)]
        self.color_wid_list = []
        self.marker_val_list = [tk.StringVar(value=self.marker_list[0]) for i in range(self.num)]
        self.marker_wid_list = []
        self.line_val_list = [tk.StringVar(value=self.line_list[0]) for i in range(self.num)]
        self.line_wid_list = []
        #self.width_val_list = [tk.DoubleVar(value=1.0) for i in range(self.num)]
        #self.width_wid_list = []
        for i in range(self.num):
            if i < len(self.signal_list):
                self.signal = self.signal_list[i]
            else:
                self.signal = 'no signal'
            self.signal_wid_list.append(tk.Checkbutton(self.inner_frame01,
                                                       variable=self.signal_val_list[i],
                                                       text=self.signal,
                                                       fg='#000000',
                                                       bg=self.inner_frame01.cget('bg'),
                                                       )
                                       )
            self.color_wid_list.append(ttk.Combobox(self.inner_frame01,
                                                    textvariable=self.color_val_list[i],
                                                    values=self.color_list,
                                                    height=3,
                                                    width=2,
                                                    state='readonly',
                                                    )
                                      )
            self.marker_wid_list.append(ttk.Combobox(self.inner_frame01,
                                                     textvariable=self.marker_val_list[i],
                                                     values=self.marker_list,
                                                     height=3,
                                                     width=2,
                                                     state='readonly',
                                                     )
                                       )
            self.line_wid_list.append(ttk.Combobox(self.inner_frame01,
                                                   textvariable=self.line_val_list[i],
                                                   values=self.line_list,
                                                   height=3,
                                                   width=2,
                                                   state='readonly',
                                                   )
                                     )
            #self.width_wid_list.append(tk.Spinbox(self.inner_frame01,
            #                                      textvariable=self.width_val_list[i],
            #                                      format='%.1f',
            #                                      from_=0,
            #                                      to=5,
            #                                      increment=0.1,
            #                                      width=5,
            #                                      state='readonly',
            #                                      )
            #                          )

        # Create and put labels.
        self.item_list = ['Color', 'Marker', 'Line']
        for i, item in enumerate(self.item_list):
            self.lbl_item = tk.Label(self.inner_frame01,
                                     text=item,
                                     fg='#000000',
                                     bg=self.inner_frame01.cget('bg'),
                                     )
            self.lbl_item.grid(row=0, column=i+1)

        # Arrange each widget.
        for i in range(self.num):
            self.signal_wid_list[i].grid(row=i+1, column=0, sticky=tk.W, pady=5)
            self.color_wid_list[i].grid(row=i+1, column=1, sticky=tk.W)
            self.marker_wid_list[i].grid(row=i+1, column=2, sticky=tk.W, padx=5)
            self.line_wid_list[i].grid(row=i+1, column=3, sticky=tk.W)
            #self.width_wid_list[i].grid(row=i+1, column=4, sticky=tk.W)

    def make_frame02(self, parent, row, col):
        """Create a frame and generate widgets.
        Select parameters of axes.
        """

        # Create a frame.
        self.frame02 = tk.Frame(parent,
                                width=300,
                                height=300,
                                relief='solid',
                                #borderwidth=1,
                                bg=parent.cget('bg'),
                                padx=10,
                                pady=10,
                                )
        self.frame02.grid(row=row, column=col, sticky=tk.E+tk.W)

        self.inner_frame02 = tk.LabelFrame(self.frame02,
                                           text='Set axes',
                                           padx=10,
                                           pady=10,
                                           fg='#000000',
                                           bg=self.frame02.cget('bg'),
                                           )
        self.inner_frame02.pack(side=tk.LEFT)

        self.range_val = tk.StringVar(value='auto')
        self.range_auto = tk.Radiobutton(self.inner_frame02,
                                         text='Auto',
                                         value='auto',
                                         variable=self.range_val,
                                         fg='#000000',
                                         bg=self.inner_frame02.cget('bg'),
                                         )
        self.range_manu = tk.Radiobutton(self.inner_frame02,
                                         text='Manual',
                                         value='manual',
                                         variable=self.range_val,
                                         fg='#000000',
                                         bg=self.inner_frame02.cget('bg'),
                                         )

        self.lbl_x = tk.Label(self.inner_frame02,
                              text='x axis',
                              fg='#000000',
                              bg=self.inner_frame02.cget('bg'),
                              )
        self.entry_x_min = tk.Entry(self.inner_frame02,
                                    width=10,
                                    fg='#000000',
                                    bg='#ffffff',
                                    )
        self.entry_x_max = tk.Entry(self.inner_frame02,
                                    width=10,
                                    fg='#000000',
                                    bg='#ffffff',
                                    )
        self.lbl_y = tk.Label(self.inner_frame02,
                              text='y axis',
                              fg='#000000',
                              bg=self.inner_frame02.cget('bg'),
                              )
        self.entry_y_min = tk.Entry(self.inner_frame02,
                                    width=10,
                                    fg='#000000',
                                    bg='#ffffff',
                                    )
        self.entry_y_max = tk.Entry(self.inner_frame02,
                                    width=10,
                                    fg='#000000',
                                    bg='#ffffff',
                                    )
        self.entry_x_min.insert(0, np.amin(self.df.index.values))
        self.entry_x_max.insert(0, np.amax(self.df.index.values))
        self.entry_y_min.insert(0, np.amin(self.df.values))
        self.entry_y_max.insert(0, np.amax(self.df.values))

        self.grid_val = tk.BooleanVar(value=False)
        self.grid_wid = tk.Checkbutton(self.inner_frame02,
                                       variable=self.grid_val,
                                       text='Show grid',
                                       fg='#000000',
                                       bg=self.inner_frame02.cget('bg'),
                                       )

        # Arrange each widget.
        self.range_auto.grid(row=0, column=0, sticky=tk.W)
        self.range_manu.grid(row=1, column=0, sticky=tk.W)
        self.lbl_x.grid(row=2, column=0)
        self.entry_x_min.grid(row=2, column=1, padx=10)
        self.entry_x_max.grid(row=2, column=2)
        self.lbl_y.grid(row=3, column=0)
        self.entry_y_min.grid(row=3, column=1, pady=3)
        self.entry_y_max.grid(row=3, column=2)
        self.grid_wid.grid(row=4, column=0)

    def btn_browse_click(self, event):
        """A callback function.
        Get the file path from the filedialog.
        Input the file path to the entry.
        """

        # Get a file path.
        self.file_type = [('csv', '*.csv'), ('text', '*.txt')]
        self.dir_path = os.path.abspath(os.path.dirname(__file__))
        self.file_path = filedialog.askopenfilename(filetypes=self.file_type,
                                                    initialdir=self.dir_path,
                                                    )
        # Input the file path in the entry.
        if len(self.file_path) > 0:
            self.entry_path.delete(0, 'end')
            self.entry_path.insert(0, self.file_path)

    def btn_load_click(self, event):
        """A callback function.
        Load the file in the entry.
        Delete the frame and redraw it.
        """

        if os.path.exists(self.entry_path.get()):
            # Load the file.
            self.enc = 'utf-8'
            self.df = pd.read_csv(self.entry_path.get(),
                                  sep='\t',
                                  encoding=self.enc,
                                  )
            self.df = self.df.set_index('time')
            self.df_col = self.df.columns.values
            # Delete and redraw frames with the new data.
            self.frame01.destroy()
            self.make_frame01(self.parent, 1, 0)
            self.frame02.destroy()
            self.make_frame02(self.parent, 2, 0)
        else:
            # Error message.
            tk.messagebox.showerror('No files',
                                    'Please input a correct file path.',
                                    )

    def btn_draw_click(self, event):
        """A callback function.
        Get the status of check buttons.
        Get colors and markers from comboboxes.
        Show a graph.
        """

        self.flag_list = [i.get() for i in self.signal_val_list]
        self.sel_signal_list = []
        self.sel_color_list = []
        self.sel_marker_list = []
        self.sel_line_list = []
        for i, flg in enumerate(self.flag_list):
            if flg:
                self.sel_signal_list.append(self.signal_list[i])
                self.sel_color_list.append(self.color_val_list[i].get())
                self.sel_marker_list.append(self.marker_val_list[i].get())
                self.sel_line_list.append(self.line_val_list[i].get())

        print('\n''Informations for drawing the graph.')
        print('Signals:', self.sel_signal_list)
        print('Colors:', self.sel_color_list)
        print('Markers:', self.sel_marker_list)
        print('Lines:', self.sel_line_list)

        # Extract signals from df and show a graph.
        if self.sel_signal_list:  # len(self.sel_signal_list) > 0
            self.draw_graph()

    def draw_graph(self):
        """Set the graph style.
        Draw a graph.
        """

        # Get styles. '[color][marker][line]'
        self.style_list = zip(self.sel_color_list,
                              self.sel_marker_list,
                              self.sel_line_list,
                              )
        self.style_list = [i+j+k for i, j, k in self.style_list]

        # Extract selected signals.
        self.df_graph = self.df[self.sel_signal_list]

        # Draw a graph.
        if self.range_val.get() == 'auto':
            self.df_graph.plot(style=self.style_list, grid=self.grid_val.get())
        else:
            # Get x and y ranges.
            self.x_lim = [float(self.entry_x_min.get()),
                          float(self.entry_x_max.get()),
                          ]
            self.y_lim = [float(self.entry_y_min.get()),
                          float(self.entry_y_max.get()),
                          ]
            self.df_graph.plot(style=self.style_list,
                               xlim=self.x_lim,
                               ylim=self.y_lim,
                               )
        plt.show()


if __name__ == '__main__':
    main()
