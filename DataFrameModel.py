# -*- coding: utf-8 -*-

"""This module contains 2 dataframe models dedicated to unique display of
tables inside the GUI.

This module contains 2 different classes. The first one TableModel_1 enables
an usual display of tables with text data with a green background and numeric data in
yellow. The second one TableModel_2 is dedicated to germination percent data
with heatmap gradients making easier to read these data.

:Classes: TableModel_1

    The main model for displaying tables inside de GUI. It is used to display raw germination data, germination parameters and statistics.

:Classes: TableModel_2

    A second model for displaying table with an heatmap style to represent
    percent data in a clearer way.


:Dependencies:

    :PyQt5:	5.15.1
    :numpy:	1.19.2
    :matplotlib: 3.3.3
"""

# Packages imports

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import matplotlib as mpl
import numpy as np


# first class definition

class TableModel_1(QtCore.QAbstractTableModel):

    """TableModel_1 for usual display
    """

    def __init__(self, data):
        super(TableModel_1, self).__init__()
        self._data = data

    def data(self, index, role):

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

        if role == Qt.TextAlignmentRole:
            value = self._data.iloc[index.row(), index.column()]
            return Qt.AlignHCenter + Qt.AlignVCenter

        if role == Qt.BackgroundRole:
            value = self._data.iloc[index.row(), index.column()]

            if isinstance(value, str):
                return QtGui.QColor("YellowGreen")

            elif isinstance(value, float):
                return QtGui.QColor("Gold")

            else :
                return QtGui.QColor("Gold")

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


# second class definition

class TableModel_2(QtCore.QAbstractTableModel):

    """TableModel_2 for an heatmap display    
    """

    def __init__(self, data):
        super(TableModel_2, self).__init__()
        self._data = data

    def colorFader(self, c1, c2, mix=0):
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    def data(self, index, role):

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

        if role == Qt.TextAlignmentRole:
            value = self._data.iloc[index.row(), index.column()]
            return Qt.AlignHCenter + Qt.AlignVCenter

        if role == Qt.BackgroundRole:
            value = self._data.iloc[index.row(), index.column()]
            c1 = '#fcfc6f'  # yellow
            c2 = '#ff0000'  # red
            n = 100

            colors = []
            for i in range(n+1):
                color = self.colorFader(c1,c2,i/n)
                colors.append(color)

            if (isinstance(value, int) or isinstance(value, float)):
                value = int(value.round(0))
                return QtGui.QColor(colors[value])

            else:
                return QtGui.QColor("YellowGreen")

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])