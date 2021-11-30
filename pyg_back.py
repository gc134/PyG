# -*- coding: utf-8 -*-

"""This is the main module of the tool that holds all the functionalities for the
GUI.

This module contains all the functions and signal-slots methods for GUI
navigation, data loading and processing, graphical representation, fitting
and statistical analysis.

:Classes: MyMainWindow

    The main class of the project that create the GUI with all the
    functionalities

:Dependencies:

    GUI

    :PyQt5:	5.15.1
    :PyQt5sip:	12.8.1
    :PyQt5stubs:	5.14.2.2
    :PyQt5Designer:	5.14.1
    :PyQtWebEngine:	5.15.2
    :PyQtWebKit:	5.13.1
    :pyqt5guiyyj:	0.1.12
    :pyqt5plugins:	5.15.1.2.0.1
    :pyqt5tools:	5.15.1.3
    :pyqtgraph:	0.11.0
    :qt5applications:	5.15.1.2
    :qt5tools:	5.15.1.1.0.1

    Data handling

    :pandas:	1.1.4
    :pandasflavor:	0.2.0

    Computing

    :numpy:	1.19.2
    :scipy:	1.5.4

    Graphics

    :matplotlib:	3.3.3
    :seaborn:	0.11.0

    Statistics

    :pingouin:	0.3.8
    :statannot:	0.2.3

    Clipboard management

    :pyperclip:	1.8.2
"""


# Packages and modules imports

import sys
import os

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QDir, Qt, QFile, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad

import seaborn as sns

import pingouin as pg
import statannot as sta

import io
import pyperclip

from pyg_ui import Ui_MainWindow
from DataFrameModel import TableModel_1, TableModel_2
from styleSheet_dark_orange import styleSheet
import pyg_ressources_rc


# Main class definition

class MyMainWindow(QMainWindow, Ui_MainWindow):

    """The central class of the project that holds all functionalities and
    distributes them across the GUI

    This MyMainWindow class inherits from QMainWindow (PyQt5) and
    Ui_MainWindow (pyg_ui.py) classes. It holds every features from GUI
    navigation to the last step of statistical analysis.
    """

    # Constructor
    def __init__(self, parent=None):
        """Constructor method

        :param parent: A parental class to define
        :type parent: class
        """
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setStyleSheet(styleSheet)


    # A function that helps generate error message

    def error_gc(self, text1, text2):
        """Error message generator

        Two parameters have to be given. A first string that provides the
        nature of error and a second that may guide to a solution

        :param text1: Error type
        :type text1: str
        :param text2: An advice
        :type text2: str
        :return: An error message box
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setWindowTitle("Error")
        msg.exec_()


    # A function that helps generate information message

    def information_gc(self, text1, text2):
        """Warning message generator

        Two parameters have to be given. A first string that provides the
        primary information and a second to possibly complete the message

        :param text1: Primary information to provide
        :type text1: str
        :param text2: Additional information to provide
        :type text2: str
        :return: A informative message box
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text1)
        msg.setInformativeText(text2)
        msg.setWindowTitle("Information")
        msg.exec_()


    # Signal-slots group to switch betweeen GUI pages by clicking the
    # respective buttons from the left panel

    @pyqtSlot()
    def on_pushButton_1_clicked(self):
        """Move to page1
        """
        self.stackedWidget.setCurrentIndex(0)


    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """Move to page2
        """
        self.stackedWidget.setCurrentIndex(1)

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        """Move to page3
        """
        self.stackedWidget.setCurrentIndex(2)

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        """Move to page4
        """
        self.stackedWidget.setCurrentIndex(3)

    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        """Move to page5
        """
        self.stackedWidget.setCurrentIndex(4)

    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        """Move to page6
        """
        self.stackedWidget.setCurrentIndex(5)

    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        """Move to page7
        """
        self.stackedWidget.setCurrentIndex(6)


    # page1 methods

    # working directory definition

    @pyqtSlot()
    def on_dir_page1_clicked(self):
        """Working directory selection

        Open a window to select a working directory, by clicking dir_page1
        push button.

        :return: A path to the selected working directory
        """
        self.mydirectory = QFileDialog.getExistingDirectory(self, 'Select directory')
        mpl.rcParams["savefig.directory"] = self.mydirectory
        self.textEdit1_page1.setText(self.mydirectory)
        self.textEdit1_page1.setAlignment(Qt.AlignHCenter)

    # data import, display and QC

    @pyqtSlot()
    def on_import_page1_clicked(self):
        """Data import, display and QC

        Clicking the import_page1 push button opens a window to select a file of raw counts to analyze
        . The intended format is csv. A file is loaded into session, its content is displayed in a table
        view and summary statistics are calculated.

        :param mydirectory: A path
        :type mydirectory: str
        :param file_name: A .csv file
        :type file_name: str

        :return: A pandas dataframe containing the raw data (first level of
            data)
        :return: QC data
        """
        try :
            self.mydirectory

        except :
            self.file_name= QFileDialog.getOpenFileName(self, caption="Choose a file")
            self.mydirectory = os.path.dirname(self.file_name[0])
            mpl.rcParams["savefig.directory"] = self.mydirectory


        else :

            self.file_name = QFileDialog.getOpenFileName(self, caption="Choose a file", directory=self.mydirectory)

        try :

            self.df_raw = pd.read_csv(self.file_name[0], sep=";")

        except :

            self.error_gc("Problème de fichier", "Veuillez sélectionner un autre fichier")
            pass

        else :

            self.textEdit2_page1.setText(self.file_name[0])
            self.textEdit2_page1.setAlignment(Qt.AlignHCenter)

            model_df_raw = TableModel_1(self.df_raw)
            self.tableView_page1.setModel(model_df_raw)
            self.tableView_page1.installEventFilter(self)

            # QC data : several variables calculation with formatting output

            try :

                row_count = self.df_raw.shape[0]
                column_count = self.df_raw.shape[1]
                timepoint_count = column_count - 3
                timepoint = ", ".join(list(self.df_raw.columns[3:column_count]))
                nb_levels = len(self.df_raw["groupe"].unique())
                self.group_levels =", ".join(list(self.df_raw["groupe"].unique()))
                NA_count = self.df_raw.isna().sum().sum()
                NA_percent = ((NA_count/(row_count*timepoint_count))*100).round(1)
                zero_count = self.df_raw.isin([0]).sum().sum()
                zero_percent = ((zero_count/(row_count*timepoint_count))*100).round(1)
                min_total = self.df_raw["total"].min()
                max_total = self.df_raw["total"].max()
                min_count = self.df_raw.iloc[:,3:column_count].min().min()
                max_count = self.df_raw.iloc[:,3:column_count].max().max()

                qc_text = "- dimensions : {} lignes x {} colonnes\n\
                - {} points temporels : {}\n\
                - {} niveaux : {}\n\
                - nb de NA (%) : {} ({}%)\n\
                - nb de zéros (%) : {} ({}%)\n\
                - intervalle nb initial de graines : {} - {}\n\
                - intervalle nb de graines germées : {} - {}" \
                    .format(row_count, column_count, timepoint_count, \
                            timepoint,nb_levels, self.group_levels, NA_count, NA_percent, \
                            zero_count, zero_percent, min_total, \
                            max_total, min_count, max_count)

            except :

               self.error_gc("Impossibilité de calculer tous les paramètres de QC", "Veuillez vérifier le contenu du tableau de données")

            else :
                self.textEdit3_page1.clear()
                self.textEdit3_page1.setAlignment(Qt.AlignHCenter)
                self.textEdit3_page1.append(qc_text)
                self.textEdit3_page1.setAlignment(Qt.AlignHCenter)

                self.levels_order= list(self.df_raw["groupe"].unique())


    # groups order definition for the boxplots

    @pyqtSlot()
    def on_validation_page1_clicked(self):
        """Groups ordering

        The group levels order is entered in textEdit4_page1 widget. The
        levels have to be separated by a return. The ordered list is created
        once validation_page1 push button is pressed.

        :param levels_order: Custom group levels sorting for following boxplots
        :type levels_order: list

        :return: An ordered list of levels
        """
        self.levels_order = list(self.textEdit4_page1.toPlainText().split("\n"))

        if sorted(self.levels_order) == list(self.df_raw["groupe"].unique()) :
            self.information_gc("Niveaux ordonnés","")
        else :
            self.error_gc("Les niveaux saisis ne correspondent pas à ceux du tableau",
                          "Veuillez corriger votre saisie")
            self.levels_order = list(self.df_raw["groupe"].unique())


    # page2 methods

    # germination percents calculation

    @pyqtSlot()
    def on_calcul_page2_clicked(self):
        """Germination percents calculation

        Generate the germination percent matrix from the raw counts data,
        on calcul_page2 push. The percent data are displayed in
        tableView_page2 output, in a heatmap style.

        :param df_raw: The first level data of raw germination counts
        :type df_raw: pandas.Dataframe
        :return: A pandas dataframe containing the percent data required for
            the next steps of analysis
        :return: The time serie of the germination test
        """
        try :
                self.df_count = self.df_raw.drop(['echantillon', 'groupe', 'total'], 1)

                self.total= self.df_raw['total']

                self.df_percent = pd.DataFrame()

                for i in range(0, self.df_count.shape[0]):
                    vec = self.df_count.iloc[i, :]
                    vec = round((vec / self.total[i]) * 100, 1)
                    vec = pd.DataFrame(vec).transpose()
                    self.df_percent = pd.concat([self.df_percent, vec])

                self.df_percent_final = pd.concat([self.df_raw[["echantillon","groupe"]],self.df_percent], axis=1)

        except :

               self.error_gc("Le calcul des pourcentages de germination est impossible","Vérifier le bon chargement et la qualité des données à la page précédente")

        else :
                model_df_percent_final = TableModel_2(self.df_percent_final)
                self.tableView_page2.setModel(model_df_percent_final)
                self.tableView_page2.installEventFilter(self)

                # extraction de la série temporelle d'observations à partir des entêtes de colonnes

                self.time_serie = list(map(float, list(self.df_count.columns)))


    # germination percents matrix export

    @pyqtSlot()
    def on_export_page2_clicked(self):
        """Germination percents export

        Open a dialog to save a file, once export_page2 pushed.

        :param df_percent_final: The germination percents matrix with samples metadata
        :type df_percent_final: pandas.Dataframe

        :return: A .csv file
        """
        try :
            self.df_percent_final

        except :

           self.error_gc("L'export est impossible","Veuillez lancer le calcul des pourcentages de germination")

        else:

            name = QFileDialog.getSaveFileName(self, 'Save into a file',self.mydirectory)
            self.df_percent_final.to_csv(name[0] + ".csv", sep=";", decimal=".", index=False)


    # page3 methods

    # A function to integrate tiff files to label widgets

    def image_display_gc(self,file,object):
        """Display an image into a widget

        Here a saved tiff file is used to generate an image object that is
        tranformed before being displayed in a specific GUI widget

        :param file: A tiff file in the working directory
        :type file: str
        :param object: A display widget in the GUI
        :type object: PyQt5 widget
        :return: A display
        """
        image = QPixmap(file)
        image = image.scaled(450, 500, Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
        object.setPixmap(image)


    # individual germination curves display

    @pyqtSlot()
    def on_courbe_ind_page3_clicked(self):
        """Display individual germination curves

        From the percents matrix, germination curves are plotted for each
        seeds sample, for the duration of the test. Matplotlib windows are
        used to display the graph and a miniature inside the GUI keeps track
        of it.

        :param df_percent: The restricted germination percents matrix
        :type df_percent: pandas.Dataframe
        :param time_serie: The temporal sequence of the germination test
        :type time_serie: list
        :param textEdit1_page3: A plot title
        :type textEdit1_page3: str
        :return: A matplotlib graph in a external window
        :return: A thumbnail of the matplotlib plot
        """

        try :
            curves=[]

            for i in range(0, self.df_percent.transpose().shape[1]):
                x = np.array(self.time_serie)
                y = self.df_percent.transpose().iloc[:, i]
                curves += plt.plot(x, y)

        except :

           self.error_gc("Le calcul des courbes de germination est impossible", "Vérifiez que le calcul des pourcentages de germination à la page précédente est correct")

        else :

            plt.title(self.textEdit1_page3.toPlainText())
            plt.xlabel("temps")
            plt.ylabel("% germination")
            plt.legend(curves, self.df_percent_final['echantillon'], loc=0)
            plt.savefig(self.mydirectory + "courbes_germ_indiv.tiff")
            plt.show()
            plt.figure()

            self.image_display_gc(file= self.mydirectory + "courbes_germ_indiv.tiff", object= self.label1_page3)


    # grouped germination curves display

    @pyqtSlot()
    def on_courbe_groupe_page3_clicked(self):
        """Grouped germination curves display

        The group means and standard deviation is used to plot germination
        curves by group of replicates. Matplotlib windows are
        used to display the graph and a miniature inside the GUI keeps track
        of it.

        :param df_percent_final: The germination percents matrix with samples metadata
        :type df_percent_final: pandas.Dataframe
        :param textEdit2_page3: A plot title
        :type textEdit2_page3: str
        :return: A matplotlib graph in a external window
        :return: A thumbnail of the matplotlib plot
        """

        try :

            self.df_percent_mean = pd.DataFrame()

            for name in self.df_percent_final.columns[2:]:
                vec = self.df_percent_final.groupby('groupe')[name].mean().values
                vec = pd.DataFrame(vec)
                self.df_percent_mean = pd.concat([self.df_percent_mean, vec], axis=1, ignore_index=True)

            self.df_percent_sd = pd.DataFrame()

            for name in self.df_percent_final.columns[2:]:
                vec = self.df_percent_final.groupby('groupe')[name].std().values
                vec = pd.DataFrame(vec)
                self.df_percent_sd = pd.concat([self.df_percent_sd, vec], axis=1, ignore_index=True)

            curves=[]

            for i in range(0, self.df_percent_mean.shape[0]):
                legend_text = self.df_percent_final['groupe'].unique().tolist()[i]
                curves += plt.errorbar(np.array(self.time_serie), self.df_percent_mean.iloc[i, :], \
                                       self.df_percent_sd.iloc[i, :], linestyle='solid', \
                                       marker='.', label="{}".format(legend_text))

        except :

           self.error_gc("Le calcul des courbes de germination est impossible",
                          "Vérifiez que le calcul des pourcentages de germination à la page précédente est correct")

        else :

            plt.title(self.textEdit2_page3.toPlainText())
            plt.xlabel("temps")
            plt.ylabel("% germination")
            plt.legend(loc=0)
            plt.savefig(self.mydirectory + "courbes_germ_groupes.tiff")
            plt.show()
            plt.figure()

            self.image_display_gc(file=self.mydirectory + "courbes_germ_groupes.tiff", object=self.label2_page3)

    # page4 methods

    # fitting functions

    def hill_function(self,x, a, b, c, yo):
        """The 4 parameters Hill function (4PHF)

        This is the mathematical model best fit to adjust experimental
        germination sigmoid curves.

        :param x: Time
        :type x: float
        :param a: Final level
        :type a: float
        :param b: Shape of the curve
        :type b: float
        :param c: Inflection point of the sigmoid curve
        :type c: float
        :param yo: Initial level
        :type yo: float
        :return: y (germination level in our case)
        """
        return yo + (a * np.power(x, b) / (np.power(c, b) + np.power(x, b)))


    def optimized_hill_function(self,x):
        """The Hill function with extracted parameters from fitting

        This function is used to compute the estimated germination levels (
        percents) along time from an optimized model that fits the
        experimental data

        :param x: Time
        :type x: float
        :param fittedParameters: Fitting parameters
        :type fittedParameters: float
        :return: estimation of y
        """
        return self.fittedParameters[3] + (self.fittedParameters[0] * np.power(x, self.fittedParameters[1]) / (
                np.power(self.fittedParameters[2], self.fittedParameters[1]) + np.power(x, self.fittedParameters[1])))


    # experimental data fitting step

    @pyqtSlot()
    def on_calcul_page4_clicked(self):
        """Fitting step

        Every germination curve will be adjusted with the 4 parameters Hill
        function, with a parameters calculation that starts from initial
        entered values to final optimized ones, once they converged. These
        final parameters values give a model that best fit the experimental
        data from which it will be possible to extract germination parameters.

        :param textEdit1_page4: initial value for a, overall final germination
            percent
        :type textEdit1_page4: float
        :param textEdit2_page4: initial value for b, curves shape
        :type textEdit2_page4: float
        :param textEdit3_page4: initial value for c, overall time to reach
            50% of germination
        :type textEdit3_page4: float
        :param textEdit4_page4: initial value for y0, germination percent at
            experiment start
        :type textEdit4_page4: float
        :param textEdit5_page4: Time span
        :type textEdit5_page4: float
        :param df_percent: The restricted germination percent matrix
        :type df_percent: pandas.Dataframe
        :param time_serie: The temporal sequence of the germination test
        :type time_serie: list
        :return: The dataframe of fitted parameters for the all seeds samples
        :return: A thumbnail of the matplotlib graph of adjustments
        """

        try :

            self.xmax = float(self.textEdit5_page4.toPlainText())

            x_interpol = np.linspace(0, self.xmax, 100)

            self.initialParameters = np.array([float(self.textEdit1_page4.toPlainText()),
                                               float(self.textEdit2_page4.toPlainText()),
                                               float(self.textEdit3_page4.toPlainText()),
                                               float(self.textEdit4_page4.toPlainText())])

            # plutôt que d'afficher simplement les données y d'ajustement pour les points de temps d'observation avec des cassures
            # on préfère représenter des courbes lissées = par interpolation

            self.adj_curves = []
            self.df_fittedParameters= pd.DataFrame()

            for i in range(0, self.df_percent.shape[0]):
                yData = np.array(self.df_percent.iloc[i, :])
                fittedParameters, pcov = curve_fit(self.hill_function,
                                                   self.time_serie, yData,
                                                   self.initialParameters)
                f = interp1d(self.time_serie, self.hill_function(
                    self.time_serie, *fittedParameters), kind="slinear")
                y_fit= f(x_interpol)
                self.adj_curves += plt.plot(x_interpol, y_fit)

                fittedParameters_vec = pd.DataFrame(fittedParameters).transpose()
                self.df_fittedParameters = pd.concat([self.df_fittedParameters, fittedParameters_vec], \
                                                     axis=0, ignore_index=True)
        except :

            self.error_gc("Les données sont insuffisantes pour l'ajustement de toutes les courbes de germination",
                          "Vérifiez que les pourcentages de germination ont été calculés\nRéessayer avec des données plus complètes\nAffiner les valeurs initiales des paramètres (a,b,c,yo,intervalle de temps) ")
            del self.adj_curves
            del self.df_fittedParameters


        else:

            plt.title("ajustements")
            plt.xlabel("temps")
            plt.ylabel("% germination")
            plt.legend(self.adj_curves, self.df_percent_final['echantillon'], loc=0)
            plt.savefig(self.mydirectory + "courbes_ajustement.tiff")

            self.image_display_gc(file=self.mydirectory + "courbes_ajustement.tiff", object=  self.label6_page4)


    # Adjustement curves in matplotlib

    @pyqtSlot()
    def on_courbe_page4_clicked(self):
        """Adjustement curves in a matplotlib window

        Display a window for the matplotlib graph of germination curves
        adjustement, with a linear interpolation. For the graph to be
        possibly saved.

        :param adj_curves: The adjustement curves basic plot
        :type adj_curves: matplotlib object
        :return: A matplotlib graph in a external window
        """

        try :

            self.adj_curves

        except :

           self.error_gc("Impossible de générer le graphe d'ajustements","")

        else:

            plt.title("ajustements")
            plt.xlabel("temps")
            plt.ylabel("% germination")
            plt.legend(self.adj_curves, self.df_percent_final['echantillon'], loc=0)
            plt.show()
            plt.figure()


    # Germination parameters extraction from the fitted parameters

    @pyqtSlot()
    def on_param_ind_page4_clicked(self):
        """Individual germination parameters calculation

        The germination parameters will be extracted from the adjustement model
        fitted parameters.

        :param df_fittedParameters: The dataframe of fitted parameters by
            seeds sample
        :type df_fittedParameters: pandas.Dataframe
        :param self.xmax: The time span of experiment
        :type self.xmax: float
        :return: A daframe of the 5 germination parameters (Gmax,t1,t50,D,
            AUC) for every seeds sample
        """
        try :

            self.df_fittedParameters

        except:

           self.error_gc("Ajustement impossible avec ces données","Pas de paramètres de germination à extraire")

        else :

            try :
                self.df_germ_parameters = pd.DataFrame()

                for i in range(0, self.df_percent_final.shape[0]):

                    self.Gmax, self.t50= round(self.df_fittedParameters.iloc[i,0], 2), \
                                         round(self.df_fittedParameters.iloc[i,2], 2)

                    self.xmin = 0

                    self.fittedParameters = self.df_fittedParameters.iloc[i,].values

                    self.integral_res, self.integral_err = quad(self.optimized_hill_function, self.xmin, self.xmax)

                    self.lag = round(np.power(((-self.df_fittedParameters.iloc[i,3] \
                                                *np.power(self.df_fittedParameters.iloc[i,2], \
                                                          self.df_fittedParameters.iloc[i,1])) \
                                               /(self.df_fittedParameters.iloc[i,0]+self.df_fittedParameters.iloc[i,3])), \
                                              1/self.df_fittedParameters.iloc[i,1]),2)

                    self.D = round(self.t50 - self.lag,2)

                    self.germ_parameters = [self.Gmax, self.lag, self.t50, self.D, round(self.integral_res,2)]

                    self.germ_parameters = pd.DataFrame(self.germ_parameters).transpose()

                    self.df_germ_parameters = self.df_germ_parameters.append(self.germ_parameters, ignore_index=True)

                self.df_germ_parameters.columns = ['Gmax','lag','t50', 'D', 'AUC']

                self.df_germ_parameters = pd.concat([self.df_percent_final['echantillon'], \
                                                     self.df_percent_final["groupe"],self.df_germ_parameters],axis=1)

            except :

               self.error_gc("Impossibilité d'extraire des paramètres de germination après ajustement","")

            else:

                self.model_indiv_germ = TableModel_1(self.df_germ_parameters)
                self.tableView_page4.setModel(self.model_indiv_germ)
                self.tableView_page4.installEventFilter(self)

    # Germination parameters by group

    @pyqtSlot()
    def on_param_groupe_page4_clicked(self):
        """Germination parameters by group

        Group means are calculated for every parameter to facilitate groups
        comparison.

        :param df_germ_parameters: The germination parameters data for every
            sample
        :type df_germ_parameters: pandas.Dataframe
        :return: The mean germination parameters for every group
        """
        try :

            self.df_germ_parameters

        except :

           self.error_gc("Impossible de calculer les paramètres moyennés","Veuillez lancer préalablement le calcul des paramètres individuels")

        else:

            try :

                self.df_germ_parameters_mean=pd. DataFrame()

                for name in self.df_germ_parameters.columns[2:]:
                    vec = self.df_germ_parameters.groupby('groupe')[name].mean().values
                    vec = (round(num,2) for num in vec)
                    vec = pd.DataFrame(vec)
                    self.df_germ_parameters_mean = pd.concat([self.df_germ_parameters_mean, vec], axis=1, ignore_index=True)

                self.df_germ_parameters_mean.insert(0, 'groupe', self.df_percent_final['groupe'].unique().tolist())

                self.df_germ_parameters_mean.columns = ['groupe', 'Gmax','lag','t50', 'D', 'AUC']

            except :

               self.error_gc("Impossible de calculer les paramètres moyennés","")

            else:

                model_group_germ = TableModel_1(self.df_germ_parameters_mean)
                self.tableView_page4.setModel(model_group_germ)
                self.tableView_page4.installEventFilter(self)

    # Germination parameters tables export

    @pyqtSlot()
    def on_export_page4_clicked(self):
        """Germination parameters tables export

        Here we can save the data in .csv files

        :param mydirectory: A selected working directory
        :type mydirectory: str
        :return: A .csv file
        """
        try :

            self.df_germ_parameters

        except :

           self.error_gc("Export impossible","Aucun tableau de paramètres n'a été généré")


        else:

            name = QFileDialog.getSaveFileName(self, 'Save into a file', self.mydirectory)

            tableView_page4_model = self.tableView_page4.model()

            if tableView_page4_model is self.model_indiv_germ:
                self.df_germ_parameters.to_csv(name[0] + ".csv", sep=";", decimal=".", index=False)
            else:
                self.df_germ_parameters_mean.to_csv(name[0] + ".csv", sep=";", decimal=".", index=False)



    # page5 methods

    # function to generate a seaborn boxplot

    def boxplot_gc(self,y):
        """Boxplot for germination parameters

        This function helps create a boxplot for germination parameters,
        possibly with a group order set by the user

        :param y: A germination parameter
        :type y: str
        :param data: A dataset to analyze
        :type data: pandas.Dataframe
        :param order: An ordered list of group levels
        :type order: list
        :param boxplot_title: The main title we want for the plot
        :type boxplot_title: str
        :param mydirectory: The user defined working directory
        :type mydirectory: A path
        :return: A matplotlib boxplot
        :return: A .tiff file of the plot exported into the working directory
        """
        boxplot = sns.boxplot(x="groupe", y=y, data=self.df_germ_parameters, palette="tab20", linewidth=1, saturation=2, order =  self.levels_order)
        plt.title(self.boxplot_title)
        plt.ylabel(self.selected_param)
        plt.show()
        plt.figure()
        boxplot.get_figure().savefig(self.mydirectory + "boxplot_{}.tiff".format(self.selected_param))


    # minimal boxplots for germination parameters

    @pyqtSlot()
    def on_pushButton_page5_clicked(self):
        """Minimal boxplots for germination parameters

        By clicking pushButton_page5, boxplots are created for the parameter
        selected in comboBox_page5. A title can be added by filling
        textEdit_page5. The boxplots are displayed in matplotlib windows and
        a GUI thumbnail appears.

        :param comboBox_page5: A selected germination parameter
        :type comboBox_page5: str
        :param textEdit_page5: A boxplot title
        :type textEdit_page5: str
        :return: A boxplot in a matplotlib window
        :return: A boxplot thumbnail inside the GUI

        """

        # plt.clf() to remove any previous matplotlib residuals

        plt.clf()

        if self.comboBox_page5.currentIndex() == 0 :

           self.error_gc("Veuillez choisir un paramètre de germination","")

        else :

            # In order to limit the basic "whitegrid" style to the boxplot output
            # we use a "with" block

            with sns.axes_style("whitegrid"):

                try :

                    self.selected_param = self.comboBox_page5.currentText()

                    self.boxplot_title = self.textEdit_page5.toPlainText()

                    self.boxplot_gc(y=self.selected_param)

                except :

                   self.error_gc("Impossible de générer le boxplot demandé","Veuillez choisir un autre paramètre de germination\nLes valeurs de germination extraites ne suffisent pas aux boxplots ")

                else :

                    self.image_display_gc(file=self.mydirectory + "boxplot_{}.tiff".format(self.selected_param), object=self.label_page5)


    # page6 methods

    # anova analysis with pingouin tools

    @pyqtSlot()
    def on_anova_page6_clicked(self):
        """Anova analysis

        An usual group comparison is performed with the pingouin package
        anova method to better characterize the overall variation of every
        germination parameters.

        :param comboBox_page6: A selected gerrmination parameter
        :type comboBox_page6: str
        :param df_germ_parameters: The dataframe of individual germination
            parameters
        :type df_germ_parameters: pandas.Dataframe
        :return: A pandas dataframe of anova's outputs
        """

        if self.comboBox_page6.currentIndex() == 0 :

           self.error_gc("Veuillez choisir un paramètre de germination","")

        else :

            try :

                self.selected_param = self.comboBox_page6.currentText()

                self.aov = pg.anova(data=self.df_germ_parameters, dv=self.selected_param, between='groupe', detailed=True).round(6)

            except :

               self.error_gc("Impossibilité de faire une anova","Veuillez choisir un autre paramètre de germination\nDonnées insuffisantes")
            else :

                self.aov = pd.DataFrame(self.aov.T).transpose()[['Source', 'p-unc', 'F', 'SS', 'DF', "MS", "np2"]]

                self.model_aov = TableModel_1(self.aov)

                self.tableView1_page6.setModel(self.model_aov)

                self.tableView1_page6.installEventFilter(self)


    # multiple comparisons with pingouin tools (post-hoc tests)

    @pyqtSlot()
    def on_multcomp_page6_clicked(self):
        """Multiple comparisons (post-hoc tests)

        The pingouin pairwise_ttests function allows to compare the group
        levels pairwise, with a Benjamini Hochberg fdr correction of
        p-values. The results are used to better characterize the origins of
        the overall variations as seen in anova.

        :param comboBox_page6: A selected germination parameter
        :type comboBox_page6: str
        :param df_germ_parameters: The dataframe of individual germination
            parameters
        :type df_germ_parameters: pandas.Dataframe
        :return: A pandas dataframe with all the multiple comparisons results
        """

        if self.comboBox_page6.currentIndex() == 0 :

           self.error_gc("Veuillez choisir un paramètre de germination","")

        else :

            try :

                self.selected_param = self.comboBox_page6.currentText()

                self.posthoc = pg.pairwise_ttests(data=self.df_germ_parameters, dv=self.selected_param, between='groupe', parametric=True, padjust='fdr_bh',
                                                  effsize='hedges').round(6)
            except :

               self.error_gc("Impossible de faire les comparaisons multiples","Veuillez choisir un autre paramètre de germination\nDonnées insuffisantes")

            else :
                self.posthoc= pd.DataFrame(self.posthoc.T).transpose()[
                    ['Contrast', 'A', 'B', 'p-unc', 'p-corr', 'p-adjust', 'T', 'Paired', 'Parametric', 'dof', 'Tail',
                     'BF10', 'hedges']]

                self.model_posthoc = TableModel_1(self.posthoc)

                self.tableView1_page6.setModel(self.model_posthoc)

                self.tableView1_page6.installEventFilter(self)


    # statistical results export

    @pyqtSlot()
    def on_export_page6_clicked(self):
        """Export of statistical results in a csv file

        Depending of the displayed results in the GUI, either the anova or
        the post-hoc tests results are exported in a .csv format, into the
        working directory by default, or in a newly selected folder.

        :param mydirectory: A selected working directory
        :type mydirectory: str
        :return: A .csv file
        """

        try :

            if self.tableView1_page6.model() is None :
                pass
            else :
                name = QFileDialog.getSaveFileName(self, 'Save into a file', self.mydirectory)

            try :
                self.model_aov

            except :

                self.model_aov = None

            if self.tableView1_page6.model() is self.model_aov:
                self.aov.to_csv(name[0] + ".csv", sep=";", decimal=".", index=False)
            else:
                self.posthoc.to_csv(name[0] + ".csv", sep=";", decimal=".", index=False)

        except :

           self.error_gc("Export de résultats statistiques impossible","Veuillez lancer un test statistique")


    # anova p-values report

    @pyqtSlot()
    def on_bilan_page6_clicked(self):
        """Anova report with all the germination parameters

        Here we catch the anova p-values calculated on all germination
        parameters and report them into a single table.

        :param df_germ_parameters: The dataframe of individual germination
            parameters
        :type df_germ_parameters: pandas.Dataframe
        :return: A pandas dataframe summarizing all the anova p-values
        """

        try :

            self.df_param_pvalue = pd.DataFrame()

            for indiv_param in ('Gmax', 'lag', 't50', 'D', "AUC") :

                aov = pg.anova(data=self.df_germ_parameters, dv=indiv_param, between='groupe',
                               detailed=True).round(6)

                param_pvalue = pd.DataFrame([indiv_param, aov['p-unc'][0]]).transpose()

                self.df_param_pvalue = self.df_param_pvalue.append(param_pvalue, ignore_index=True)

        except :

           self.error_gc("Impossibilité de calculer les p-values d'anova des 5 paramètres","")

        else :

            self.df_param_pvalue.columns=['parameter','p-value']

            self.model_param_pvalues = TableModel_1(self.df_param_pvalue)

            self.tableView2_page6.setModel(self.model_param_pvalues)

            self.tableView2_page6.installEventFilter(self)


    # page7 methods

    # a function that generates a Tukey boxplot

    def t_boxplot_gc(self,y):
        """A function for Tukey boxplot (with pairwise comparisons results)

        With this function we can easily generate boxplots of germination
        parameters with an additionnal data from the pairwise post_hoc tests,
        namely the list of p-values from all comparisons. This gives an
        higher insight into the nature of variations.

        :param y: A germination parameter
        :type y: str
        :param data: The individual germination parameters table
        :type data: pandas.Dataframe
        :param order: A user-defined order for the group levels
        :type order: list
        :param boxplot_title: A main title for the plot
        :type boxplot_title: str
        :param groups_couple: A list of pairwise comparisons
        :type groups_couple: list
        :param posthoc: A dataframe containing the post-hoc tests results
        :type posthoc: pandas.Dataframe
        :param selected_param: A selected germination parameter
        :type selected_param: str
        :return: A Tukey boxplot in a matplotlib window
        :return: A .tiff file of the Tukey boxplot
        """
        boxplot = sns.boxplot(x="groupe", y=y, data=self.df_germ_parameters, palette="tab20", linewidth=1, saturation=2, order = self.levels_order)
        plt.title(self.boxplot_title)
        plt.ylabel(self.selected_param)
        test_results = sta.add_stat_annotation(boxplot, data=self.df_germ_parameters, x='groupe', y=y,
                                               box_pairs=self.groups_couple,
                                               test='t-test_ind',
                                               loc='outside', verbose=1, text_annot_custom= \
                                                   list(map('p={}'. format, self.posthoc['p-corr'])),
                                               perform_stat_test=True,
                                               show_test_name=True, line_offset_to_box=0.1,
                                               order = self.levels_order)
        plt.tight_layout()
        plt.show()
        plt.figure()
        boxplot.get_figure().savefig("t_boxplot_{}.tiff".format(self.selected_param))


    # Tukey boxplots on germination parameters

    @pyqtSlot()
    def on_pushButton_page7_clicked(self):
        """Tukey boxplots for germination parameters

        Here the Tukey boxplots with multiple comparison data will be
        created, for a selected parameter, possibly with an entered main
        title, by clicking pushButton_page7. The plot is first displayed in a
        matplotlib window but also as a thumbnail in the GUI.

        :param comboBox_page7: A selected germination parameter
        :type comboBox_page7: str
        :param df_germ_parameters: The individual germination parameters
            dataframe
        :type df_germ_parameters: pandas.Dataframe
        :return: A Tukey boxplot in a matplotlib window
        :return: A thumbnail of the plot inside the GUI
        """
        # plt.clf() to remove any previous matplotlib residuals

        plt.clf()

        if self.comboBox_page7.currentIndex() == 0:

           self.error_gc("Veuillez choisir un paramètre de germination", "")

        else:

            # In order to limit the basic "whitegrid" style to the boxplot output
            # we use a "with" block

            with sns.axes_style("whitegrid"):

                try :

                    self.selected_param = self.comboBox_page7.currentText()

                    self.posthoc = pg.pairwise_ttests(data=self.df_germ_parameters, dv=self.selected_param,
                                                      between='groupe', parametric=True, padjust='fdr_bh',
                                                      effsize='hedges').round(6)

                    self.posthoc= pd.DataFrame(self.posthoc.T).transpose()

                    self.groups_couple= [tuple(val) for val in self.posthoc[['A','B']].values.tolist()]

                    self.boxplot_title = self.textEdit_page7.toPlainText()

                    self.t_boxplot_gc(y=self.selected_param)

                except :

                   self.error_gc("Impossible de générer le boxplot de Tukey","Veuillez sélectionner un autre paramètre de germination\nDonnées insuffisantes")

                else :
                    self.image_display_gc(file="t_boxplot_{}.tiff".format(self.selected_param), object=self.label_page7)



    # eventFilter method combined with copySelection to activate a copy-paste
    # feature in the GUI for the tableView widgets

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
                event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection(source)
        return True
        return super(MyMainWindow, self).eventFilter(source, event)

    def copySelection(self, table):
        clipboardString = io.StringIO()
        selectedIndexes = table.selectedIndexes()
        if selectedIndexes:
            countList = len(selectedIndexes)
            for r in range(countList):
                current = selectedIndexes[r]
                displayText = current.data(QtCore.Qt.DisplayRole)
                if r + 1 < countList:
                    next_ = selectedIndexes[r + 1]
                    if next_.row() != current.row():
                        displayText += ("\n")
                    else:
                        displayText += ("\t")
                clipboardString.write(displayText)
            pyperclip.copy(clipboardString.getvalue())


# execution bloc to launch the GUI and its features

# app = QtWidgets.QApplication(sys.argv)
# app.setStyle('Fusion')
# ui = MyMainWindow()
# ui.show()
# sys.exit(app.exec_())
#




