# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\gwcueff\Documents\Bioinfo\outils_perso\python_germination\PyG_interface_2\interface_v2_2.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import rc_rc
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox)
from PyQt5.QtCore import QDir, Qt, QFile, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad

import seaborn as sns

import pingouin as pg

import statannot as sta


app = QtWidgets.QApplication(sys.argv)
app.setStyle('Fusion')

from pygui_v2 import Ui_MainWindow
from DataFrameModel_v2 import TableModel
from styleSheet_dark_orange import styleSheet


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setStyleSheet(styleSheet)

    # méthodes nécessaires à l'application

    # méthodes de changement de pages sur clic boutons latéraux de la VBoxLayout

    @pyqtSlot()
    def on_pushButton_1_clicked(self):
        self.stackedWidget.setCurrentIndex(0)


    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        self.stackedWidget.setCurrentIndex(1)

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.stackedWidget.setCurrentIndex(2)

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        self.stackedWidget.setCurrentIndex(3)

    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        self.stackedWidget.setCurrentIndex(4)

    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        self.stackedWidget.setCurrentIndex(5)

    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        self.stackedWidget.setCurrentIndex(6)

#1_p
    # méthode import bouton import_page1

    def update(self):
        self.label_page1.adjustSize()

    @pyqtSlot()
    def on_import_page1_clicked(self):
        '''
        méthode pour importer des données brutes de germination
        et pour leur affichage dans un objet tableView
        :return: self.data
        '''

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setFilter(QDir.Files)

        if dialog.exec_():
            self.file_name = dialog.selectedFiles()[0]



        try :

            self.df_raw = pd.read_csv(self.file_name, sep=";")

        except :

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Le fichier n'est pas au bon format\nSeul le format .csv avec un séparateur ; est pris en charge")
            msg.setInformativeText('Veuillez sélectionner un autre fichier')
            msg.setWindowTitle("Error")
            msg.exec_()

        else :

            self.label_page1.setText(self.file_name)
            self.update()

            model_df_raw = TableModel(self.df_raw)
            self.tableView_page1.setModel(model_df_raw)

#2_p

    # méthode de calcul des pourcentages

    @pyqtSlot()
    def on_calcul_page2_clicked(self):

        self.df_count = self.df_raw.drop(['echantillon', 'groupe', 'total'], 1)

        self.total= self.df_raw['total']

        self.df_percent = pd.DataFrame()

        for i in range(0, self.df_count.shape[0]):
            vec = self.df_count.iloc[i, :]
            vec = round((vec / self.total[i]) * 100, 1)
            vec = pd.DataFrame(vec).transpose()
            self.df_percent = pd.concat([self.df_percent, vec])

        self.df_percent_final = pd.concat([self.df_raw[["echantillon","groupe"]],self.df_percent], axis=1)

        model_df_percent_final = TableModel(self.df_percent_final)
        self.tableView_page2.setModel(model_df_percent_final)

        # extraction de la série temporelle d'observations à partir des entêtes de colonnes

        self.time_serie = list(map(float, list(self.df_count.columns)))


    @pyqtSlot()
    def on_export_page2_clicked(self):

        self.df_percent_final.to_csv("table_pourcentages.csv", sep=";", decimal=".", index=False)

    # méthode de génération des courbes de germination individuelles

# 3_p

    def image_display_gc(self,file,object):
        image = QPixmap(file)
        image = image.scaled(450, 500, Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
        object.setPixmap(image)

    @pyqtSlot()
    def on_courbe_ind_page3_clicked(self):

        curves=[]

        for i in range(0, self.df_percent.transpose().shape[1]):
            x = np.array(self.time_serie)
            y = self.df_percent.transpose().iloc[:, i]
            curves += plt.plot(x, y)

        plt.title(self.textEdit1_page3.toPlainText())
        plt.xlabel("temps")
        plt.ylabel("% germination")
        plt.legend(curves, self.df_percent_final['echantillon'], loc=0)
        plt.savefig("courbes_germ_indiv.tiff")
        plt.show()
        plt.figure()

        self.image_display_gc(file='courbes_germ_indiv.tiff', object= self.label1_page3)

    # méthode de génération de courbes de germination groupées

    @pyqtSlot()
    def on_courbe_groupe_page3_clicked(self):

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
            curves += plt.errorbar(np.array(self.time_serie), self.df_percent_mean.iloc[i, :],\
                                   self.df_percent_sd.iloc[i, :], linestyle='solid',\
                                   marker='.', label="{}".format(legend_text))

        plt.title(self.textEdit2_page3.toPlainText())
        plt.xlabel("temps")
        plt.ylabel("% germination")
        plt.legend(loc=0)
        plt.savefig("courbes_germ_groupes.tiff")
        plt.show()
        plt.figure()

        self.image_display_gc(file="courbes_germ_groupes.tiff", object=self.label2_page3)

#4_p

    # méthode d'ajustement


    # on écrit  à ce niveau la fonction de Hill pour qu'elle soit accessible partout


    def hill_function(self,x, a, b, c, yo):
        return yo + (a * np.power(x, b) / (np.power(c, b) + np.power(x, b)))


    def optimized_hill_function(self,x):
        return self.fittedParameters[3] + (self.fittedParameters[0] * np.power(x, self.fittedParameters[1]) / (
                np.power(self.fittedParameters[2], self.fittedParameters[1]) + np.power(x, self.fittedParameters[1])))

    @pyqtSlot()
    def on_calcul_page4_clicked(self):

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
            fittedParameters, pcov = curve_fit(self.hill_function, self.time_serie, yData, self.initialParameters)
            f = interp1d(self.time_serie, self.hill_function(self.time_serie, *fittedParameters),kind="slinear")
            y_fit= f(x_interpol)
            self.adj_curves += plt.plot(x_interpol, y_fit)

            fittedParameters_vec = pd.DataFrame(fittedParameters).transpose()
            self.df_fittedParameters = pd.concat([self.df_fittedParameters, fittedParameters_vec],\
                                                 axis=0, ignore_index=True)


        plt.title("ajustements")
        plt.xlabel("temps")
        plt.ylabel("% germination")
        plt.legend(self.adj_curves, self.df_percent_final['echantillon'], loc=0)
        plt.savefig("courbes_ajustement.tiff")

        self.image_display_gc(file="courbes_ajustement.tiff", object=  self.label6_page4)


    @pyqtSlot()
    def on_courbe_page4_clicked(self):

        self.adj_curves
        plt.title("ajustements")
        plt.xlabel("temps")
        plt.ylabel("% germination")
        plt.legend(self.adj_curves, self.df_percent_final['echantillon'], loc=0)
        plt.show()
        plt.figure()


    # méthode de calcul des paramètres de germination après ajustement

    @pyqtSlot()
    def on_param_ind_page4_clicked(self):

        self.df_germ_parameters = pd.DataFrame()

        for i in range(0, self.df_percent_final.shape[0]):

            self.Gmax, self.t50= round(self.df_fittedParameters.iloc[i,0], 2),\
                                 round(self.df_fittedParameters.iloc[i,2], 2)

            self.xmin = 0

            self.fittedParameters = self.df_fittedParameters.iloc[i,].values

            self.integral_res, self.integral_err = quad(self.optimized_hill_function, self.xmin, self.xmax)

            self.lag = round(np.power(((-self.df_fittedParameters.iloc[i,3]\
                                        *np.power(self.df_fittedParameters.iloc[i,2],\
                                                  self.df_fittedParameters.iloc[i,1]))\
                                 /(self.df_fittedParameters.iloc[i,0]+self.df_fittedParameters.iloc[i,3])),\
                                      1/self.df_fittedParameters.iloc[i,1]),2)

            self.D = round(self.t50 - self.lag,2)

            self.germ_parameters = [self.Gmax, self.lag, self.t50, self.D, round(self.integral_res,2)]

            self.germ_parameters = pd.DataFrame(self.germ_parameters).transpose()

            self.df_germ_parameters = self.df_germ_parameters.append(self.germ_parameters, ignore_index=True)

        self.df_germ_parameters.columns = ['Gmax','lag','t50', 'D', 'AUC']

        self.df_germ_parameters = pd.concat([self.df_percent_final['echantillon'],\
                                             self.df_percent_final["groupe"],self.df_germ_parameters],axis=1)

        self.model_indiv_germ = TableModel(self.df_germ_parameters)
        self.tableView_page4.setModel(self.model_indiv_germ)

    # méthode de calcul des paramètres de germination par groupe

    @pyqtSlot()
    def on_param_groupe_page4_clicked(self):

        self.df_germ_parameters_mean=pd. DataFrame()

        for name in self.df_germ_parameters.columns[2:]:
            vec = self.df_germ_parameters.groupby('groupe')[name].mean().values
            vec = (round(num,2) for num in vec)
            vec = pd.DataFrame(vec)
            self.df_germ_parameters_mean = pd.concat([self.df_germ_parameters_mean, vec], axis=1, ignore_index=True)

        self.df_germ_parameters_mean.insert(0, 'groupe', self.df_percent_final['groupe'].unique().tolist())

        self.df_germ_parameters_mean.columns = ['groupe', 'Gmax','lag','t50', 'D', 'AUC']

        model_group_germ = TableModel(self.df_germ_parameters_mean)
        self.tableView_page4.setModel(model_group_germ)

    # méthode pour export des tableaux de paramètres de germination individuels ou groupés

    @pyqtSlot()
    def on_export_page4_clicked(self):

        tableView_page4_model = self.tableView_page4.model()

        if tableView_page4_model is self.model_indiv_germ:
            self.df_germ_parameters.to_csv("table_param_germ_indiv.csv", sep=";", decimal=".", index=False)
        else:
            self.df_germ_parameters_mean.to_csv("table_param_germ_groupe.csv", sep=";", decimal=".", index=False)

# 5_p

    # méthode pour afficher les boxplots des paramètres de germination en fonction du paramètre choisi dans le comboBox (liste déroulante)

    def boxplot_gc(self,y):
        boxplot = sns.boxplot(x="groupe", y=y, data=self.df_germ_parameters, palette="Set1", linewidth=1, saturation=2)
        plt.title(self.boxplot_title)
        plt.ylabel(self.selected_param)
        plt.show()
        plt.figure()
        boxplot.get_figure().savefig("boxplot_{}.tiff".format(self.selected_param))

    @pyqtSlot()
    def on_pushButton_page5_clicked(self):

        # important : pour éviter la contamination sur le 1er graphe généré, par des données précédentes
        # qui trainent, on commence dans la méthode de création de graphes par un plt.clf() pour nettoyer
        # le contenu matplotlib !!!

        plt.clf()

        # pour n'utiliser le style "whitegrid" de seaborn (quadrillage gris léger en trame de fond)
        # uniquement pour ces boxplots, sans que ça ne se répercute aus fenêtres graphiques suivantes
        # on inclut nos commandes boxplot dans un bloc with: !!!

        with sns.axes_style("whitegrid"):

            self.selected_param = self.comboBox_page5.currentText()

            self.boxplot_title = self.textEdit_page5.toPlainText()

            self.boxplot_gc(y=self.selected_param)

            self.image_display_gc(file="boxplot_{}.tiff".format(self.selected_param), object=self.label_page5)

#6_p

    @pyqtSlot()
    def on_anova_page6_clicked(self):

        self.selected_param = self.comboBox_page6.currentText()

        self.aov = pg.anova(data=self.df_germ_parameters, dv=self.selected_param, between='groupe', detailed=True).round(6)

        self.aov = pd.DataFrame(self.aov.T).transpose()[['Source', 'p-unc', 'F', 'SS', 'DF', "MS", "np2"]]

        self.model_aov = TableModel(self.aov)

        self.tableView_page6.setModel(self.model_aov)


    @pyqtSlot()
    def on_multcomp_page6_clicked(self):

        self.selected_param = self.comboBox_page6.currentText()

        self.posthoc = pg.pairwise_ttests(data=self.df_germ_parameters, dv=self.selected_param, between='groupe', parametric=True, padjust='fdr_bh',
                                     effsize='hedges').round(6)

        self.posthoc= pd.DataFrame(self.posthoc.T).transpose()[
            ['Contrast', 'A', 'B', 'p-unc', 'p-corr', 'p-adjust', 'T', 'Paired', 'Parametric', 'dof', 'Tail',
             'BF10', 'hedges']]

        self.model_posthoc = TableModel(self.posthoc)

        self.tableView_page6.setModel(self.model_posthoc)


    @pyqtSlot()
    def on_export_page6_clicked(self):

        if self.tableView_page6.model() is self.model_aov :
            self.aov.to_csv("table_anova_{}.csv".format(self.selected_param), sep=";", decimal=".", index=False)
        else:
            self.posthoc.to_csv("table_multcomp_{}.csv".format(self.selected_param), sep=";", decimal=".", index=False)



# 7p

    def t_boxplot_gc(self,y):
        boxplot = sns.boxplot(x="groupe", y=y, data=self.df_germ_parameters, palette="Set1", linewidth=1, saturation=2)
        plt.title(self.boxplot_title)
        plt.ylabel(self.selected_param)
        test_results = sta.add_stat_annotation(boxplot, data=self.df_germ_parameters, x='groupe', y=y,
                                               box_pairs=self.groups_couple,
                                               test='t-test_ind',
                                               loc='outside', verbose=1, text_annot_custom=\
                                                   list(map('p={}'. format, self.posthoc['p-corr'])),
                                               perform_stat_test=True,
                                               show_test_name=True, line_offset_to_box=0.1)
        plt.tight_layout()
        plt.show()
        plt.figure()
        boxplot.get_figure().savefig("t_boxplot_{}.tiff".format(self.selected_param))


# méthode pour afficher les boxplots des paramètres de germination avec résultats stat = "Boxplots de Tukey"

    @pyqtSlot()
    def on_pushButton_page7_clicked(self):

        # important : pour éviter la contamination sur le 1er graphe généré, par des données précédentes
        # qui trainent, on commence dans la méthode de création de graphes par un plt.clf() pour nettoyer
        # le contenu matplotlib !!!

        plt.clf()

        # pour n'utiliser le style "whitegrid" de seaborn (quadrillage gris léger en trame de fond)
        # uniquement pour ces boxplots, sans que ça ne se répercute aus fenêtres graphiques suivantes
        # on inclut nos commandes boxplot dans un bloc with: !!!

        with sns.axes_style("whitegrid"):

            self.selected_param = self.comboBox_page7.currentText()

            self.posthoc = pg.pairwise_ttests(data=self.df_germ_parameters, dv=self.selected_param,
                                                        between='groupe', parametric=True, padjust='fdr_bh',
                                                        effsize='hedges').round(6)
            self.posthoc= pd.DataFrame(self.posthoc.T).transpose()

            self.groups_couple= [tuple(val) for val in self.posthoc[['A','B']].values.tolist()]

            self.boxplot_title = self.textEdit_page7.toPlainText()

            self.t_boxplot_gc(y=self.selected_param)

            self.image_display_gc(file="t_boxplot_{}.tiff".format(self.selected_param), object=self.label_page7)



# bloc d'exécution

ui = MyMainWindow()
ui.show()
sys.exit(app.exec_())
