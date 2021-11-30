=============
Project files
=============


Here is the list of the main project files. Most of them are required to launch the tool from the source code within a complete environment.


:pyg_back.py: The main module of the project containing the main class
    MyMainWindow implementing all the features

:pyg_ui.py: The GUI module containing the interface class Ui_MainWindow

:pyg_ui.ui: The xml file created by PyQt5 designer software from the
    graphically designed interface and its initial settings

:DataFrameModel.py: A module defining models to display data tables inside
    interface

:pyg_ressources.qrc: The raw external ressources file

:pyg_ressources_rc.py: An external ressources module to feed the
    interface, holding static images for example

:germinator_v5.tiff: A .tiff image that feeds the interface. This picture is
    taken from the Germinator tool article (Joosen et al., 2010).

:styleSheet_dark_orange.py: A template for GUI style customization.

:table_germination_ex_v3.csv: An example file in the right format to test the
    application