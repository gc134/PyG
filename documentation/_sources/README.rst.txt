======
Readme
======

PyG v1.0

-------
Summary
-------

PyG is a PyQt5 interface developped in Python3 for Windows 10. It is designed to
process germination data, in the form of raw counts of germinated seeds over time. It intends to
represent germination curves, to fit them in order to
extrapolate usual germination parameters. The variations of these parameters
between groups (genotypes, conditions, treatments...) are then represented.
The last part of PyG is dedicated to a simple statistical analysis to further
select the most interesting parameters and better characterize the variations of germination aspects.

-------
Context
-------

PyG may fill a gap between very manual processing of such data (with Excel by
example) and more sophisticated tools (such as "germination metrics" packages in R
language) by providing a complete set of tools in a easy to use and robust
GUI. The main advantage is that this interface provides a clickable tool
guiding the user through clear steps in a reliabel manner. Moreover,
PyG is a stand-alone application which doesn't require any specific technical
skills to launch and use.

--------
Features
--------

* import of csv file
* QC data
* groups sorting for visualisations
* calculation of germination percents from raw data
* visualisation of germination curves
* experimental germination curves fitting
* calculation of 5 germination parameters
* visualisation of germination parameters variations
* statistical analysis : groups comparisons
* visualisation of statistical results
* export of every intermediate output
* plots editing

-------------
How to get it
-------------
You can recover PyG on github (link). You have to download the pyg_pc.exe

----------------
How to launch it
----------------
PyG is a standalone tool, available as an a windows executable. Just to double-click the .exe file to start the interface. The interface
may take a moment to start, the time for the required modules to load.

-------
Support
-------
You can get help by contacting the project collaborators :

* shuang.peng@inrae.fr
* gwendal.cueff@inrae.fr
* loic.rajjou@agroparistech.fr

----------
Contribute
----------
this project is hosted on github (link). Please feel free to contribute in
any manner, but especially on code and interface optimimization, testing and
debugging. Additionnal features are missing, especially about germination
parameters and indices whose list is for the moment limited to the 5 essential ones.

-------------
Release notes
-------------
This is a first release of PyG, waiting for new improvements

-------
License
-------
This project is opensource under a BSD license

