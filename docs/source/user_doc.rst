User documentation
==================

This document will guide the users through the steps of germination data
analysis with PyG. The tool usage is very sequential, linear, and each step
have to be performed before reaching the next.

Once data are collected the right way, the main steps are :

    * data loading and QC
    * percentages calculation
    * germination curves display
    * experimental curve fitting and germination parameters calculation
    * germination parameters variations with simple boxplots
    * statistical tests : anova and paired comparisons
    * Tukey's boxplot with multiple comparisons results

Data format
-----------
File format
^^^^^^^^^^^
A .csv file with a ";" separator

File structure
^^^^^^^^^^^^^^
:column 1: * sample names
    * column label = "sample"
    * type = text

:column 2: * group names
    * column label = "group"
    * type = text

:column 3: * initial total number of seeds in each tray
    * column label = "total"
    * type = numeric

:next columns: * germinated seeds counts accumulation
    * columns labels = observation time serie in hours or days
    * type = numeric
    * an unlimited number of columns can be filled after the third one

.. warning::
    Any lost data or single symbol outside the main dataframe in the .csv
    file may prevent the tool from working.

.. tip::
    In the case of data available only in the form of germination percentages,
    the user may introduce a "total" column filled with a 100 value, in order for
    the application to be able to re-calculate the same percentages at the second
    step of the analysis.

page 1 : Data loading
-------------------------------

Here the user can benefit from the following features
    * working directory choice
    * data import
    * data visualisation
    * QC statistics
    * custom group levels ordering for later boxplots

:"working directory" button: working directory definition with a file
    explorer to get every outputs in a single user-defined folder

:field 1: selected working directory path

:"import" button: data file selection through an explorer

.. note::
    If no working directory has been previously defined, the selected file
    path will become the default one.

:field 2: selected file path

:field 3: | imported data display
    | text data in green, numeric data in yellow

:field 4: QC statistics from the imported data

:entry field: custom group levels ordering using a "Return" to separate the
    levels. Here the user can modify the default alphabetical order for the
    later boxplots

:"submit" button: custom groups order validation

.. note::
    The custom groups order will only be used for the boxplots representations.

page 2 : Germination percentages calculation
------------------------------------------------------

Here the user will
    * proceed with cumulative percentages calculations for each sample
      at every observation point
    * have access to these percentages under a heatmap style
    * have the chance to export these new data into a .csv format

:"proceed" button: starts the percentages calculation up

:field: percentages data display with a heatmap style from light yellow to
    dark red for 0 to 100% values

:"export" button: percentages dataframe export, into the working directory by
    default

page 3 : Germination curves display
---------------------------------------------

At this step the user will get germination curves in matplotlib separate windows
    * germination curves by sample
    * germination curves by group with percentages means and standard deviation

:entry field 1: an optional title for the sample curves plot

:"individual curves" button: open a matplotlib window displaying the curves
    by sample
:field 1: display a thumbnail of the last individual curves plot

:entry field 2: a optional title for the grouped curves plot

:"grouped curves" button: open a matplotlib window displaying the curves by
    group

:field 2: display a thumbnail of the last grouped curves plot

.. note::
    The plots can be saved in different formats in the matplotlib window,
    with "Save the figure" tool.
    Many modifications can also be done on the plots through the available
    matplotlib tools. Try it!

page 4 : Fitting and germination parameters
-----------------------------------------------------
This is the central part of the tool. Here the user will find
    * a list of the calculated germination parameters with a quick definition
    * a graphical explanation of these parameters
    * a serie of 5 entry field for the user to set up the fitting step
    * plot of the experimental data adjustments
    * tables of germination parameters by sample or by group

At this step the following germination parameters will be computed
    :Gmax: the final percentage of germination
    :lag: the lag parameter, the time to reach the first germination, close
          to the t1 parameter
    :t50: the time needed to reach an absolute 50% of germination
    :D: the difference between t50 an lag. It can be considered as an
        estimation of germination homogeneity
    :AUC: the Area Under Curve parameter. It represents the overall
          germination capacity of a seed lot. It's the most integrative one, a
          combination of the other parameters. Through that it is considered as
          the most sensitive parameter to compare germinations.

:field 1: list of the 5 calculated germination parameters with a quick
    definition

:field 2: graphical explanation of these parameters

:entry field 1 "a-Gmax": initial value for the Hill function first parameter a. This
    value can be extrapolated from a consensus Gmax value that could approximate
    all the observed Gmax in the experiment.

:entry field 2 "b-steepness": intial value for the Hill function second parameter b. It's a
    shape parameter that describes the overall look of the curve. For
    Arabidopsis germination curves, the default value can be set at 20.

:entry field 3 "c-t50": initial value for the Hill function third parameter c. It
    corresponds to the inflexion point of the curve. It can be set based on a
    overall t50 typical of all experimental curves.

:entry field 4 "y0-intercept": initial value for the Hill function fourth parameter y0. It
    represents the intercept. In practise, its value is set to zero, because
    germination tests are performed on non germinated dry seeds.

:entry field 5 "time limit": a value between 0 and the final duration of the germination
    test that defines the temporal window for the fitting and germination
    parameters calculation steps. The typical value is the assay last temporal
    point.

.. note::
    Globally the fitting results are not too sensitive to the initial values
    provided by the user. Approximative initial values are sufficient to get
    reproducible and consistent results.

:"fitting" button: proceed with the experimental curves fitting step

:field 3: a thumbnail of the adjustement curves plot

:"adjustement curves" button: open a separate matplotlib window for a full
    display of the adjustement curves. It's then possible to save the plot.

:field 4: germination parameters tables display, either by sample or by group
    means, depending on the clicked button

:"individual parameters" button: calculate and display the 5 germination
    parameters by sample
:"grouped paramaters" button: calculate and display the group means for the 5
    germination parameters

:"export" button: pressing this button enables an export of germination
    parameter table displayed in field 4. The table will be saved in a .csv
    format, by default into the working directory.

.. note::
    In case of calculation inability due to poor initial data, NA value may
    be introduced in the parameters tables.

.. important::
    At this point of the analysis, the tool main goal is met, namely the
    calculation of essential germination parameters for each seed lot. By
    exporting the corresponding table, the user can then performed its own
    statistical analysis with the help of external sofwares and pipelines.

page 5 : Boxplots
---------------------------

In this module the user can generate independent boxplots for each
germination parameter, allowing a first view on their variations between groups.

:dropdown list: germination parameter selection

:entry field: a title for the boxplot

:"boxplots" button: open a matplotlib window that displays the boxplot of the
    selected parameter. It's then possible to save it thanks to the "Save the
    figure" tool.

:field: display a thumbnail of the latest boxplot

.. tip::
    The whole boxplots can be generated and kept side by side to make a first
    comparison between the whole parameters in the way they vary. This is one of
    the possibility given by these independent matplotlib windows.


page 6 : Statistical analysis
---------------------------------------

Here users can perform on germination parameters
    * anova
    * paired comparisons
    * p-values comparison

:dropdown list: germination parameter selection

:"anova" button: perform and display an anova analysis for the selected
    parameter. Here we take advantage of pingouin package methods.
:"multiple comparisons" button: perform and display post-hoc tests comparing
    the whole groups pairwise, for the selected parameter. Again it's pingouin
    package methods that are used.
:field 1: statistical tests outputs display
:"export" button: give the chance to export the displayed outputs, in a .csv
    format
:"summary" button: perform and display a gathering of all possible p-values from Anovas
    and paired tests.    
:entry field: a p-value threshold, between 0 and 1, to highlight the below most significant p-values,
    in red in the close display field.
:field: display the collection of p-values. In green the text data, in yellow the p-values.
    The p-values below the user-defined threshold are highlighted in red. The first row collects
    the uncorrected Anova p-values for the five parameters. The next collect, for the same parameters,
    the FDR corrected p-values of the whole paired comparisons.
:"export" button: save the whole p-values table in a .csv format. In this file, the color code is lost.



page 7 : Tukey's boxplots
-----------------------------------

In this part of the GUI the user can generate independent Tukey's boxplots for
each germination parameter, allowing a deeper understanding of their variations
between groups. These boxplots are similar to the first ones, but here
statistical results are included, namely the corrected p-values of paired
comparisons. These p-values and the relative levels of the boxplots allow a
full characterisation of the variations and eventually of the impacts of the
conditions, treatments, genotypes and so on, on germination physiology.

:dropdown list: germination parameter selection

:entry field: a title for the boxplot

:"Tukey's boxplot" button: open a matplotlib window that displays the
    Tukey's boxplot of the selected parameter. It's then possible to save it
    thanks to the "Save the figure" tool.

:field: display a thumbnail of the latest Tukey's boxplot

.. tip::
    The whole boxplots can be generated and kept side by side to make a
    comparison between the whole parameters in the way they vary. This is one of
    the possibility given by these independent matplotlib windows.