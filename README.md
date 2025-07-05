# <ins>Mitochondrial_activity_data_analysis</ins>
## This respiratory contains scripts generated for data analysis steps using mitochondrial activity assessment, including the SEAHORSE assay.
### 3 scripts are available - Mitochondrial_activity is an automatic script that allows easy and effective SEAHORSE analysis by incoprating OCR_STATS functions to improve SEAHORSE data analysis- it runs on a folder with the initial csv files generated from the SEAHORSE assay ( raw files needs to be organized slightly and normalized to cell count - recommended with CyQuant) - see file examples above. The TMRE and ROS python scripts both are used for analyzing ROS/TMRE imaging data, the scripts take as input a main folder with a file folder in it with txt files - see file examples above. 
### All output files will be saved in the main folder.
### SEAHORSE analysis steps include :
* First, OCR vs time plots are plotted in 3 disticnt versions.
* User can choose plates for downstream analysis and wells that should be excluded based on the experiment, the function will automatically exclude wells that do not follow the OCR general rules.
* Next, outlier wells are detected based on entire wells or specific timepoints for each sample.
* All plates will now be merged and normalization is applied on the data.
* Several plotting folders are generated - Normalization plots, correlation between features and OCR mean vs sd plots.
* Final OCR vs time plotting and final data output is generated at the end of the analysis for final conclusions and downstream statistics.
### TMRE & ROS analysis steps include : 
* Converting txt files to csv files step - including convertion of txt to csv files, adding PC ( plate condition such as 24, 48 hours of regular/starvation medium or any other noted condtion ) and an option to remove wells that were flagged during the experiment. 
* For combine plate scripts, next step will combine these csv files into one data frame. 
* Choosing columns for the analysis step - choosing unwanted column to remove, important feature columns which will be used for the analysis and index columns that will be used for grouping wells ( rows ) from the files. 
* Data processing and main analysis step, this includes -
  - Creating multiple indexes based on the index columns ( all possible combination of indexes )
  - Option to remove values based on different columns, removing specific cell ids based on cell_id and to remove additional columns if needed.
  - Creating statistical summary based on chosen index & removal of rows based on cell count
  - Creating box plots for each feature based on chosen index
  - creating an excel & ppt file for each group ( cell_type, compund, concentartion)
  - Creating a heatmap and pca's based on chosen index.
  - Outlier detection based on chosen index and based on one of four methods - z score, iqr, percintiles or Local outlier factor.
  - Choice of downstreram index columns
  - Feature quality control based on random forest scoring, coefficient of variation and correlations.
  - Option to rename feature names.
  - Data imputation and normalization step - includes imputation based on median values and normalization using - z score, min-max, central log, logarithmic scale, box-cox, Q-Q plots and histograms will be genertaed for each method.
  - Choice of normalization method to end analysis. 
* A dashobard will be generated to simplify the examination of results.
  ### Output data is optimized for statistics and final visualzation.
## References : 
- Caicedo et al. Data-analysis strategies for image-based cell profiling.
- Pedregosa et al. Scikit-learn: Machine Learning in Python.
- Virtanen et al. SciPy 1.0: fundamental algorithms for scientific computing in Python.
- Breunig et al. LOF: Identifying Density-Based Local Outliers. 
- Yépez VA et al. OCR-Stats: Robust estimation and statistical testing of mitochondrial respiration activities using Seahorse XF Analyzer.
- Measurement of mitochondrial respiration in adherent cells by Seahorse XF96 Cell Mito Stress Test.
- https://www.agilent.com/cs/library/usermanuals/public/S7894-10000_Rev_C_Wave_2_6_User_Guide.pdf 
