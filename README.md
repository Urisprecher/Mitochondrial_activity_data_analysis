# <ins>Mitochondrial_activity_data_analysis</ins>
## This respiratory contains scripts generated for data analysis steps using mitochondrial activity assessment, including the SEAHORSE assay.
### 3 scripts are available - Mitochondrial_activity is incoprating OCR_STATS functions to improve SEAHORSE data analysis- it runs on a folder with the initial csv files generated from the SEAHORSE assay ( raw files needs to be organized slightly and normalized to cell count - recommended wiuth CyQuant), the functions performs multiple processing and visualization steps- each one is mentioned during the run, generally- OCR vs time plots are plotted in 3 disticnt versions, bad wells are removed, outlier wells are detected, normalization is applied and final plotting is done to ensure data analysis steps are appropraite. The TMRE and ROS python scripts both are used for analyzing ROS/TMRE imaging data, the scripts take as input a main folder with a file folder in it with txt files - see file examples above. 
### All output files will be saved in the main folder.
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


### All output files will be saved in the main folder fro both scripts.
### Analysis steps for Statistics_analysis include : 
* Processing data- removing unwanted column prior to analysis and choosing index column for the analysis.
* Multi-group analysis, this step includes -
  - Outlier detection uisng mad/iqr.
  - Multiple regression anakysis.
  - Box plots for each feature.
  - Multi-group statistical testing using Anova with Tukey or Kruskal-Wallis with Dunns.
  - Manova analysis.
  - Bootstrap confidence interval analysis and visualization.
* Two-group analysis ( by user choice ), this step includes -
  - Bootstrap confidence interval analysis and visualization.
  - Variance testing ( Levene's test ) & Normality testing ( Shapiro-Wilk test ).
  - Two-group statistical testing using T-test, Wilcoxon or Mann-Whitney U.
  - Optional FDR correction & p value histograms.
  - Statistical summary with many paramters including cohens effect size calculation and plotting/
  - Power analysis.
  - Bayes factor analysis for each feature.
  - Permutation testing for each feature.
  - Logisitics regression for each feature and combination of features.
  - Linear regression for each feature and combination of features.
### Analysis steps for Batch_analysis include : 
* Processing data- choosing the row index for the feature data, choosing the treatment and batch parameters to test and data normalization. 
* Batch-effect analysis, this step includes -
  - PCA for batch detection.
  - BOX & Density plots for each feature.
  - RLE plots for each group in the treatment parameter groups.
  - Linear Regression analysis for batch effect detection.
  - Heatmap on all features and on selected features.
  - Variance calculation.
  - Batch effect type analysis. 
## References : 
- Caicedo et al. Data-analysis strategies for image-based cell profiling.
- Pedregosa et al. Scikit-learn: Machine Learning in Python.
- Virtanen et al. SciPy 1.0: fundamental algorithms for scientific computing in Python.
- Breunig et al. LOF: Identifying Density-Based Local Outliers. 
- 
- Wang et al. Managing batch effects in microbiome data.
- Leek et al. Tackling the widespread and critical impact of batch effects in high-throughput data.
