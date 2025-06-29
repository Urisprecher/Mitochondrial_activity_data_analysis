# <ins>Mitochondrial_activity_data_analysis</ins>
## This respiratory contains scripts generated for data analysis steps using mitochondrial activity assessment, including the SEAHORSE assay.
### 3 scripts are available - Mitochondrial_activity is incoprating OCR_STATS functions to improve SEAHORSE data analysis- it runs on a folder with the initial csv files generated from the SEAHORSE assay ( raw files needs to be organized slightly and normalized to cell count - recommended wiuth CyQuant), the functions performs multiple processing and visualization steps- each one is mentioned during the run, generally- OCR vs time plots are plotted in 3 disticnt versions, bad wells are removed, outlier wells are detected, normalization is applied and final plotting is done to ensure data analysis steps are appropraite. The python scripts are both... 
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
- Cumming et al. The New Statistics: Why and How.
- Halsey et al. The fickle P value generates irreproducible results.
- Ho et al. Moving beyond P values: data analysis with estimation graphics.
- Krzywinski et al. Power and sample size.
- Cumming et al. Replication and p Intervals.
- Simpson et al. Package ‘permute’.
- Morey et al. Package ‘BayesFactor’.
- Wang et al. Managing batch effects in microbiome data.
- Leek et al. Tackling the widespread and critical impact of batch effects in high-throughput data.
