###Seahorse Analysis Pipeline
## takes as input a folder with a  folder including initial csv files from seahorse exp. 
## all output files will be saved in the main folder. 
###########################################
SH_main_analysis <- function() {
  tryCatch({
    cat("\033[1;32m==== Starting Seahorse analysis!====\033[0m\n")
    message("Please ensure plate files are correctly formatted and in a folder inside the main directory.")
    message("loading function and libraries, please wait...")
    library(ggplot2)
    library(readr)
    library(plotly)
    library(dplyr)
    library(readr)
    library(data.table)
    library(magrittr)
    library(ggplot2)
    library(plotly)
    library(tidyr)  
    library(ggthemes)
    library(cli)
    library(tidyverse)
    library(corrplot)
    ## functions for SH analysis
    remove_na <- function(v){
      return(v[complete.cases(v)])
    }
    ### fit- outlier - OCR-stats edited
    fit_function_o <- function(DT = dt, Method = "LR_ao", Out_co = 5, Out_cop = 7,
                               group1 = "cell_culture", group2 = "Fibroblast_id", offset = NULL, y = "OCR") {
      print("Starting fit_function_o for outlier detection")
      DT <- copy(DT)
      cat("Using parameters: Out_co =", Out_co, "Out_cop =", Out_cop, "\n")
      
      necessary_cols <- c("plate_id", group1, group2, "well", "time", y, "Interval")
      if (!all(necessary_cols %in% names(DT))) {
        stop(paste0("Missing columns in DT: ", paste(setdiff(necessary_cols, names(DT)), collapse = ", ")))
      }
      
      if (!Method %in% c("LR", "RR", "LR_ao"))
        stop("Method unavailable. Available methods: LR, RR, LR_ao")
      if (!y %in% c("OCR", "ECAR", "lOCR", "lECAR", "OCR_n"))
        stop("Invalid response variable")
      
      if (y %in% c("OCR", "lOCR") && is.null(DT$lOCR)) DT[, lOCR := log(OCR)]
      if (y %in% c("ECAR", "lECAR") && is.null(DT$lECAR)) DT[, lECAR := log(ECAR)]
      
      if (Method == "LR_ao") {
        if (nrow(DT[is.out == FALSE]) == 0) {
          message("No data left after filtering with is.out == FALSE.")
          return(list(coefficients = NULL, fitted = NULL))
        }
        DT <- DT[is.out == FALSE]
      }
      
      var <- ifelse(y %in% c("OCR", "lOCR"), "lOCR", ifelse(y %in% c("ECAR", "lECAR"), "lECAR", "lOCR_n"))
      setnames(DT, var, "x")
      
      coef_res <- NULL
      DF <- NULL
      any_fit <- FALSE
      
      for (cc in remove_na(unique(DT[[group1]]))) {
        print(paste("Processing group:", cc))
        df <- DT[get(group1) == cc, c("plate_id", group1, group2, "x", "Interval", "well", "time", offset), with = FALSE]
        df <- na.omit(df)
        print(paste("Rows in df:", nrow(df)))
        
        if (nrow(df) > 2) {
          tryCatch({
            if (Method %in% c("LR", "LR_ao")) {
              fit <- if (is.null(offset)) {
                if (length(unique(df$well)) == 1) lm(x ~ -1 + Interval, data = df)
                else lm(x ~ -1 + Interval + well, data = df)
              } else {
                if (length(unique(df$well)) == 1) lm(x - get(offset) ~ -1 + Interval, data = df)
                else lm(x - get(offset) ~ -1 + Interval + well, data = df)
              }
            } else if (Method == "RR") {
              fit <- if (length(unique(df$well)) == 1) rq(x ~ -1 + Interval, data = df)
              else rq(x ~ -1 + Interval + well, data = df)
            }
            
            df$fitted <- fitted(fit)
            df$residuals <- residuals(fit)
            
            s <- summary(fit)$coefficients
            colnames(s)[4] <- "pvalue"
            sdt <- as.data.table(s)
            sdt$Parameter <- row.names(s)
            sdt$cell_culture <- cc
            sdt[[group2]] <- unique(df[[group2]])
            
            coef_res <- rbind(sdt, coef_res, fill = TRUE)
            DF <- rbind(df, DF, fill = TRUE)
            any_fit <- TRUE
          }, error = function(e) {
            print(paste("Error for group:", cc, "- Message:", e$message))
          })
        } else {
          print(paste("Skipping group", cc, "- not enough data"))
        }
      }
      
      if (!any_fit) {
        print("No groups had sufficient data for fitting. Returning input with outlier columns set to FALSE.")
        DT[, is.outw := FALSE]
        DT[, is.out := FALSE]
        return(list(coefficients = NULL, fitted = DT))
      }
      
      #post-process
      DF$method <- Method
      DF[, e_fitted := exp(fitted)]
      DF[, lInt_fit := mean(fitted, na.rm = TRUE), by = list(Interval, get(group1))]
      DF[, Int_fit := exp(lInt_fit)]
      DF[, sqE := (x - lInt_fit)^2]
      DF[, mean_sqE := mean(sqE, na.rm = TRUE), by = list(well, get(group1))]
      DF[, median_mean_sqE := median(mean_sqE, na.rm = TRUE), by = get(group1)]
      DF[, mad_mean_sqE := mad(mean_sqE, na.rm = TRUE), by = get(group1)]
      DF[, is.outw := median_mean_sqE + Out_co * mad_mean_sqE < mean_sqE]
      DF[, median_sqE := median(sqE, na.rm = TRUE), by = list(get(group1), Interval)]
      DF[, mad_sqE := mad(sqE, na.rm = TRUE), by = list(get(group1), Interval)]
      DF[, is.out := median_sqE + Out_cop * mad_sqE < sqE]
      DF[, sd_res := sd(x - lInt_fit)]
      setorderv(DF, c(group1, "well"))
      
      #reattach plate_id
      DF <- merge(unique(DT[, .(plate_id, well, get(group1))]), DF, by = c("well"), all.y = TRUE)
      
      print("Completed fit_function_o.")
      return(list(coefficients = coef_res, fitted = DF))
    }
    
    plot_csv_files <- function(folder_path, output_dir = "OCR_plots") {
      #output directory if it doesn't exist
      if (!dir.exists(output_dir)) {
        dir.create(output_dir)
      }
      
      #list of csv files in folder
      file_list <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)
      
      #run over each file
      for (file_path in file_list) {
        data <- read_csv(file_path)
        
        #necessary columns exist?
        if (all(c("time_real", "OCR", "Fibroblast_id", "plate_id", "type") %in% colnames(data))) {
          
          #mean, standard deviation, and sample size for OCR levels at each time_real point per condition
          summary_data <- data %>%
            group_by(time_real, type) %>%
            summarize(
              mean_ocr = mean(`OCR`, na.rm = TRUE),
              sd_ocr = sd(`OCR`, na.rm = TRUE),
              n = n(),  # Sample size for confidence interval calculation
              .groups = 'drop'
            ) %>%
            mutate(
              ci_lower = mean_ocr - (1.96 * sd_ocr / sqrt(n)),  # Lower bound of the 95% CI
              ci_upper = mean_ocr + (1.96 * sd_ocr / sqrt(n))   # Upper bound of the 95% CI
            )
          y_ax <- readline(prompt = "Enter the y axis name based on scale factor used (i.e, DEFAULT - OCR/pmol/min/1000 cells): ")
          if (tolower(y_ax) == "") {
            y_ax <- "OCR/pmol/min/1000 cells"
          }
          #plot 1: original plot with mean OCR, error bars, and data points
          p1 <- ggplot(summary_data, aes(x = time_real, y = mean_ocr, color = type)) +
            geom_line() +
            geom_errorbar(aes(ymin = mean_ocr - sd_ocr, ymax = mean_ocr + sd_ocr), width = 0.2) +
            geom_point(data = data, aes(x = time_real, y = `OCR`, shape = factor(Fibroblast_id))) +
            labs(title = paste("Original Plot for", basename(file_path)),
                 x = "Time",
                 y = paste("OCR Levels", y_ax)) +
            theme_minimal() +
            geom_vline(xintercept = 19.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 39.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 59.5, linetype = "dashed", color = "gray") +
            annotate("text", x = 19.5, y = max(summary_data$mean_ocr) + 10, label = "OLIGOMYCIN", vjust = -0.5) +
            annotate("text", x = 39.5, y = max(summary_data$mean_ocr) + 10, label = "FCCP", vjust = -0.5) +
            annotate("text", x = 59.5, y = max(summary_data$mean_ocr) + 10, label = "ROTENONE", vjust = -0.5)
          
          #plot 2: with CI shade and data points
          p2 <- ggplot(summary_data, aes(x = time_real, y = mean_ocr, color = type)) +
            geom_line() +
            geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = type), alpha = 0.2) +  # CI shade
            geom_errorbar(aes(ymin = mean_ocr - sd_ocr, ymax = mean_ocr + sd_ocr), width = 0.2) +
            geom_point(data = data, aes(x = time_real, y = `OCR`, shape = factor(Fibroblast_id))) +
            labs(title = paste("Plot with CI and Data Points for", basename(file_path)),
                 x = "Time",
                 y = paste("OCR Levels", y_ax)) +
            theme_minimal() +
            geom_vline(xintercept = 19.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 39.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 59.5, linetype = "dashed", color = "gray") +
            annotate("text", x = 19.5, y = max(summary_data$mean_ocr) + 10, label = "OLIGOMYCIN", vjust = -0.5) +
            annotate("text", x = 39.5, y = max(summary_data$mean_ocr) + 10, label = "FCCP", vjust = -0.5) +
            annotate("text", x = 59.5, y = max(summary_data$mean_ocr) + 10, label = "ROTENONE", vjust = -0.5)
          
          
          #plot 3: with CI shade but without data points
          p3 <- ggplot(summary_data, aes(x = time_real, y = mean_ocr, color = type)) +
            geom_line() +
            geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = type), alpha = 0.2) +  # CI shade
            geom_errorbar(aes(ymin = mean_ocr - sd_ocr, ymax = mean_ocr + sd_ocr), width = 0.2) +
            labs(title = paste("Plot with CI and No Data Points for", basename(file_path)),
                 x = "Time",
                 y = paste("OCR Levels", y_ax)) +
            theme_minimal() +
            geom_vline(xintercept = 19.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 39.5, linetype = "dashed", color = "gray") +
            geom_vline(xintercept = 59.5, linetype = "dashed", color = "gray") +
            annotate("text", x = 19.5, y = max(summary_data$mean_ocr) + 10, label = "OLIGOMYCIN", vjust = -0.5) +
            annotate("text", x = 39.5, y = max(summary_data$mean_ocr) + 10, label = "FCCP", vjust = -0.5) +
            annotate("text", x = 59.5, y = max(summary_data$mean_ocr) + 10, label = "ROTENONE", vjust = -0.5)
          
          #save
          file_base <- tools::file_path_sans_ext(basename(file_path))
          #first plot
          output_path1 <- file.path(output_dir, paste0(file_base, "_original.png"))
          ggsave(output_path1, plot = p1)
          interactive_plot1 <- ggplotly(p1)
          output_path4 <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(file_path)), "_original.html"))
          htmlwidgets::saveWidget(interactive_plot1, output_path4)
          
          #second plot
          output_path2 <- file.path(output_dir, paste0(file_base, "_CI_with_data.png"))
          ggsave(output_path2, plot = p2)
          interactive_plot2 <- ggplotly(p2)
          output_path5 <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(file_path)), "_CI_with_data.html"))
          htmlwidgets::saveWidget(interactive_plot2, output_path5)
          
          #third plot
          output_path3 <- file.path(output_dir, paste0(file_base, "_CI_no_data.png"))
          ggsave(output_path3, plot = p3)
          interactive_plot3 <- ggplotly(p3)
          output_path6 <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(file_path)), "_CI_no_data.html"))
          htmlwidgets::saveWidget(interactive_plot3, output_path6)
          
          
        } else {
          message("Skipping ", basename(file_path), ": required columns not found.")
        }
      }
    }
    ## PROCESSING FUNCTION
    process_data <- function(df, results_dir) {
      #results dir
      if (!dir.exists(results_dir)) {
        dir.create(results_dir)
      }
      
      cat("Step 1: Creating new 'well_plate' column based on 'well' and 'plate' columns...\n")
      df <- df %>%
        mutate(plate_well = paste(well, plate_id, sep = "_"))
      cat("Step 1 completed.\n")
      
      #save data frame
      write_csv(df, file.path(results_dir, "step1_sh_proc.csv"))
      
      #remove flagged wells based on "well_plate"
      cat("Step 2:  choose wells to remove rows based on 'well_plate' column...\n")
      cat("Current unique 'well_plate' values: \n")
      print(unique(df$plate_well))
      
      remove_well_plate <- readline(prompt="Please enter 'well_plate' values to remove (comma-separated), press enter if no wells should be removed: ")
      remove_well_plate <- unlist(strsplit(remove_well_plate, ","))  #input to vector
      
      df <- df %>%
        filter(!plate_well %in% remove_well_plate)
      cat("Step 2 completed. Rows removed based on 'well_plate'.\n")
      
      #save data frame
      write_csv(df, file.path(results_dir, "step2_sh_proc.csv"))
      
      #remove rows with NA
      cat("Step 3: Removing rows with NA values...\n")
      df <- df %>%
        drop_na()
      cat("Step 3 completed. Rows with NA values removed.\n")
      
      #save data frame
      write_csv(df, file.path(results_dir, "step3_sh_proc.csv"))
      
      #remove rows based on median logic for OCR ints columns
      cat("Step 4: Removing rows based on median[OCR(Int3)] > median[OCR(Int1)] > median[OCR(Int2)] > median[OCR(Int4)]...\n")
      
      #compute median OCR for each plate_well and interval combination
      df_median <- df %>%
        group_by(plate_well, Interval) %>%
        summarise(median_OCR = median(OCR, na.rm = TRUE), .groups = 'drop')
      
      #pivot data to get the median OCR values for each interval in separate columns
      df_wide <- df_median %>%
        pivot_wider(names_from = Interval, values_from = median_OCR, names_prefix = "OCR_")  # Pivot to get OCR_Int1, OCR_Int2, etc.
      
      #filter plate_wells based on the OCR condition
      df_filtered_wells <- df_wide %>%
        filter(OCR_Int3 > OCR_Int1 &  # Apply the condition on medians
                 OCR_Int1 > OCR_Int2 &
                 OCR_Int2 > OCR_Int4)
      
      # filter original dataframe to keep only the plate_well values that meet the condition
      df_final <- df %>%
        filter(plate_well %in% df_filtered_wells$plate_well)
      
      cat("Step 4 completed. Rows removed based on OCR median condition.\n")
      
      numeric_cols <- sapply(df_final, is.numeric)
      df_final[, numeric_cols] <- lapply(df_final[, numeric_cols], function(x) ifelse(is.na(x), 0, ifelse(x == 0, 0.00001, x)))
      
      
      #save data frame
      write_csv(df_final, file.path(results_dir, "step4_sh_proc.csv"))
      
      
      return(df_final)
    }
    ## outlier ploting
    plot_OCR_by_fibroblast <- function(DT, output_dir) {
      #output directory
      if (!dir.exists(output_dir)) {
        dir.create(output_dir)
      }
      #necessary columns present?
      necessary_cols <- c("Fibroblast_id", "OCR", "time", "plate_id", "well")
      if (!all(necessary_cols %in% colnames(DT))) {
        stop("Data frame is missing one or more necessary columns: Fibroblast_id, OCR, time, plate_id, well")
      }
      
      #run over each unique Fibroblast_id to create and save plots
      for (fibroblast_id in unique(DT$Fibroblast_id)) {
        #subset current Fibroblast_id
        data_subset <- DT[DT$Fibroblast_id == fibroblast_id, ]
        
        #plot
        plot <- ggplot(data_subset, aes(x = time, y = OCR, color = .data$well, shape = .data$plate_id)) +
          geom_point() +
          labs(title = paste("OCR over Time for Fibroblast ID:", fibroblast_id),
               x = "Time", y = "OCR") +
          theme_minimal()
        
        #save
        ggsave(filename = paste0(output_dir, "/", fibroblast_id, "OCR_plot_Fibroblast.png"), plot = plot, width = 8, height = 6)
        cat("one down")
        cat(" plot has been saved in", output_dir, "directory.\n")
      }
    }
    ### outlier detection -OCR-stats
    add_outlier_col <- function(DT, Out_co = 5, Out_cop = 7,
                                group1 = "cell_culture", group2 = "Fibroblast_id", y = "OCR") {
      DP <- copy(DT)
      cat("Using parameters: Out_co =", Out_co, "Out_cop =", Out_cop, "\n")
      
      # Initialize outlier columns if missing
      if (!"is.outw" %in% names(DP)) DP[, is.outw := FALSE]
      if (!"is.out" %in% names(DP)) DP[, is.out := FALSE]
      
      # === Well-level outlier detection ===
      DP[, aux := paste(get(group1), well)]
      n_outw <- numeric()
      iter <- 1
      keep <- TRUE
      x_lr_ao_fitted <- NULL
      
      while (keep) {
        message(paste("Running well-level outlier detection - iteration", iter))
        
        x_lr_ao <- tryCatch({
          fit_function_o(Method = "LR_ao", DT = DP[is.outw == FALSE],
                         Out_co = Out_co, Out_cop = Out_cop,
                         group1 = group1, group2 = group2, y = y)
        }, error = function(e) {
          warning(paste("Error during well-level detection:", e$message))
          return(NULL)
        })
        
        if (is.null(x_lr_ao) || is.null(x_lr_ao$fitted)) {
          message("No fitted results. Skipping to point-level detection.")
          break
        }
        
        x_lr_ao_fitted <- x_lr_ao$fitted
        
        if (!"is.outw" %in% names(x_lr_ao_fitted)) {
          warning("'is.outw' column missing. Skipping well-level detection.")
          break
        }
        
        x_out <- unique(x_lr_ao_fitted[is.outw == TRUE, .SD, .SDcols = c(group1, "well", "is.outw")])
        n_outw[iter] <- nrow(x_out)
        
        message(paste(n_outw[iter], "well outliers found on iteration", iter))
        
        if (n_outw[iter] == 0) break
        
        x_out[, aux := paste(get(group1), well)]
        DP[aux %in% x_out$aux, is.outw := TRUE]
        
        iter <- iter + 1
      }
      
      DP[, aux := NULL]
      
      tryCatch({
        p <- sum(n_outw) / (sum(x_lr_ao_fitted[time == 1, .N, by = get(group1)]$N) + sum(n_outw))
        message(paste0(round(p * 100, 2), "% well outliers found in total."))
      }, error = function(e) {
        warning(paste("Warning: Failed well-level percentage calc for", group1, "-", e$message))
      })
      
      # === Point-level outlier detection ===
      message("Starting point-level outlier detection...")
      DP[, is.out := is.outw]
      DP[, aux := paste(get(group1), well, time)]
      
      n_out <- numeric()
      iter <- 1
      keep <- TRUE
      x_lr_ao_fitted <- NULL
      
      while (keep) {
        message(paste("Running point-level outlier detection - iteration", iter))
        
        x_lr_ao <- tryCatch({
          fit_function_o(Method = "LR_ao", DT = DP[is.out == FALSE],
                         Out_co = Out_co, Out_cop = Out_cop,
                         group1 = group1, group2 = group2, y = y)
        }, error = function(e) {
          warning(paste("Error during point-level detection:", e$message))
          return(NULL)
        })
        
        if (is.null(x_lr_ao) || is.null(x_lr_ao$fitted)) {
          message("No fitted results. Stopping point-level detection.")
          break
        }
        
        x_lr_ao_fitted <- x_lr_ao$fitted
        
        if (!"is.out" %in% names(x_lr_ao_fitted)) {
          warning("'is.out' column missing. Skipping point-level detection.")
          break
        }
        
        x_out <- unique(x_lr_ao_fitted[is.out == TRUE, .SD, .SDcols = c(group1, "time", "well", "is.out")])
        n_out[iter] <- nrow(x_out)
        
        message(paste(n_out[iter], "point outliers found on iteration", iter))
        
        if (n_out[iter] == 0) break
        
        x_out[, aux := paste(get(group1), well, time)]
        DP[aux %in% x_out$aux, is.out := TRUE]
        
        iter <- iter + 1
      }
      
      DP[, aux := NULL]
      
      if (!is.null(x_lr_ao_fitted)) {
        denom <- nrow(x_lr_ao_fitted) + sum(n_out)
        if (denom == 0) denom <- 1
        p <- sum(n_out) / denom
        message(paste0(round(p * 100, 2), "% point outliers found in total."))
      }
      
      message("Returning from add_outlier_col() with dim: ", paste(dim(DP), collapse = " x "))
      return(DP)
    }
    ### main 1
    process_selected_plates <- function(folder_path) {
      
      #specified folder exists?
      if (!dir.exists(folder_path)) {
        stop("Error: The specified folder does not exist.")
      }
      
      #list of CSV files in folder
      csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
      
      #are there are any CSV files in  folder?
      if (length(csv_files) == 0) {
        stop("Error: No CSV files found in the folder.")
      }
      
      #plate names -CSV
      plate_names <- basename(csv_files)
      plate_names <- sub("\\.csv$", "", plate_names)
      
      #prompt user to select plates to process
      message("Available plates:")
      print(plate_names)
      selected_plates <- readline("Enter plate names separated by commas to select plates for processing: ")
      selected_plates <- unlist(strsplit(selected_plates, ","))
      selected_plates <- trimws(selected_plates)  #whitespace removal
      
      #files based on selected plate names
      selected_files <- csv_files[basename(csv_files) %in% paste0(selected_plates, ".csv")]
      
      #are  any files were selected?
      if (length(selected_files) == 0) {
        stop("Error: No valid files selected. Please check plate names.")
      }
      
      #folder for processed plates within the main folder
      processed_folder <- file.path("processed_plates")
      if (!dir.exists(processed_folder)) {
        dir.create(processed_folder)
        message("Processed folder created at:", processed_folder)
      }
      
      #run over each selected plate file
      for (file_path in selected_files) {
        plate_data <- tryCatch({
          read.csv(file_path)
        }, error = function(e) {
          warning(paste("Warning: Unable to read file:", file_path, "-", e$message))
          return(NULL)
        })
        
        if (is.null(plate_data)) next
        print(paste("reading file:", file_path, "-"))
        
        #plots folder
        plate_name <- sub("\\.csv$", "", basename(file_path))  # Remove ".csv" from file name
        plot_folder <- file.path(paste0(plate_name, "_scatter_plots"))
        if (!dir.exists(plot_folder)) {
          dir.create(plot_folder)
          message("Plot folder created at:", plot_folder)
        }
        ##
        plate_name <- sub("\\.csv$", "", basename(file_path))  # Remove ".csv" from file name
        proc_dir <- file.path(paste0(plate_name, "_process_files"))
        if (!dir.exists(proc_dir)) {
          dir.create(proc_dir)
          message("process files folder created at:", proc_dir)
        }
        
        #process_function
        processed_data <- tryCatch({
          process_data(plate_data, results_dir = proc_dir)
        }, error = function(e) {
          warning(paste("Warning: process_data function failed for", basename(file_path), "-", e$message))
          return(NULL)
        })
        
        if (is.null(processed_data)) next
        
        ##outlier plot function
        plate_name <- sub("\\.csv$", "", basename(file_path))  #removing .csv from file name
        outlier_plot_dir <- file.path(paste0(plate_name, "_outlier_plots"))
        if (!dir.exists(outlier_plot_dir)) {
          dir.create(outlier_plot_dir)
          message("outlier Plot folder created at:", outlier_plot_dir)
        }
        print("outlier plots initiated...")
        plot_OCR_by_fibroblast(processed_data, output_dir = outlier_plot_dir)
        
        ##
        processed_data_2 <- as.data.table(processed_data)
        #user to choose Out_co - default = 5
        Out_co <- as.numeric(readline(prompt = "Enter the value for Out_co-threshold for well outliers(higher values is more stringent, default is 5): "))
        if (is.na(Out_co)) Out_co <- 5 # Use default if input is blank or invalid
        Out_cop <- as.numeric(readline(prompt = "Enter the value for Out_cop-threshold for timepoint outliers(higher values is more stringent, default is 7): "))
        if (is.na(Out_cop)) Out_cop <- 7
        cat("Using parameters: Out_co =", Out_co, "Out_cop =", Out_cop, "\n")
        #outlier detection function
        print("outlier detection started...")
        processed_data_3 <- tryCatch({
          add_outlier_col(processed_data_2, Out_co = Out_co, Out_cop = Out_cop, group1 = "cell_culture", group2 = "Fibroblast_id", y = 'OCR')
        }, error = function(e) {
          warning(paste("Warning: add_outlier_col function failed for", basename(file_path), "-", e$message))
          return(NULL)
        })
        
        if (is.null(processed_data_3)) next
        
        #processed data to the"processed_plates folder
        plate_name <- sub("\\.csv$", "", basename(file_path))  #removing .csv from file name
        output_file <- file.path(processed_folder, paste0(plate_name, "_processed.csv"))
        tryCatch({
          write.csv(processed_data_3, output_file, row.names = FALSE)
          message(paste("Processed data saved for plate:", plate_name))
        }, error = function(e) {
          warning(paste("Warning: Failed to save processed data for plate", plate_name, "-", e$message))
        })
        
        #scatter plots for each Fibroblast_id
        print("mean vs sd scatter plots initiated")
        unique_fibroblast_ids <- unique(processed_data_3$Fibroblast_id)
        
        for (fibro_id in unique_fibroblast_ids) {
          fibro_data <- processed_data_3[Fibroblast_id == fibro_id]
          
          #mean and standard deviation of OCR for each well + Interval combination
          summary_stats <- fibro_data[, .(mean_OCR = mean(OCR, na.rm = TRUE),
                                          sd_OCR = sd(OCR, na.rm = TRUE)),
                                      by = .(well, Interval)]
          
          #linear model and extract R sq and coeff
          lm_fit <- lm(sd_OCR ~ mean_OCR, data = summary_stats)
          intercept <- coef(lm_fit)[1]
          slope <- coef(lm_fit)[2]
          r_squared <- summary(lm_fit)$r.squared
          
          #scatter plot for regular scale
          p1 <- ggplot(summary_stats, aes(x = mean_OCR, y = sd_OCR)) +
            geom_point() +
            geom_smooth(method = "lm", se = FALSE, color = "blue") +
            labs(title = paste("Mean vs SD of OCR for Fibroblast:", fibro_id),
                 x = "Mean OCR",
                 y = "SD OCR") +
            theme_minimal() +
            annotate("text",
                     x = max(summary_stats$mean_OCR, na.rm = TRUE),
                     y = min(summary_stats$sd_OCR, na.rm = TRUE),
                     label = paste0("y = ", round(intercept, 2), " + ", round(slope, 2), "x\nR² = ", round(r_squared, 2)),
                     hjust = 1, vjust = 0, color = "red", size = 3)
          
          #save
          ggsave(filename = file.path(plot_folder, paste0("OCR_mean_vs_sd_", fibro_id, ".png")), plot = p1)
          
          #scatter plot for log-transformed OCR values
          summary_stats[, log_mean_OCR := log(mean_OCR)]
          summary_stats[, log_sd_OCR := log(sd_OCR)]
          
          lm_fit_log <- lm(log_sd_OCR ~ log_mean_OCR, data = summary_stats)
          intercept_log <- coef(lm_fit_log)[1]
          slope_log <- coef(lm_fit_log)[2]
          r_squared_log <- summary(lm_fit_log)$r.squared
          
          p2 <- ggplot(summary_stats, aes(x = log_mean_OCR, y = log_sd_OCR)) +
            geom_point() +
            geom_smooth(method = "lm", se = FALSE, color = "blue") +
            labs(title = paste("Log-Mean vs Log-SD of OCR for Fibroblast:", fibro_id),
                 x = "Log Mean OCR",
                 y = "Log SD OCR") +
            theme_minimal() +
            annotate("text",
                     x = max(summary_stats$log_mean_OCR, na.rm = TRUE),
                     y = min(summary_stats$log_sd_OCR, na.rm = TRUE),
                     label = paste0("y = ", round(intercept_log, 2), " + ", round(slope_log, 2), "x\nR² = ", round(r_squared_log, 2)),
                     hjust = 1, vjust = 0, color = "red", size = 3)
          
          #save
          ggsave(filename = file.path(plot_folder, paste0("OCR_log_mean_vs_log_sd_", fibro_id, ".png")), plot = p2)
          
        }
        
        message(paste("Processed data and scatter plots saved for plate:", plate_name))
      }
      
      message("All selected plates have been processed, and plots have been saved. Please review prior to the next step.")
    }
    ## main 2
    merge_and_process <- function(folder_path, output_file_path = "merged_processed_output.csv") {
      file_folder <- file.path("merged_proc_files")
      if (!dir.exists(file_folder)) {
        dir.create(file_folder)
        message("Plot folder created at:", file_folder)
      }
      
      final_file_folder <- file.path("final_file_folder")
      if (!dir.exists(final_file_folder)) {
        dir.create(final_file_folder)
        message("final file folder created at:", final_file_folder)
      }
      
      #does specified folder exists
      if (!dir.exists(folder_path)) {
        stop("Error: The specified folder does not exist.")
      }
      
      #list of CSV files in folder
      csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
      
      #are there are any CSV files in folder?
      if (length(csv_files) == 0) {
        stop("Error: No CSV files found in the folder.")
      }
      
      #read and combine all CSV files
      message("Attempting to merge CSV files...")
      data_list <- tryCatch({
        lapply(csv_files, function(file) {
          message(paste("Reading file:", file))
          read.csv(file)
        })
      }, error = function(e) {
        stop("Error: Unable to read one or more CSV files. Check file format and content.")
      })
      
      #all CSV data into one data frame
      combined_data <- tryCatch({
        do.call(rbind, data_list)
      }, error = function(e) {
        stop("Error: Failed to merge CSV files. Ensure consistent column names across files.")
      })
      
      message("CSV files successfully merged into one data frame.")
      write_csv(combined_data, file.path(file_folder, "combined_data_proc1.csv"))
      
      #required columns check- is.outw & is.out before prompting
      if (!all(c("is.outw", "is.out") %in% colnames(combined_data))) {
        warning("Columns 'is.outw' and/or 'is.out' not found. Proceeding without outlier removal.")
      } else {
        #prompt outlier removal
        remove_outliers <- readline("Do you want to remove outliers based on 'is.outw'? (yes/no): ")
        if (tolower(remove_outliers) == "yes") {
          combined_data <- combined_data[!(combined_data$is.outw), ]
          message("Outliers have been removed based on 'is.outw' column.")
        }
        remove_outliers_2 <- readline("Do you want to remove outliers based on 'is.out'? (yes/no): ")
        if (tolower(remove_outliers_2) == "yes") {
          combined_data <- combined_data[!(combined_data$is.out), ]
          message("Outliers have been removed based on 'is.out' column.")
        }
      }
      
      #processed data to output file
      write_csv(combined_data, file.path(file_folder, "combined_data_proc2.csv"))
      ## final for ocr plot
      write_csv(combined_data, file.path(final_file_folder, "final_combined_data.csv"))
      ## add final ocr plot
      #process csv file in the final file folder
      message("Creating final OCR plot for the experiment!")
      tryCatch({
        plot_csv_files(final_file_folder)
        message("plotting ocr plot for final combined file!")
      }, error = function(e) {
        stop("Error during ocr final plotting: ", e$message)
      })
      
      
      #scatter plots for each Fibroblast_id
      plot_folder <- file.path("combined_scatter_plots")
      if (!dir.exists(plot_folder)) {
        dir.create(plot_folder)
        message("Plot folder created at:", plot_folder)
      }
      
      #handling Inf and NA values to prevent calculations and plotting issues
      scat_data <- combined_data
      #only numeric columns are modified for NA or Inf
      numeric_columns <- sapply(scat_data, is.numeric)
      scat_data[numeric_columns] <- lapply(scat_data[numeric_columns], function(col) {
        col[is.infinite(col) | is.na(col)] <- NA
        return(col)
      })
      scat_data <- scat_data[complete.cases(scat_data), ]
      scat_data <- as.data.table(scat_data)
      #scatter plots with enhanced error handling
      unique_fibroblast_ids <- unique(scat_data$Fibroblast_id)
      message("Attempting to create mean vs sd scatter plots...")
      
      tryCatch({
        for (fibro_id in unique_fibroblast_ids) {
          fibro_data <- scat_data[scat_data$Fibroblast_id == fibro_id, ]
          
          #mean and standard deviation of OCR for each well + Interval combination
          summary_stats <- fibro_data[, .(mean_OCR = mean(OCR, na.rm = TRUE),
                                          sd_OCR = sd(OCR, na.rm = TRUE)),
                                      by = .(plate_id, well, Interval)]
          # remove rows with NA values to avoid lm.fit errors
          summary_stats <- summary_stats[complete.cases(summary_stats), ]
          summary_stats <- summary_stats[is.finite(mean_OCR) & is.finite(sd_OCR)]
          
          #linear model and check for potential NA/NaN issues in lm() inputs
          if (nrow(summary_stats) > 1) {  
            lm_fit <- lm(sd_OCR ~ mean_OCR, data = summary_stats)
            intercept <- coef(lm_fit)[1]
            slope <- coef(lm_fit)[2]
            r_squared <- summary(lm_fit)$r.squared
            
            #scatter plot- regular scale
            p1 <- ggplot(summary_stats, aes(x = mean_OCR, y = sd_OCR)) +
              geom_point() +
              geom_smooth(method = "lm", se = FALSE, color = "blue") +
              labs(title = paste("Mean vs SD of OCR for Fibroblast:", fibro_id),
                   x = "Mean OCR",
                   y = "SD OCR") +
              theme_minimal() +
              annotate("text",
                       x = max(summary_stats$mean_OCR, na.rm = TRUE),
                       y = min(summary_stats$sd_OCR, na.rm = TRUE),
                       label = paste0("y = ", round(intercept, 2), " + ", round(slope, 2), "x\nR² = ", round(r_squared, 2)),
                       hjust = 1, vjust = 0, color = "red", size = 3)
            
            #save
            ggsave(filename = file.path(plot_folder, paste0("OCR_mean_vs_sd_", fibro_id, ".png")), plot = p1)
          } else {
            warning(paste("Insufficient data points for linear regression for Fibroblast_id:", fibro_id))
          }
          
          #log-transformed OCR values if mean_OCR and sd_OCR are positive
          summary_stats[, log_mean_OCR := log(mean_OCR)]
          summary_stats[, log_sd_OCR := log(sd_OCR)]
          summary_stats <- summary_stats[is.finite(log_mean_OCR) & is.finite(log_sd_OCR)]
          summary_stats <- summary_stats[complete.cases(summary_stats), ]
          
          #linear model for log-transformed data
          if (nrow(summary_stats) > 1) {
            lm_fit_log <- lm(log_sd_OCR ~ log_mean_OCR, data = summary_stats)
            intercept_log <- coef(lm_fit_log)[1]
            slope_log <- coef(lm_fit_log)[2]
            r_squared_log <- summary(lm_fit_log)$r.squared
            
            p2 <- ggplot(summary_stats, aes(x = log_mean_OCR, y = log_sd_OCR)) +
              geom_point() +
              geom_smooth(method = "lm", se = FALSE, color = "blue") +
              labs(title = paste("Log-Mean vs Log-SD of OCR for Fibroblast:", fibro_id),
                   x = "Log Mean OCR",
                   y = "Log SD OCR") +
              theme_minimal() +
              annotate("text",
                       x = max(summary_stats$log_mean_OCR, na.rm = TRUE),
                       y = min(summary_stats$log_sd_OCR, na.rm = TRUE),
                       label = paste0("y = ", round(intercept_log, 2), " + ", round(slope_log, 2), "x\nR² = ", round(r_squared_log, 2)),
                       hjust = 1, vjust = 0, color = "red", size = 3)
            
            #save
            ggsave(filename = file.path(plot_folder, paste0("OCR_log_mean_vs_log_sd_", fibro_id, ".png")), plot = p2)
          } else {
            warning(paste("Insufficient data points for log-transformed linear regression for Fibroblast_id:", fibro_id))
          }
          
          message(paste("Processed data and scatter plots saved for Fibroblast_id:", fibro_id))
        }
      }, error = function(e) {
        message("Error encountered in scatter plot generation:", e)
      })
      
      message("All scatter plots generated and saved.")
      # compute_bioenergetics function - OCR-stats
      message("Applying compute_bioenergetics function to the combined data...")
      processed_data <- tryCatch({
        combined_data <- as.data.table(combined_data)
        compute_bioenergetics(DT = combined_data, methods = c("LR"))
      }, error = function(e) {
        stop("Error: compute_bioenergetics function failed. Check data structure and compute_bioenergetics definition.")
      })
      message("bioenergetics features computed and saved!")
      #processed data to output file
      tryCatch({
        write.csv(processed_data, output_file_path, row.names = FALSE)
        message(paste("Processed data saved to:", output_file_path))
      }, error = function(e) {
        stop("Error: Failed to save processed data. Check file path and permissions.")
      })
      
      #directory for plots
      plot_folder <- file.path(dirname(output_file_path), "normalization_plots")
      if (!dir.exists(plot_folder)) {
        dir.create(plot_folder)
        message("Plot folder created at:", plot_folder)
      }
      plot_folder2 <- file.path(dirname(output_file_path), "correlation_data")
      if (!dir.exists(plot_folder2)) {
        dir.create(plot_folder2)
        message("Plot folder2 created at:", plot_folder2)
      }
      message("generating normalization plots and correlation data...")
      cor_frame <- as.data.frame(processed_data)
      perform_correlation_analysis(data = cor_frame, output_folder = plot_folder2)
      #QQ plots and histograms for each numeric column in processed_data
      numeric_features <- sapply(processed_data, is.numeric)
      for (feature in names(numeric_features)[numeric_features]) {
        tryCatch({
          # QQ Plot
          qqplot_path <- file.path(plot_folder, paste0("QQplot_", feature, ".png"))
          png(qqplot_path)
          qqnorm(processed_data[[feature]], main = paste("QQ Plot of", feature))
          qqline(processed_data[[feature]], col = "red")
          dev.off()
          
          #histogram
          hist_path <- file.path(plot_folder, paste0("Histogram_", feature, ".png"))
          png(hist_path)
          hist(processed_data[[feature]], main = paste("Histogram of", feature),
               xlab = feature, col = "lightblue", border = "black")
          dev.off()
          
          message(paste("Plots saved for feature:", feature))
          
        }, error = function(e) {
          warning(paste("Warning: Failed to create plots for feature", feature, "-", e$message))
        })
      }
      
      message("All plots generated and saved.")
    }
    ## compute bioenergetics function with dependencies - OCR-stats
    fit_function = function(DT = dt, Method = "LR", Out_co = 5, Out_cop = 7,
                            group1 = "plate_well", offset = NULL, y = "OCR"){
      
      DT <- copy(DT)
      
      # Check columns
      necessary_cols <- c("plate_id", group1, "time", y, "Interval", "Fibroblast_id", "type")
      if(! all(necessary_cols %in% names(DT)))
        stop(paste0("The following columns have to be in DT: ", paste(necessary_cols, collapse = ", ")))
      
      # Check methods
      if(! Method %in% c("LR", "RR", "LR_ao") )
        stop("Method unavailable. Available methods: LR, RR, LR_ao")
      
      # Check y
      if(! y %in% c("OCR", "ECAR", "lOCR", "lECAR", "OCR_n") )
        stop("Response variable unavailable. Available y: OCR, ECAR")
      
      if(y %in% c("OCR", "lOCR"))
        if(is.null(DT$lOCR))  DT[, lOCR := log(OCR)]
      
      if(y %in% c("ECAR", "lECAR"))
        if(is.null(DT$lECAR))  DT[, lECAR := log(ECAR)]
      
      if(Method == "LR_ao") DT = DT[is.out == FALSE]
      
      var = ifelse(y %in% c("OCR", "lOCR"),
                   "lOCR", ifelse(y %in% c("ECAR", "lECAR"), "lECAR", "lOCR_n"))
      setnames(DT, var, "x")
      
      coef_res = NULL  # contains the coefficients and pvalues
      DF = NULL    # contains the fitted values
      
      for(cc in remove_na(unique(DT[[group1]])) ){
        
        # Subset for each plate_well, keeping all descriptive columns including fibroblast_id
        df = DT[get(group1) == cc, c(group1, "x", "Interval", "time", offset, "Fibroblast_id", "type"), with = FALSE]
        df = na.omit(df)
        
        if(nrow(df) > 2 ){
          if(Method %in% c("LR", "LR_ao")){   # Linear Regression
            if(is.null(offset)){
              fit = lm(x ~ -1 + Interval, data = df)
            }else{
              fit = lm(x - get(offset) ~ -1 + Interval, data = df)
            }
            s = summary(fit)$coefficients
            
          }else if(Method == "RR"){    # Robust (abs value) Regression
            fit = rq(x ~ -1 + Interval, data = df)
            s = summary.rq(fit, se = "iid")$coefficients
          }
          
          df$fitted = fitted(fit)   # Add fitted values to the data table
          df$residuals = residuals(fit) # Add residual values to the data table
          
          colnames(s)[4] = "pvalue"
          sdt = as.data.table(s)
          sdt$Parameter = row.names(s)
          sdt[[group1]] = cc
        }
        
        coef_res = rbind(sdt, coef_res, fill = TRUE)   # contains the coefficients and pvalues
        DF = rbind(df, DF, fill = TRUE) # contains the fitted values and residuals
      } # closes the for loop
      
      coef_res$method = Method
      DF$method = Method
      DF[, e_fitted := exp(fitted)]
      DF[, lInt_fit := mean(fitted, na.rm=TRUE), by = .(Interval, get(group1))]
      DF[, Int_fit := exp(lInt_fit)]
      DF[, sqE := (x - lInt_fit)^2]
      DF[, mean_sqE := mean(sqE, na.rm=TRUE), by = .(get(group1))]
      DF[, median_mean_sqE := median(mean_sqE, na.rm=TRUE), by = get(group1)]
      DF[, mad_mean_sqE := mad(mean_sqE, na.rm=TRUE), by = get(group1)]
      DF[, is.outw := median_mean_sqE + Out_co * mad_mean_sqE < mean_sqE]
      DF[, median_sqE := median(sqE, na.rm=TRUE), by = .(get(group1), Interval)]
      DF[, mad_sqE := mad(sqE, na.rm=TRUE), by = .(get(group1), Interval)]
      DF[, is.out := median_sqE + Out_cop * mad_sqE < sqE]
      DF[, sd_res := sd(x - lInt_fit)]
      setorderv(DF, c(group1, "Interval"))
      
      # Add plate_id and fibroblast_id
      DF = right_join(unique(DT[, c("plate_id", group1, "Fibroblast_id", "type"), with = FALSE]), DF, by = group1) %>% as.data.table
      
      l = list(coefficients = coef_res, fitted = DF)
      return(l)
    }
    
    
    get_int_levels = function(DT_fitted, group1 = "plate_well"){
      ## Subset to desired columns, including fibroblast_id
      xc = DT_fitted[, c("plate_id", group1, "method", "Interval", "Int_fit", "sd_res", "Fibroblast_id.x", "type.x"), with = FALSE] %>% unique
      x_cast = dcast.data.table(xc, ... ~ Interval, value.var = "Int_fit")
      setorderv(x_cast, c("plate_id", group1))
      x_cast
    }
    
    compute_bioenergetics = function(DT, methods, Out_co = 5, Out_cop = 7,
                                     group1 = "plate_well", offset = NULL, y = 'OCR'){
      
      l_fits = l_ints = list()
      ints_dt = data.table()
      
      # Compute the well-wise fit and then obtain one value per plate_well
      for(m in methods){
        l_fits[[m]] = fit_function(DT, Method = m,  Out_co = Out_co, Out_cop = Out_cop, group1 = group1, offset = offset, y = y)
        l_ints[[m]] = get_int_levels(l_fits[[m]]$fitted, group1 = group1)
        ints_dt = rbind(ints_dt, l_ints[[m]])
      }
      
      ints_dt[, `:=` (lInt1 = log(Int1), lInt2 = log(Int2), lInt3 = log(Int3), lInt4 = log(Int4))]
      setorderv(ints_dt, c("plate_id", group1))
      
      # Keep fibroblast_id and other descriptive columns
      bio_dt = ints_dt[, .(plate_id, Fibroblast_id.x, get(group1), method, type.x,
                           Int1, Int2, Int3, Int4,
                           lInt1, lInt2, lInt3, lInt4,
                           EI = lInt1 - lInt4,  # Corresponds to basal resp.
                           AI = lInt1 - lInt2,  # Corresponds to ATP prod.
                           EAi = lInt2 - lInt4, # Corresponds to proton leak
                           MI = lInt3 - lInt1,  # Corresponds to spare cap.
                           MEi = lInt3 - lInt4, # Corresponds to maximal resp.
                           basal_resp = Int1 - Int4,
                           ATP_production = Int1 - Int2,
                           proton_leak = Int2 - Int4,
                           max_respiration = Int3 - Int4,
                           spare_capacity = Int3 - Int1,
                           non_mito_resp = Int4)]
      
      return(bio_dt)
    }
    
    perform_correlation_analysis <- function(data, output_folder) {
      if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
      }
      
      numeric_data <- data[, sapply(data, is.numeric), drop = TRUE]
      
      # Handle NA values: Pairwise deletion for correlation computation
      correlation_matrix <- cor(numeric_data, use = "pairwise.complete.obs", method = "pearson")
      
      # Save correlation matrix to CSV
      write.csv(correlation_matrix, file.path(output_folder, "correlation_matrix.csv"), row.names = TRUE)
      
      # Plot heatmap using corrplot
      png(file.path(output_folder, "correlation_heatmap.png"), width = 800, height = 800)
      corrplot(correlation_matrix, method = "pie", type = "upper", order="hclust", addCoef.col = "black",
               col = colorRampPalette(c("blue", "white", "red"))(200),
               tl.col = "black", tl.srt = 45)
      dev.off()
      
      # Optionally return results
      list(correlation_matrix = correlation_matrix)
    }
    cli_alert_success("Function and packages loaded successfully!")
    #main directory
    setwd("~")
    main_dir <- readline("Enter the main folder path (or press Enter to use the current directory): ")
    if (main_dir != "") {
      if (dir.exists(main_dir)) {
        setwd(main_dir)
        message("Directory set to: ", main_dir)
      } else {
        stop("Error: The specified main directory does not exist.")
      }
    } else {
      message("Proceeding with the current directory: ", getwd())
    }
    
    #initial folder inside dir
    folder_path <- readline("Enter the name of the folder containing your plate files: ")
    if (!dir.exists(folder_path)) {
      stop("Error: The specified folder does not exist within the main directory.")
    }
    
    #display directory information
    cli_alert_success("main folder, output folder and csv files were handled successfully!")
    message("using main directory: ", getwd())
    message("using plate data folder: ", folder_path)
    
    
    cat("\033[1;31m==== Step 1 - plotting OCR plots for each plate... ====\033[0m\n")
    #process csv files in the folder
    tryCatch({
      plot_csv_files(folder_path)
      cat("\033[1;32m==== plotting csv files for ocr plot is completed. it is recommended to inspect each plate separately!====\033[0m\n")
    }, error = function(e) {
      stop("Error during CSV plotting: ", e$message)
    })
    
    #user confirmation before proceeding
    cli_alert_success("Ocr plots per plate generated successfully!")
    message("Press enter when ready for the next step...")
    invisible(readline())
    
    
    cat("\033[1;31m==== Step 2 - processing data and detecting outliers... ====\033[0m\n")
    #process selected plates
    tryCatch({
      process_selected_plates(folder_path)
      cat("\033[1;32m==== Selected plates have been processed and outliers were detected. it is recommended to inspect output!====\033[0m\n")
      message("Selected plates have been processed.")
    }, error = function(e) {
      stop("Error during plate processing: ", e$message)
    })
    
    #user confirmation before proceeding
    cli_alert_success("Processed data and detected outliers successfully!")
    message("Press enter when ready for the next step...")
    invisible(readline())
    
    cat("\033[1;31m==== Step 3 - Merging data, proccesing and computing bioenergetics features... ====\033[0m\n")
    #merge and process the output from the processed plates
    folder_path_2 <- "processed_plates"
    if (!dir.exists(folder_path_2)) {
      stop("Error: Processed plates folder does not exist. Ensure previous steps completed successfully.")
    }
    
    tryCatch({
      merge_and_process(folder_path_2, output_file_path = "merged_processed_output.csv")
      cat("\033[1;32m==== Data merged and processed successfully. Output saved as 'merged_processed_output.csv'!====\033[0m\n")
    }, error = function(e) {
      stop("Error during data merging and processing: ", e$message)
    })
    
    cli_alert_success("Final step completed successfully!")
    cat("\033[1;31m==== Analysis completed, Proceed to the statistical analysis stage, good luck:) ====\033[0m\n")
    cat("\033[1;31m==== IMPORTANT! PLEASE REMOVE YOUR FILES FROM THE DIRECTORY ====\033[0m\n")
  },
  error = function(e) {
    message("Seahorse analysis encountered an error: ", e$message)
  })
}
SH_main_analysis()
getwd()
