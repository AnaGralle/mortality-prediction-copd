# =============================================================================
# Script: Predicting 5-Year Mortality in COPD Patients Using Deep Learning
# Author: Ana Paula Pena-Gralle
# Description: This script builds, trains, test and interprets a deep neural network
#              using medication and demographic data to predict 5-year all-cause 
#              mortality in COPD patients. Model interpretability is enhanced 
#              using SHAP values.
# =============================================================================

#---- Clean workspace and load input data ----
rm(list = ls())
load("mpoc_cont.RData", verbose = TRUE)

#---- Load packages ----
reticulate::use_virtualenv("r-keras", required = TRUE)
library(reticulate)
library(keras3)
library(caret)
library(smotefamily)
library(splines)
library(tidyverse)
library(PRROC)
library(pROC)
library(data.table)


#---- Age spline transformation ----
age_spline <- ns(mpoc_cont$ageDIM, df = 3) %>% as.data.frame()
mpoc_cont <- cbind(mpoc_cont, age_spline)
names(mpoc_cont)[(ncol(mpoc_cont)-2):ncol(mpoc_cont)] <- c("age_spline1", "age_spline2", "age_spline3")
mpoc_cont <- subset(mpoc_cont, select = -ageDIM)

#---- Variable types ----
mpoc_cont$sexe <- 2 - as.numeric(as.factor(mpoc_cont$sexe))
mpoc_cont$Residence <- ifelse(mpoc_cont$Residence == 2, 1, 0)
dicho_cols <- c("sexe", "Residence", "SESLOW", names(mpoc_cont)[grep("use", names(mpoc_cont), ignore.case = TRUE)])
continuous_cols <- setdiff(names(mpoc_cont), c(dicho_cols, "death"))

#---- Train-test split ----
set.seed(42)
split <- createDataPartition(mpoc_cont$death, p = 0.80, list = FALSE)
mpoc_train <- mpoc_cont[split, ]
mpoc_test  <- mpoc_cont[-split, ]

#---- Prepare matrices ----
train_dicho_mat <- as.matrix(as.data.frame(mpoc_train[, dicho_cols, with = FALSE]))
train_cont_mat  <- scale(as.matrix(as.data.frame(mpoc_train[, continuous_cols, with = FALSE])))
train_features  <- cbind(train_dicho_mat, train_cont_mat)
train_target    <- as.matrix(mpoc_train$death)

test_dicho_mat <- as.matrix(as.data.frame(mpoc_test[, dicho_cols, with = FALSE]))
test_cont_mat  <- scale(
  as.matrix(as.data.frame(mpoc_test[, continuous_cols, with = FALSE])),
  center = attr(train_cont_mat, "scaled:center"),
  scale  = attr(train_cont_mat, "scaled:scale")
)
test_features <- cbind(test_dicho_mat, test_cont_mat)
test_target   <- as.matrix(mpoc_test$death)

#---- Deep Artificial Neural Network (Keras) ----
input <- layer_input(shape = ncol(train_features), name = "input")
output <- input %>%
  layer_dense(units = 160, activation = 'tanh') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 128, activation = 'selu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 88, activation = 'tanh') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 48, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model_75_uni <- keras_model(inputs = input, outputs = output)
model_75_uni %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(1e-4),
  metrics = c("AUC", "f1_score", "precision")
)

history_75_uni <- model_75_uni %>% fit(
  train_features,
  train_target,
  epochs = 1000,
  batch_size = 64,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(patience = 20, restore_best_weights = TRUE))
)

#---- Evaluation of D-ANN model ----
plot(history_75_uni)
round(max(history_75_uni$metrics$val_AUC), 4)
sd(history_75_uni$metrics$val_AUC, na.rm = TRUE)

predictions_75_uni <- model_75_uni %>% predict(test_features)
roc_uni <- roc.curve(predictions_75_uni, weights.class0 = test_target, curve = TRUE)
prc_uni <- pr.curve(predictions_75_uni, weights.class0 = test_target, curve = TRUE)
plot(roc_uni)
plot(prc_uni)

#---- SHAP: Medication-Only Interpretability ----
shap <- import("shap")
np <- import("numpy")
plt <- import("matplotlib.pyplot")

#---- Convert test data to Numpy array ----
X_test_np <- np$array(test_features, dtype = "float32")

#---- Create ShAP masker and explainer----
masker <- shap$maskers$Independent(X_test_np)
predict_function <- function(x) {
  model_75_uni$predict(x)
}
explainer <- shap$Explainer(predict_function, masker)

#---- Calculate ShAP values ----
shap_values <- explainer(X_test_np)
shap_results <- shap_values$values
shap_base_values <- shap_values$base_values

#---- Select medication-related predictors ----
feature_names_full <- colnames(test_features)
vars_to_exclude <- c("age_spline1", "age_spline2", "age_spline3", "sexe", "SESLOW", "Residence")
cols_to_keep_idx <- which(!(feature_names_full %in% vars_to_exclude))
X_selected_np <- np$array(test_features[, cols_to_keep_idx], dtype = "float32")
shap_selected <- shap_results[, cols_to_keep_idx]
feature_names_selected <- feature_names_full[cols_to_keep_idx]

#---- SHAP beeswarm plot ----
png("shap_beeswarm_onlyMed.png", width = 12, height = 12, units = "in", res = 300)
plt$figure()
shap$summary_plot(shap_selected, X_selected_np, feature_names = feature_names_selected, plot_type = "dot", plot_size = 0.8, color_bar = TRUE, show = TRUE)
plt$tight_layout(); plt$show(); dev.off()

#---- SHAP bar plot ----
shap_selected_abs <- np$abs(shap_selected)
png("shap_bar_abs_onlyMed.png", width = 12, height = 12, units = "in", res = 300)
plt$figure()
shap$summary_plot(shap_selected_abs, X_selected_np, feature_names = feature_names_selected, plot_type = "bar", show = TRUE)
plt$show(); dev.off()

#---- SHAP: Individual-Level Explanation (median prediction) ----
i <- which(predictions_75_uni == median(predictions_75_uni))
shap_i <- shap_results[i,]
names(shap_i) <- feature_names_full
top_idx <- order(abs(shap_i), decreasing = TRUE)[1:20]
shap_top <- shap_i[top_idx]
names_top <- names(shap_top)
colors <- ifelse(shap_top >= 0, "#D62728", "#1F77B4")

png("waterfall_median_patient.png")
plt$figure(figsize = c(7, 5))
bars <- plt$barh(rev(names_top), rev(shap_top), color = rev(colors), edgecolor = "black")
plt$title("SHAP explanation – median patient")
plt$xlabel("SHAP value (impact on model output)")
plt$axvline(x = 0, color = "grey", linestyle = "--")
plt$grid(TRUE, axis = "x", linestyle = "dotted")
plt$tight_layout()
plt$show()
dev.off()
