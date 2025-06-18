# =============================================================================
# Script: Mortality Prediction in COPD Patients – Model Training, Testing, and Calibration
# Description: This script trains and tests multiple predictive models to estimate 
# 5-year all-cause mortality in COPD patients from Quebec using medication-based predictors, 
# and generates calibration plots for model evaluation.
# =============================================================================

# Load required libraries
library(caret)
library(pROC)
library(PRROC)
library(ggplot2)
#library(dplyr)
library(splines)

setwd("D:/Ana")

# Load data
load("mpoc_cont.RData")

# Prepare data
mpoc_cont$Residence <- ifelse(mpoc_cont$Residence == 2, 1, 0)
mpoc_cont$death <- factor(mpoc_cont$death, labels = c("C", "D"))
mpoc_cont$death <- relevel(mpoc_cont$death, "D")

# Apply spline transformation to age
age_spline <- ns(mpoc_cont$ageDIM, df = 3) %>% as.data.frame()
names(age_spline) <- c("age_spline1", "age_spline2", "age_spline3")
mpoc_cont <- cbind(mpoc_cont, age_spline)
mpoc_cont$ageDIM <- NULL

# Split into training and testing sets
set.seed(42)
split <- createDataPartition(mpoc_cont$death, p = 0.80, list = FALSE)
mpoc_train <- mpoc_cont[split, ]
mpoc_test  <- mpoc_cont[-split, ]

# Define training control
myControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  verboseIter = TRUE
)

# ========================
# Model Training
# ========================

# Logistic Regression
set.seed(42)
model_lr <- train(
  death ~ ., data = mpoc_train,
  method = "glm",
  metric = "ROC",
  trControl = myControl
)

# Elastic Net
set.seed(42)
model_glmnet <- train(
  death ~ ., data = mpoc_train,
  method = "glmnet",
  tuneLength = 5,
  metric = "ROC",
  trControl = myControl
)

# Neural Network
set.seed(42)
model_nnet <- train(
  death ~ ., data = mpoc_train,
  method = "nnet",
  tuneLength = 5,
  metric = "ROC",
  trControl = myControl
)

# XGBoost
set.seed(42)
model_xgb <- train(
  death ~ ., data = mpoc_train,
  method = "xgbDART",
  tuneLength = 5,
  metric = "ROC",
  trControl = myControl
)

# AdaBoost
set.seed(42)
model_ada <- train(
  death ~ ., data = mpoc_train,
  method = "AdaBoost.M1",
  metric = "ROC",
  trControl = myControl
)

# Random Forest (Ranger)
set.seed(42)
model_ranger <- train(
  death ~ ., data = mpoc_train,
  method = "ranger",
  tuneLength = 5,
  metric = "ROC",
  trControl = myControl
)


# ========================
# Model Evaluation
# ========================

evaluate_model <- function(model, test_data, name) {
  # Probabilities and true labels
  probs <- predict(model, newdata = test_data, type = "prob")[, "D"]
  labels <- as.numeric(test_data$death == "D")
  
  # ROC AUC
  roc <- roc.curve(scores.class0 = probs, weights.class0 = labels, curve = FALSE)
  cat(name, "- AUC-ROC:", round(roc$auc, 4), "\n")
  
  # PR AUC
  prc <- pr.curve(scores.class0 = probs, weights.class0 = labels, curve = FALSE)
  cat(name, "- AUC-PR:", round(prc$auc.integral, 4), "\n")
  
  # Brier Score
  brier <- mean((probs - labels)^2)
  cat(name, "- Brier Score:", round(brier, 5), "\n")
  
  # Calibration plot
  calibration_data <- data.frame(predicted = probs, observed = labels)
  calibration_data <- calibration_data %>%
    mutate(bin = cut(predicted, breaks = seq(0, 1, 0.1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarize(mean_pred = mean(predicted), obs_rate = mean(observed)) %>%
    na.omit()
  
  plot <- plot <- ggplot(calibration_data, aes(x = mean_pred, y = obs_rate)) +
    geom_line(color = "blue") +
    geom_point(size = 2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(x = "Predicted Probability", y = "Observed Probability",
         title = paste("Calibration -", name)) +
    theme_minimal()
  
  ggsave(paste0("calibration_", name, ".pdf"), plot = plot, width = 5, height = 5)
}

# Evaluate all models
evaluate_model(model_lr, mpoc_test, "logistic")
evaluate_model(model_glmnet, mpoc_test, "glmnet")
evaluate_model(model_nnet, mpoc_test, "nnet")
evaluate_model(model_xgb, mpoc_test, "xgb")
evaluate_model(model_ada, mpoc_test, "ada")
evaluate_model(model_ranger, mpoc_test, "ranger")

# ========================
# Calibration Plots
# ========================
# Function to generate calibration plots
plot_calibration <- function(model, test_data, file_name) {
  pred_probs <- predict(model, newdata = test_data, type = "prob")[, "D"]
  true_labels <- as.numeric(test_data$death == "D")
  
  calibration_data <- data.frame(predicted = pred_probs, observed = true_labels)
  calibration_data <- calibration_data %>%
    mutate(bin = cut(predicted, breaks = seq(0, 1, 0.1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarize(mean_pred = mean(predicted), obs_rate = mean(observed)) %>%
    na.omit()
  
  p <- ggplot(calibration_data, aes(x = mean_pred, y = obs_rate)) +
    geom_line(color = "blue") +
    geom_point(size = 2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(x = "Predicted Probability", y = "Observed Probability") +
    theme_minimal()
  
  # salvar depois de criar o gráfico
  ggsave(filename = file_name, plot = p, width = 5, height = 5)
}

# Generate calibration plots
plot_calibration(model_lr, mpoc_test, "calibration_logistic.pdf")
plot_calibration(model_glmnet, mpoc_test, "calibration_glmnet.pdf")
plot_calibration(model_nnet, mpoc_test, "calibration_nnet.pdf")
plot_calibration(model_xgb, mpoc_test, "calibration_xgb.pdf")
plot_calibration(model_ada, mpoc_test, "calibration_ada.pdf")
plot_calibration(model_ranger, mpoc_test, "calibration_ranger.pdf")
