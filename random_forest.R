# Random Forest Model

suppressMessages(suppressWarnings({
  library(grf)
  library(pROC)
  library(dplyr)
  library(parallel)
}))

# Load and prepare training data
credit <- read.csv("clean4.csv")
credit <- credit |> select(-X)

# Convert to factors
credit <- credit |>
  mutate(TARGET = factor(TARGET, levels = c(0, 1), labels = c("No", "Yes")),
         CODE_GENDER = as.factor(CODE_GENDER),
         NAME_CONTRACT_TYPE = as.factor(NAME_CONTRACT_TYPE),
         FLAG_OWN_CAR = as.factor(FLAG_OWN_CAR),
         NAME_EDUCATION_TYPE = as.factor(NAME_EDUCATION_TYPE))

# Split data: 80% train, 20% test
set.seed(123)
credit_split <- initial_split(credit, prop = 0.8, strata = TARGET)
credit_train <- training(credit_split)
credit_test <- testing(credit_split)

# Prepare training features (convert ALL to numeric, including factors)
X_train <- credit_train |>
  select(-TARGET) |>
  as.data.frame()

# Convert all columns to numeric (factors, characters, everything)
X_train <- X_train |>
  mutate(across(everything(), function(x) {
    if (is.factor(x) | is.character(x)) {
      as.numeric(as.factor(x)) - 1
    } else if (is.numeric(x)) {
      x
    } else {
      as.numeric(x)
    }
  }))

# Replace any remaining NAs with 0
X_train <- X_train |>
  mutate(across(everything(), ~replace_na(., 0)))

# Create binary outcome: 1 = Default, 0 = No Default
Y_train <- as.numeric(credit_train$TARGET) - 1

# Prepare test features similarly (convert ALL to numeric)
X_test <- credit_test |>
  select(-TARGET) |>
  as.data.frame()

# Convert all columns to numeric (factors, characters, everything)
X_test <- X_test |>
  mutate(across(everything(), function(x) {
    if (is.factor(x) | is.character(x)) {
      as.numeric(as.factor(x)) - 1
    } else if (is.numeric(x)) {
      x
    } else {
      as.numeric(x)
    }
  }))

# Replace NAs with 0
X_test <- X_test |>
  mutate(across(everything(), ~replace_na(., 0)))

Y_test <- as.numeric(credit_test$TARGET) - 1

# Train random forest on training data
set.seed(123)
random_forest_model <- regression_forest(
  X = as.matrix(X_train),
  Y = Y_train,
  num.trees = 1000,
  num.threads = detectCores() - 1
)

# Generate predictions on test set
rf_predictions <- predict(
  random_forest_model,
  newdata = as.matrix(X_test),
  estimate.variance = TRUE
)

# Extract point predictions
rf_pred_scores <- rf_predictions$predictions

# Ensure predictions are in [0, 1] range
rf_pred_scores <- pmax(pmin(rf_pred_scores, 1), 0)

# Calculate ROC-AUC
rf_roc <- roc(Y_test, rf_pred_scores)
rf_auc <- auc(rf_roc)

print(rf_auc)

# Additional metrics
rf_pred_class <- ifelse(rf_pred_scores >= 0.5, 1, 0)
rf_accuracy <- mean(rf_pred_class == Y_test)
rf_sensitivity <- sum(rf_pred_class == 1 & Y_test == 1) / sum(Y_test == 1)
rf_specificity <- sum(rf_pred_class == 0 & Y_test == 0) / sum(Y_test == 0)

# Kaggle Submission
# Load test data
application_test <- read.csv("application_test.csv")
test_ids <- application_test$SK_ID_CURR

# Engineer features (same as training data)
application_test <- application_test |>
  mutate(
    EXT_SOURCE_MED = apply(
      cbind(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3), 
      1, 
      median, 
      na.rm = TRUE
    ),
    RATIO_AMT_CREDIT_TO_AMT_ANNUITY = AMT_CREDIT / AMT_ANNUITY,
    RATIO_AMT_ANNUITY_TO_AMT_CREDIT = AMT_ANNUITY / AMT_CREDIT
  )

# Prepare test features for prediction (convert ALL to numeric)
X_test_submission <- application_test |>
  select(-SK_ID_CURR) |>
  as.data.frame()

# Convert all columns to numeric (factors, characters, everything)
X_test_submission <- X_test_submission |>
  mutate(across(everything(), function(x) {
    if (is.factor(x) | is.character(x)) {
      as.numeric(as.factor(x)) - 1
    } else if (is.numeric(x)) {
      x
    } else {
      as.numeric(x)
    }
  }))

# Replace ALL NAs with 0 before making predictions
X_test_submission <- X_test_submission |>
  mutate(across(everything(), ~replace_na(., 0)))

# Verify no NAs remain
cat("\nNA count in submission data:", sum(is.na(X_test_submission)), "\n")

# Select only columns that exist in training data
test_cols <- intersect(colnames(X_train), colnames(X_test_submission))
X_test_submission <- X_test_submission |>
  select(all_of(test_cols))

# Generate predictions for submission
rf_submission_predictions <- predict(
  random_forest_model,
  newdata = as.matrix(X_test_submission)
)

# Extract and clip predictions to [0, 1]
rf_submission_scores <- pmax(pmin(rf_submission_predictions$predictions, 1), 0)

# Create submission file
submission_rf <- data.frame(
  SK_ID_CURR = test_ids,
  TARGET = rf_submission_scores
)

# Verify no NAs in submission
cat("NA count in submission:", sum(is.na(submission_rf$TARGET)), "\n")

# Save submission
write.csv(submission_rf, "submission_random_forest.csv", row.names = FALSE)
cat("Submission saved to submission_random_forest.csv\n")
