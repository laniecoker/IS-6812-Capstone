#Logistic Regression Model

suppressMessages(suppressWarnings({
  library(tidymodels)
  library(glmnet)
  library(pROC)
  library(dplyr)
  library(doParallel)
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

# Create recipe: dummy code categorical variables, normalize numeric
logistic_recipe <- recipe(TARGET ~ ., data = credit_train) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_normalize(all_numeric_predictors())

# Define logistic regression model with tuning
logistic_model <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet", family = "binomial") |>
  set_mode("classification")

# Create workflow
logistic_wf <- workflow() |>
  add_model(logistic_model) |>
  add_recipe(logistic_recipe)

# Set up hyperparameter grid
logistic_grid <- grid_regular(
  penalty(range = c(-6, -1)),
  mixture(range = c(0, 1)),
  levels = 5
)

# Set up cross-validation and parallel processing
credit_folds <- vfold_cv(credit_train, v = 5, strata = TARGET)
registerDoParallel(cores = detectCores() - 1)

# Tune model
set.seed(123)
logistic_tune <- logistic_wf |>
  tune_grid(
    resamples = credit_folds,
    grid = logistic_grid,
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(save_pred = TRUE, verbose = FALSE)
  )

# Select best model based on ROC AUC
best_params <- select_best(logistic_tune, metric = "roc_auc")

# Finalize workflow and fit on full training data
final_wf <- logistic_wf |>
  finalize_workflow(best_params) |>
  last_fit(split = credit_split)

# Print performance metrics
metrics <- final_wf |> collect_metrics()
print(metrics)

logistic_auc <- metrics |>
  filter(.metric == "roc_auc") |>
  pull(.estimate)

print(logistic_auc)

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

# Replace ALL NAs with 0 before saving
application_test <- application_test |>
  mutate(across(everything(), ~replace_na(., 0)))

# Verify no NAs remain
cat("NA count in test data:", sum(is.na(application_test)), "\n")

# Load full training data to refit model on all data
credit_full <- read.csv("clean4.csv") |> select(-X)
credit_full <- credit_full |>
  mutate(TARGET = factor(TARGET, levels = c(0, 1), labels = c("No", "Yes")),
         CODE_GENDER = as.factor(CODE_GENDER),
         NAME_CONTRACT_TYPE = as.factor(NAME_CONTRACT_TYPE),
         FLAG_OWN_CAR = as.factor(FLAG_OWN_CAR),
         NAME_EDUCATION_TYPE = as.factor(NAME_EDUCATION_TYPE))

# Refit final model on all training data
final_model <- logistic_wf |>
  finalize_workflow(best_params) |>
  fit(credit_full)

# Select only columns that exist in training data
test_cols <- intersect(colnames(credit_full), colnames(application_test))
application_test_clean <- application_test |>
  select(all_of(test_cols)) |>
  mutate(CODE_GENDER = as.factor(CODE_GENDER),
         NAME_CONTRACT_TYPE = as.factor(NAME_CONTRACT_TYPE),
         FLAG_OWN_CAR = as.factor(FLAG_OWN_CAR),
         NAME_EDUCATION_TYPE = as.factor(NAME_EDUCATION_TYPE))

# Generate predictions
predictions <- predict(final_model, new_data = application_test_clean, type = "prob")

# Create submission file
submission <- data.frame(
  SK_ID_CURR = test_ids,
  TARGET = predictions$.pred_Yes
)

# Save submission
write.csv(submission, "submission.csv", row.names = FALSE)
