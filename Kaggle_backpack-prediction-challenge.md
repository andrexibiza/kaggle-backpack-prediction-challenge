---
title: "Kaggle: Predicting Backpack Prices with Multiple Regression"
author: "Andrex Ibiza, MBA"
date: 2025-02-01
output: html_notebook
---

# Kaggle: Predicting Backpack Prices with Multiple Regression

Welcome to my notebook for the Backpack Prediction Challenge, part of Kaggle's Playground Series - Season 5, Episode 2. This competition offers an excellent opportunity to apply and enhance machine learning skills through a practical, hands-on problem.

**Competition Objective**

The goal of this challenge is to develop a predictive model that accurately forecasts a specific target variable related to backpack usage. Participants are provided with a dataset containing various features, and the task is to analyze this data to build a model that can make precise predictions.

**Approach Overview**

In this notebook, I will undertake a comprehensive analysis of the provided dataset, which includes:

- **Data Exploration**: Examining the dataset to understand the structure, identify patterns, and detect any anomalies or missing values.

- **Model Selection and Training**: Evaluating various machine learning algorithms to determine the most effective model for this task, followed by training the selected model on the dataset.

- **Model Evaluation**: Assessing the performance of the trained model using appropriate metrics to ensure its accuracy and generalizability.

- **Prediction and Submission**: Generating predictions on the test set and preparing the submission file in accordance with the competition guidelines.

Throughout this process, I will implement best practices in data analysis and machine learning, ensuring a methodical and transparent approach. The aim is not only to achieve a high-performing model but also to provide clear insights and documentation that can be valuable for future reference and learning. By engaging in this competition, I look forward to deepening my understanding of predictive modeling and contributing to the collaborative learning environment that Kaggle fosters. 

## Data Dictionary

| Column Name                  | Data Type  | Description |
|------------------------------|-----------|-------------|
| **id**                       | `<int>`    | Unique identifier for each backpack. |
| **Brand**                    | `<chr>`    | The brand of the backpack (e.g., Nike, Adidas, Jansport). |
| **Material**                 | `<chr>`    | The primary material used for the backpack (e.g., Leather, Canvas, Nylon, Polyester). |
| **Size**                     | `<chr>`    | The size category of the backpack (e.g., Small, Medium, Large). |
| **Compartments**             | `<dbl>`    | The number of compartments available in the backpack. |
| **Laptop.Compartment**       | `<chr>`    | Indicates whether the backpack has a dedicated laptop compartment (`Yes` or `No`). |
| **Waterproof**               | `<chr>`    | Indicates whether the backpack is waterproof (`Yes` or `No`). |
| **Style**                    | `<chr>`    | The style of the backpack (e.g., Backpack, Messenger, Tote). |
| **Color**                    | `<chr>`    | The primary color of the backpack. |
| **Weight.Capacity..kg.**     | `<dbl>`    | The maximum weight the backpack can carry in kilograms. |
| **Price**                    | `<dbl>`    | The price of the backpack in currency units. |

---

### Additional Notes
- Some columns (**Material**, **Size**, **Style**, **Color**) contain missing values, requiring imputation or handling during preprocessing.
- **Compartments** is stored as `<dbl>`, though it logically represents an integer count.
- **Laptop.Compartment** and **Waterproof** are categorical (`Yes/No`) and might need encoding for modeling.
- **Price** is a continuous variable and could be the target for predictive modeling.

Let's begin by loading the dataset and exploring its contents to gain a better understanding of the data we are working with.

# Load and inspect data
```{r}
# Load packages
library(ggthemes) # theme_wsj() for ggplot2
library(Hmisc) # robust describe() function
library(lightgbm) # prediction model
library(Matrix) # sparse matrix for LightGBM
library(Metrics) # evaluating model performance
library(naniar) # working with missing data
library(tidyverse) # data manipulation and visualization

# Load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

head(train)
head(test)
```


# Exploratory Data Analysis and Preprocessing

```{r}
str(train)
str(test)
```

## Check data types

### Update data types for `train`
* Brand -> Factor
* Material -> Factor
* Size -> Factor
* Compartments -> Integer
* Laptop.Compartment -> Binary Factor
* Waterproof -> Binary Factor
* Style -> Factor
* Color -> Factor

Preprocessing these categorical and numerical features is essential to ensure the dataset is clean and properly formatted for machine learning models. The first step addresses implicit missing values by replacing empty strings with `NA`, allowing for accurate handling of missing data during imputation or filtering. Without this, models might interpret empty values incorrectly, leading to biased or misleading results. Next, categorical variables such as `Brand`, `Material`, `Size`, `Style`, and `Color` are converted into factors, which enables machine learning algorithms to recognize them as discrete categories rather than arbitrary strings. Binary categorical features, `Laptop.Compartment` and `Waterproof`, are encoded as factors with values `0` and `1`, making them numerically interpretable for algorithms that require numerical inputs. Additionally, `Compartments` is explicitly cast as an integer, ensuring consistency in numeric computations. These transformations are necessary to enhance model interpretability, prevent errors in training, and optimize performance by correctly structuring the input data for machine learning algorithms.
```{r}
# Cast implicit missing values in train as NA
train$Brand[train$Brand == ""] <- NA
train$Material[train$Material == ""] <- NA
train$Size[train$Size == ""] <- NA
train$Style[train$Style == ""] <- NA
train$Color[train$Color == ""] <- NA

# Convert data types in train
train$Brand <- as.factor(train$Brand)
train$Material <- as.factor(train$Material)
train$Size <- as.factor(train$Size)
train$Compartments <- as.integer(train$Compartments)
train$Laptop.Compartment <- as.factor(ifelse(train$Laptop.Compartment == "Yes", 1, 0))
train$Waterproof <- as.factor(ifelse(train$Waterproof == "Yes", 1, 0))
train$Style <- as.factor(train$Style)
train$Color <- as.factor(train$Color)

# Rename `Weight.Capacity..kg.` -> `Weight`
names(train)[names(train) == "Weight.Capacity..kg."] <- "Weight"

str(train)
```


### Update data types for `test`

```{r}
# Cast implicit missing values in test as NA
test$Brand[test$Brand == ""] <- NA
test$Material[test$Material == ""] <- NA
test$Size[test$Size == ""] <- NA
test$Style[test$Style == ""] <- NA
test$Color[test$Color == ""] <- NA

test$Brand <- as.factor(test$Brand)
test$Material <- as.factor(test$Material)
test$Size <- as.factor(test$Size)
test$Compartments <- as.integer(test$Compartments)
test$Laptop.Compartment <- as.factor(ifelse(test$Laptop.Compartment == "Yes", 1, 0))
test$Waterproof <- as.factor(ifelse(test$Waterproof == "Yes", 1, 0))
test$Style <- as.factor(test$Style)
test$Color <- as.factor(test$Color)

# Rename test `Weight.Capacity..kg.` column
names(test)[names(test) == "Weight.Capacity..kg."] <- "Weight"
head(test, 10)

str(test)
```

## Checking for Explicit Missing Values

Handling missing values is a crucial step in data preprocessing, as they can significantly impact the performance and reliability of machine learning models. To assess the extent of missing data in our dataset, we conducted an explicit missing value check and visualized the results in the plot above.

From the visualization, we observe that several key variables contain missing values, with `Color`, `Brand`, `Material`, `Style`, and `Size` being the most affected. The `Color` feature has the highest number of missing values, followed closely by `Brand` and `Material`. On the other hand, features such as `Compartments`, `id`, and `Laptop.Compartment` have minimal or no missing values, suggesting that they are relatively complete.

Identifying these missing values early allows us to determine appropriate imputation strategies. For categorical features such as `Color`, `Brand`, `Material`, and `Style`, we can consider using mode imputation (most frequent category) or encoding missing values as a separate category. For numerical variables like `Weight`, mean or median imputation may be appropriate, depending on the data distribution. Additionally, if a large portion of a variable is missing, we may need to evaluate whether it should be removed or whether domain knowledge can help guide a more effective imputation approach.

This analysis ensures that our dataset is well-prepared for downstream machine learning tasks, reducing bias and improving the robustness of our predictive models.

```{r}
gg_miss_var(train)
```

```{r}
gg_miss_var(test)
```

Lots of missing values in: 
* Color
* Brand
* Material
* Style
* Size

```{r}
miss_var_summary(train)
```

```{r}
miss_var_summary(test)
```

## Impute missing values

Handling missing values is a crucial step in data preprocessing to ensure the dataset is suitable for machine learning. We use different imputation strategies based on the type of variable:

**Mode Imputation (Most Frequent Category) for Categorical Variables:**

* Categorical variables such as `Color`, `Brand`, `Material`, `Style`, and `Size` contain missing values.
* These are imputed using mode imputation, which replaces missing values with the most frequently occurring category in the respective column.

This approach helps preserve the distribution of categorical features without introducing artificial bias.

**Median Imputation for Numerical Variables:**

* The `Weight` variable contains missing values, which we impute using the median.
* Median imputation is preferred over mean imputation because it is robust to outliers and skewed distributions.

This ensures that missing values are replaced with a statistically representative value without distorting the distribution.

```{r}
impute_mode <- function(x) {
  x[is.na(x)] <- as.character(names(sort(table(x), decreasing = TRUE)[1]))
  return(x)
}

impute_median <- function(x) {
  # Calculate the median of non-missing values
  median_value <- median(x, na.rm = TRUE)
  
  # Replace NA values with the median
  x[is.na(x)] <- median_value
  
  return(x)
}

train$Color <- impute_mode(train$Color)
train$Brand <- impute_mode(train$Brand)
train$Material <- impute_mode(train$Material)
train$Style <- impute_mode(train$Style)
train$Size <- impute_mode(train$Size)
train$Weight <- impute_median(train$Weight)
miss_var_summary(train)
```

```{r}
test$Color <- impute_mode(test$Color)
test$Brand <- impute_mode(test$Brand)
test$Material <- impute_mode(test$Material)
test$Style <- impute_mode(test$Style)
test$Size <- impute_mode(test$Size)
test$Weight <- impute_median(test$Weight)

miss_var_summary(test)
```

## Final preprocessing checks
```{r}
describe(train)
```

```{r}
describe(test)
```

### Round weight to 3 decimal places
```{r}
train$Weight <- as.numeric(as.character(train$Weight))
test$Weight <- as.numeric(as.character(test$Weight))

train$Weight <- round(train$Weight, 3)
test$Weight <- round(test$Weight, 3)

head(train[, "Weight"])
head(test[, "Weight"])

str(train)
```


```{r}
# Inspect Weight variable for symmetric distribution
ggplot(train, aes(x = Weight)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Distribution of Weight in Train Data",
       x = "Weight (kg)",
       y = "Frequency")
```
### Why Scaling `Weight` is Unnecessary for a Random Forest Model

The histogram above represents the **distribution of the `Weight` variable** in the training dataset. In traditional regression models and algorithms like **linear regression**, Ridge, or Lasso, scaling is often necessary to prevent large-magnitude variables from disproportionately influencing model coefficients. However, in **Random Forest**, scaling is not required. The histogram provides several insights that support this:

1. Uniform and Well-Spread Distribution  
   - The `Weight` variable appears to be **evenly distributed** across its range, with no extreme outliers or sharp skewness.
   - This indicates that `Weight` is **not heavily concentrated in a small numerical range**, reducing the need for transformation.

2. Tree-Based Models Are Scale-Invariant
   - Unlike linear models, **Random Forest does not depend on feature magnitude** since it operates by **splitting data at different thresholds** rather than computing weighted sums.
   - The absolute values of `Weight` do not affect how the trees are constructed, as they only determine **split points** based on information gain or variance reduction.

3. Decision Trees Work on Rank Ordering, Not Distance Metrics
   - Algorithms like **KNN, SVM, and linear regression** rely on **distance metrics** (e.g., Euclidean distance), where unscaled numerical features can distort results.
   - **Random Forest only considers ordering and split criteria**, meaning that whether `Weight` is between **0-30** or **0-1** does not change how the trees form.

4. No Dominant Feature Magnitude Issue
   - If `Weight` had an extreme range (e.g., between 1 and 100,000), its importance in decision splits might be artificially inflated.
   - Since `Weight` falls within a reasonable range (approximately 5-30 kg), **scaling does not improve model performance**.

### Conclusion
This histogram confirms that `Weight` is **evenly distributed** and does not have extreme outliers or a disproportionate scale compared to other variables. Since **Random Forest is inherently robust to feature magnitudes**, scaling `Weight` is unnecessary. Instead, we should focus on **feature selection, hyperparameter tuning, and data quality improvements** to enhance the model's predictive power.

```{r}
# Assuming your data frame is named 'df'
# train$Weight <- scale(train$Weight)
# test$Weight <- scale(test$Weight)

# str(train)
# str(test)
```

# Regression Model for Price

## Why Regression?

In this analysis, we are predicting `Price`, a continuous numerical variable, using various categorical and numerical features (`Brand`, `Material`, `Size`, `Compartments`, `Laptop.Compartment`, `Waterproof`, `Style`, `Color`, `Weight`). The **linear regression model (`lm`)** is an appropriate choice for several key reasons:

---

### 1. Price is a Continuous Variable
- Regression is specifically designed for predicting continuous numerical outcomes.
- Since `Price` is a numeric variable, classification models (which predict categorical outputs) would not be suitable.

---

### 2. Relationship Between Predictors and Price
- The features selected (`Brand`, `Material`, `Size`, `Compartments`, etc.) are all reasonable predictors of a product’s price.
- Linear regression assumes that these features have a **linear relationship** with `Price`, meaning that changes in these attributes proportionally influence the price.

---

### 3. Interpretability of Results
- A regression model allows us to **interpret the effect** of each feature on the target variable (`Price`).
- The model coefficients show how much `Price` is expected to change when a specific predictor variable increases by one unit (e.g., the impact of adding an extra compartment or using a different material).

---

### 4. Suitable for Structured Data
- The dataset consists of structured, tabular data with **a mix of numerical and categorical features**.
- Regression models can effectively handle categorical predictors through factor encoding while treating numerical variables as continuous.

---

### 5. Handling Categorical Features
- The `train` function in R’s `caret` package automatically handles categorical variables (`Brand`, `Material`, `Size`, etc.) by encoding them as **dummy variables**.
- This allows linear regression to incorporate categorical predictors while maintaining numerical interpretability.

---

### 6. Cross-Validation for Model Reliability
- The model is trained using **10-fold cross-validation**, ensuring:
  - The model generalizes well to new, unseen data.
  - The risk of overfitting is minimized by training on multiple subsets of the data.
  - Performance is averaged across multiple training and validation splits, making the model evaluation more robust.

---

### 7. Efficiency and Scalability
- Linear regression is computationally efficient, making it suitable for large datasets.
- Unlike complex machine learning models (e.g., neural networks, decision trees), it requires minimal computational resources and is **easy to implement**.

---

### 8. Baseline Model for Further Enhancements
- Linear regression serves as an **excellent baseline model**.
- If improvements are needed, more advanced regression techniques (e.g., Ridge, Lasso, or Polynomial Regression) can be explored while maintaining the interpretability of coefficients.

---

### Conclusion
Given that we are predicting a continuous variable (`Price`), regression is the most appropriate first modeling choice. It provides an interpretable, efficient, and statistically sound framework to analyze the impact of different backpack attributes on pricing. The inclusion of **cross-validation** ensures the model’s generalizability, making it a robust choice for this problem.


```{r}
# Train linear regression model
# Load necessary library
library(caret)

set.seed(666)
# Step 1: Set up train control with 10-fold cross-validation
#train_control <- trainControl(method = "cv", number = 10)

# Step 2: Train the model using the train function from caret
#model <- train(Price ~ Brand + Material + Size + Compartments + 
#               Laptop.Compartment + Waterproof + Style + Color + Weight, 
#               data = train, 
#               method = "lm", 
#               trControl = train_control)
#print(model)
```

# Regression Model Evaluation

The linear regression model was trained and evaluated using **10-fold cross-validation**, ensuring a balanced assessment of its predictive performance across different subsets of the data. The model used **300,000 samples** and **9 predictors**, without any pre-processing applied. The key performance metrics obtained from resampling provide insight into the model's effectiveness.

The **Root Mean Squared Error (RMSE)** of **39.02** suggests a significant deviation between the predicted and actual price values, indicating that the model struggles to capture meaningful patterns in the data. The **Mean Absolute Error (MAE)** of **33.77** reinforces this finding, showing that, on average, predictions deviate from actual prices by this magnitude. More concerning is the **R-squared value of 0.001**, which indicates that the model explains almost none of the variance in the price variable. This suggests that the chosen predictors have little to no linear relationship with the target variable or that additional relevant features are missing.

Given these results, the linear regression model performs poorly as a predictor of price, likely due to the weak correlation between the input features and the target variable. The low R-squared value highlights the need for alternative modeling approaches, such as non-linear regression techniques, decision trees, or ensemble methods, which may better capture complex relationships within the data. Feature engineering, including interaction terms, transformations, or additional relevant predictors, should also be explored to improve model performance.

## Lasso vs. Ridge Regression: Which is Better for This Problem?  

Given the poor performance of the initial linear regression model, **Lasso (L1) and Ridge (L2) regression** are potential alternatives. These regularization techniques help improve model generalization and mitigate issues like overfitting or weak feature contributions. However, based on the current evaluation results, we need to determine which would be more suitable.

### Why Regularization is Needed?  
- The **low R-squared value (0.001)** suggests that the predictors do not explain much of the variance in price.  
- This could indicate a **high degree of noise** in the dataset, **multicollinearity**, or the need for **additional feature engineering**.  
- Regularization techniques like Ridge and Lasso help **reduce variance** and **improve stability** by penalizing large coefficients.

### Ridge Regression (L2 Regularization)
Ridge regression shrinks coefficients toward zero but **never fully eliminates them**. It is useful when:
- Many predictors contribute weakly to the target variable.
- Multicollinearity is present, as Ridge stabilizes coefficient estimates.
- The goal is to **retain all predictors** while reducing their impact to prevent overfitting.

However, since our **R-squared value is extremely low**, it suggests that even with all predictors, there is minimal predictive power. Ridge regression is unlikely to provide significant improvements if the problem is due to weak or irrelevant features.

### Lasso Regression (L1 Regularization)
Lasso regression **performs feature selection** by shrinking some coefficients to zero, effectively removing less important predictors. It is beneficial when:
- Many features may be irrelevant or only weakly related to the target variable.
- The model is underperforming due to unnecessary complexity.
- We need to improve interpretability by **eliminating unimportant predictors**.

Since our model exhibits extremely low explanatory power, **Lasso regression would be the better choice** because it can help eliminate predictors that contribute little to price prediction. By doing so, it simplifies the model and focuses only on the most informative features.

### Conclusion: Lasso is the Better Choice
Given the current results, **Lasso regression is the preferred approach** because:
- It helps **identify and remove irrelevant predictors** that are not contributing to price prediction.
- The dataset may contain **weak or noisy predictors**, and Lasso can focus on the most meaningful ones.
- The current linear regression model suggests **weak overall relationships**, so reducing complexity may improve interpretability.

While Ridge regression is beneficial for handling multicollinearity, our problem suggests that some features may not be useful at all. Therefore, **Lasso’s ability to perform automatic feature selection makes it the stronger candidate** for improving model performance.

# Lasso Regression Model

```{r}
# Step 1: Set up train control with 10-fold cross-validation
#train_control <- trainControl(method = "cv", number = 10)

# Step 2: Train the model using the train function from caret with Lasso Regression
#model <- train(Price ~ Brand + Material + Size + Compartments + 
#               Laptop.Compartment + Waterproof + Style + Color + Weight, 
#               data = train, 
#               method = "glmnet", 
#               trControl = train_control,
#               tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.001)))
#print(model)
```

### Why Lasso Didn’t Improve the Model and What to Do Next  

The Lasso regression model did not provide any meaningful improvement over standard linear regression. The **RMSE (39.02), MAE (33.77), and R-squared (≈0.001)** remain nearly identical across different lambda values. This strongly suggests that **the existing features have little to no predictive power for price**, and simply applying a regularization technique like Lasso is insufficient.  

### Possible Reasons for Poor Performance  
One of the main reasons for these poor results is that **the features used in the model may not have a strong or direct relationship with the target variable (`Price`)**. Some potential explanations include:  

- **Price is influenced by unobserved factors**: There may be important features missing from the dataset, such as brand reputation, customer reviews, seasonal demand, or market competition.  
- **Non-linearity in the data**: The relationship between the predictors and price may not be linear, making linear regression and its regularized variants ineffective.  
- **Weak correlation among existing predictors**: If categorical features like `Brand`, `Material`, and `Size` do not significantly impact price variation, the model struggles to find meaningful patterns.  
- **Lack of feature interactions**: Some effects on price may come from interactions between variables (e.g., a combination of `Brand` and `Material` may be more informative than each feature individually).  

# LightGBM Model
```{r}
# Convert data into a sparse matrix format
X <- model.matrix(Price ~ . -1, data = train)  # Create a matrix of predictors (remove intercept)
y <- train$Price  # Target variable

# Convert to LightGBM dataset format
lgb_train <- lgb.Dataset(data = X, label = y)

params <- list(
  objective = "regression",
  metric = "rmse",
  boosting_type = "gbdt",
  num_leaves = 63,  # Adjusted from 31
  max_depth = 10,   # Added max_depth
  learning_rate = 0.05,  # You can try 0.01 or 0.1 as well
  min_data_in_leaf = 20,  # Added to prevent overfitting
  feature_fraction = 0.8,  # Added to reduce overfitting
  bagging_fraction = 0.8,   # Added to reduce overfitting
  lambda_l1 = 0.1,  # Added L1 regularization
  lambda_l2 = 0.1   # Added L2 regularization
)

# Train the LightGBM model
lgb_model <- lgb.train(params, data = lgb_train, nrounds = 200)

# Print the model summary
print(lgb_model)

```


## Predict `Price` Using the LightGBM Model v1
```{r}
# Predict on training data
train_predictions <- predict(lgb_model, X)  # X is the model matrix created earlier

# Calculate evaluation metrics
rmse_value <- rmse(y, train_predictions)  # Compare actual vs predicted
print(paste("RMSE:", rmse_value))
mae_value <- mae(y, train_predictions)
print(paste("MAE:", mae_value))
rsq_value <- cor(y, train_predictions)^2
print(paste("R-squared:", rsq_value))

# Prepare the test data
X_test <- model.matrix(~ . - 1, data = test)  # Create a matrix of predictors for the test set

# Predict on the test data using the trained LightGBM model
test_predictions <- predict(lgb_model, X_test)

# Round the predictions to three decimal places
test$Price <- round(test_predictions, 3)

# Print the first few rows of the test dataset to verify the predictions
head(test[, "Price"])
```

## Submission File
```{r}
# Create `submission.csv` file with `id` and `Price` columns
write.csv(test[, c("id", "Price")], file = "submission.csv", row.names = FALSE)
```




## Evaluate Model Performance

```{r}
# Load necessary libraries

# --- STEP 1: Prepare Test Data and Obtain Predictions ---

# Assume 'test' is your test dataset with a column "Price"
# Create the predictor matrix (using the same formula as for training)
X_test <- model.matrix(Price ~ . - 1, data = test)  # sparse matrix for predictors
y_test <- test$Price  # actual backpack prices

# Generate predictions using your trained LightGBM model
predictions <- predict(lgb_model, newdata = X_test)

# --- STEP 2: Build a Data Frame for Analysis ---

df_results <- data.frame(Actual = y_test, Predicted = predictions)

# --- STEP 3: Fit a Linear Regression Model ---
# This step quantifies the relationship between actual and predicted values.
fit <- lm(Predicted ~ Actual, data = df_results)
fit_coef <- coef(fit)  # fit_coef[1] = intercept (b0), fit_coef[2] = slope (b1)

# --- STEP 4: Visualize the Results with ggplot2 ---

# Create a scatter plot with two reference lines:
# 1. The ideal (perfect prediction) line: slope = 1, intercept = 0 (red, dashed)
# 2. The fitted regression line from your data (green, solid)
p <- ggplot(df_results, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, 
              linetype = "dashed", color = "red", size = 1) +
  geom_abline(intercept = fit_coef[1], slope = fit_coef[2], 
              color = "green", size = 1) +
  labs(title = "Actual vs. Predicted Prices",
       subtitle = paste("Fitted line: Predicted =",
                        round(fit_coef[1], 2), "+",
                        round(fit_coef[2], 2), "* Actual"),
       x = "Actual Price",
       y = "Predicted Price") +
  theme_wsj()

print(p)

# Optionally, print a summary of the linear regression model
print(summary(fit))

```






