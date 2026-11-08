# How Accurate Is Hourly Bike Sharing Demand Estimation?


## Group 
Jinge Xu, Yujie Hu, Shuo Lyu


## Topic
own


## Question
Given observed weather and calendar information, how accurately can linear regression and k-nearest neighbors (kNN) estimate hourly bike rental counts, and which method performs better during weekday rush hours?


## Why it matters 
Bike share operators could use demand estimates to support bicycle availability and staffing decisions. Comparing linear regression and kNN will help determine whether a more flexible model improves estimation overall and during busy commuting periods.


## Data 
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
We will use UCI Bike Sharing (Fanaee-T, 2013; DOI: 10.24432/C5W894), specifically hour.csv: 17,379 hourly records from Capital Bikeshare in 2011–2012, with 17 columns and no missing values. We will use 10 raw predictors: hour, weekday, month, year, working-day status, holiday status, weather category, temperature, humidity, and wind speed. The target is total rentals (cnt). We will exclude casual and registered because their sum equals cnt, exclude the record ID, and use the date for chronological splitting. The dataset is publicly available under CC BY 4.0; no collection or scraping is needed.


## Baseline 
We will use linear regression from Chapter 2, Parametric Models (Ch02C), as the baseline to estimate hourly rental counts from weather and calendar information. Categorical variables will be encoded, and continuous variables will be standardized, with all preprocessing fitted on training data only. The model will use ordinary least squares, which requires no hyperparameter tuning.


## Comparison 
We will use kNN from Chapter 2 (Ch02A) for regression and compare it with linear regression. Following the model assessment and validation split methods in Ch02D and Ch02E, we will select k from {5, 15, 30, 60} using validation RMSE. We will also include a simple reference that predicts the training-set mean for each hour-of-day and working-day-status combination, falling back to the overall training mean if no matching combination exists.


## Evaluation 
We will split the data chronologically by complete days into approximately 60% training, 20% validation, and 20% test data. The validation set will be used to select k, and the test set will be used only for final evaluation after all modeling choices are fixed.

We will use RMSE and MAE to compare overall performance and performance during working-day rush hours (07:00–09:59 and 16:00–19:59). We will use paired block-bootstrap resampling of consecutive days to estimate 95% confidence intervals for errors and differences in errors between models.


## Key figure 
The key figure will be a grouped bar chart, with models on the x-axis and test RMSE in rentals per hour on the y-axis. Separate bars will show overall and rush-hour errors, with 95% confidence intervals.

Supporting figures will show validation error versus k and comparisons of observed and estimated demand.


## Risks and fallback 
Demand patterns may change over time, reducing model performance on later data. Neither model is guaranteed to outperform the simple reference, and differences between models may have substantial uncertainty.

If we cannot reliably complete kNN, we will narrow the study to linear regression versus the simple reference, while retaining overall and rush-hour evaluation.


## Who does what 
Jinge Xu (jingexu@engineering.upenn.edu): Data checks, exploratory analysis, chronological splits, and baseline implementation.
Yujie Hu (yujiehu@engineering.upenn.edu): Preprocessing, linear regression and kNN implementation, and validation-based tuning.
Shuo Lyu (lyu844@engineering.upenn.edu): Evaluation pipeline, rush-hour error analysis, figures, and report integration.


## AI plan
