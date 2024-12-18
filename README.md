# All LendingClub loan ABS tranches analysis
<img width="600" alt="Screenshot 2024-12-16 at 5 52 47 PM" src="https://github.com/user-attachments/assets/8ab491d3-de67-44d9-a1b3-ecb8fdceb8dc" />

## Abstract
LendingClub is a peer-to-peer (P2P) lending platform that connects borrowers seeking personal loans with investors (mainly institutional) who fund these loans to earn returns.​ Asset-backed security (ABS) tranches represent segments of these loans, structured based on varying levels of risk and return. The goal is to utilize supervised machine learning to determine the different risk premiums (liquidity, default, repayment) across risk tranches and to deploy unsupervised machine learning to categorize these tranches based on their characteristics.

## Introduction
The problem revolves around understanding and assessing the risk associated with ABS tranches issued by platforms like LendingClub. In the rapidly growing P2P lending industry, ABS tranches are structured financial products that pool loans into distinct risk segments. While these tranches offer attractive returns for investors, they also carry varying levels of risk. Investors need tools to quantify risk premiums for better pricing and identify clusters in tranche characteristics to simplify risk assessment. This is critical and interesting as it helps investors improve credit risk management, optimize investment strategies, and increase overall transparency in the lending process. 

Specifically, we plan to use Ridge regression, Lasso regression, Gradient boosting, Random forest, and deep learning model to see which method better predicts risk tranches. The findings of our study are generally good with standardized lasso regression presenting a Test RMSE of 3.51%; and Gradient boosting presents a Test RMSE of 2.805%. ​Previous research (Chang et al., 2022) also found that XGBoost was the most outstanding method with an accuracy around 88%. In the future, we might consider using more machine learning methods as previous research did to extend the depth of our study. Additionally, we will also consider setting various max depth for tree-based models and different numbers of layers for neural networks in the future when we have more computational power.

## Setup
### Dataset
* Lending Club was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. ​
* All lending club loan dataset from 2007 to 2018​
* Initially: 2260668 rows × 151 columns​
* After initial cleaned: (2260451, 52)

### Data Cleaning
* For our first and one of the question on determining the factors that influence the interest rate assigned to a loan, we removed all variables that occur after the loan’s determination, such as loan status, last payment amount, and total payment amount.​
* After converting categorical data into dummy variables, the dataset expanded to 2,260,451 rows × 247 columns. This enormous dataset imposed two key limitations:​
    * Random Forest and Gradient Boosting Models: We were unable to set the `max_depth` parameter higher than 10 due to computational constraints.​
    * Neural Network Models: The number of layers in our neural networks had to be restricted, limiting model complexity.​

| **Column Name**              | **Description**                                                                 |
|------------------------------|---------------------------------------------------------------------------------|
| `id`                         | Unique identifier for each loan application.                                     |
| `loan_amnt`                  | The loan amount requested by the borrower.                                       |
| `funded_amnt`                | The total amount funded by investors for the loan.                               |
| `funded_amnt_inv`            | The portion of the loan funded by individual investors.                          |
| `term`                       | The duration of the loan (e.g., 36 months or 60 months).                         |
| `int_rate`                   | The interest rate on the loan.                                                   |
| `installment`                | The monthly payment amount the borrower must pay.                                |
| `grade`                      | The loan grade assigned by Lending Club (A, B, C, etc.).                         |
| `sub_grade`                  | The subgrade for the loan (e.g., A1, B2, etc.).                                  |
| `home_ownership`             | The borrower's homeownership status (e.g., Own, Rent, Mortgage).                 |
| `annual_inc`                 | The borrower's self-reported annual income.                                      |
| `verification_status`        | Indicates if the borrower's income was verified.                                 |
| `issue_d`                    | The month and year the loan was issued.                                          |
| `loan_status`                | The current status of the loan (e.g., Fully Paid, Charged Off).                  |
| `pymnt_plan`                 | Indicates if the borrower is on a payment plan.                                  |
| `url`                        | URL for the loan’s public Lending Club listing.                                  |
| `purpose`                    | The purpose of the loan (e.g., debt consolidation, home improvement).            |
| `zip_code`                   | The first three digits of the borrower’s zip code.                               |
| `addr_state`                 | The U.S. state provided by the borrower in the loan application.                 |
| `delinq_2yrs`                | The number of delinquencies in the borrower’s credit history over the last 2 years.|
| `earliest_cr_line`           | The date the borrower’s earliest reported credit line was opened.                |
| `fico_range_low`             | The lower bound of the borrower’s FICO score range at the time of application.   |
| `fico_range_high`            | The upper bound of the borrower’s FICO score range at the time of application.   |
| `inq_last_6mths`             | The number of credit inquiries the borrower had in the last 6 months.            |
| `open_acc`                   | The number of open credit accounts the borrower has.                             |
| `pub_rec`                    | The number of derogatory public records (e.g., bankruptcies, liens).             |
| `revol_bal`                  | The borrower’s revolving balance (total credit card debt).                       |
| `total_acc`                  | The total number of credit accounts the borrower has (open or closed).           |
| `initial_list_status`        | The initial listing status of the loan (e.g., whole, fractional).                |
| `out_prncp`                  | Remaining outstanding principal on the loan.                                     |
| `out_prncp_inv`              | Remaining outstanding principal funded by individual investors.                  |
| `total_pymnt`                | The total amount paid by the borrower.                                           |
| `total_pymnt_inv`            | The total payment received by individual investors.                              |
| `total_rec_prncp`            | The total amount of principal received on the loan.                              |
| `total_rec_int`              | The total interest received on the loan.                                         |
| `total_rec_late_fee`         | The total late fees received on the loan.                                        |
| `recoveries`                 | The total amount recovered after the loan was charged off.                       |
| `collection_recovery_fee`    | The fees paid for recovering the charged-off loan.                               |
| `last_pymnt_amnt`            | The amount of the last payment received.                                         |
| `last_credit_pull_d`         | The date when the borrower's credit was last pulled for review.                  |
| `last_fico_range_high`       | The upper bound of the most recent FICO score range of the borrower.             |
| `last_fico_range_low`        | The lower bound of the most recent FICO score range of the borrower.             |
| `collections_12_mths_ex_med` | The number of collections in the past 12 months, excluding medical collections.   |
| `policy_code`                | A code indicating the policy used for the loan.                                  |
| `application_type`           | Indicates whether the loan is individual or joint.                               |
| `acc_now_delinq`             | The number of accounts currently delinquent.                                     |
| `chargeoff_within_12_mths`   | The number of charge-offs within the last 12 months.                             |
| `delinq_amnt`                | The dollar amount of delinquent debt.                                            |
| `tax_liens`                  | The number of tax liens on the borrower’s credit record.                         |
| `hardship_flag`              | Indicates whether the borrower has claimed hardship.                             |
| `disbursement_method`        | The method of loan disbursement (e.g., cash, direct pay).                        |
| `debt_settlement_flag`       | Indicates if the borrower has entered into a debt settlement.                    |


## Model Setup

### Linear Approach: Standardized Lasso Regression
* Density Plot
  * <details>
     <summary>Click to Expand Image</summary>
      <img width="400" alt="Screenshot 2024-12-16 at 6 22 09 PM" src="https://github.com/user-attachments/assets/b2f4116a-556c-465a-ba4e-b93b15b653d6" />
  * <details>
     <summary>Click to Expand Image</summary>
      <img width="400" alt="Screenshot 2024-12-16 at 6 30 57 PM" src="https://github.com/user-attachments/assets/e8095dbd-a81d-4e71-908d-c48868f30cb6" />
   * <details>
     <summary>Click to Expand Image</summary>
      <img width="400" alt="Screenshot 2024-12-16 at 6 32 09 PM" src="https://github.com/user-attachments/assets/3a6a5c6f-5d0d-4241-bb72-537257e0af97" />
* Ridge Regression V.S **Lasso Regression**
  
  Lasso Regression was chosen because of two main reasons. First, Lasso Regression can shrink some coefficients to exactly zero. This means Lasso can identify and eliminate redundant features, leading to a simpler and more interpretable model. This is very important for our dataset because there are many predictors in our dataset that are irrelevant, such as loan_amnt, funded_amnt, collection_recovery fee, and Lasso set their coefficients to exactly zero, and consider only significant features. Secondly, the Root MSE of Lasso Regression is 3. 514, which is lower than the Root MSE of Ridge Regression, which is 3.522, demonstrating that Lasso Regression have a higher accuracy.
* Non-Standardized Lasso Regression V.S **Standardized Lasso Regression**
  
  Standardization rescales all predictor variables to have a mean of 0 and a standard deviation of 1. This makes sure that all variables contribute proportionally to the model, regardless of their original scale.​ Without standardization, variables with larger ranges would dominate the model's coefficients, while smaller-scale variables can be ignored.

### Non-Linear Approach: Gradient Boosting
* Density Plot
   * <details> n_estimators=100, ​max_depth=5
     <summary>Click to Expand Image</summary>
      <img width="355" alt="Screenshot 2024-12-16 at 9 49 59 PM" src="https://github.com/user-attachments/assets/3980505f-1363-4a17-a4c2-60ab9cd103aa" />
   * <details> n_estimators=100, ​max_depth=10
     <summary>Click to Expand Image</summary>
      <img width="384" alt="Screenshot 2024-12-16 at 9 59 50 PM" src="https://github.com/user-attachments/assets/1c18b3ab-6d93-4e33-bf12-4fd53b752898" />
      

* **Gradient Boosting** V.S Random Forest
  
  From Lasso regression, we observe large residuals. As an iterative method, GB is better at handling such issue by refining prediction.​ For Random forest, once trained, Random Forest trees do not adapt to errors. This independence limits its ability to focus on and correct large residuals effectively, making it less flexible than Gradient Boosting for iterative error correction.​ As our dataset is large enough, there is no concern of overfitting. Large 'n_estimators' also reduce the risk by lowering learning rate.

   With the same number of estimators, increasing tree depth increases model accuracy and R^2.​ Overall, random forest underperforms gradient boosting, which aligns with our previous analysis.
  

## Results
### Findings from the Standardized Lasso Regression
* <details> 
     <summary>Click to Expand Image</summary>
      <img width="587" alt="Screenshot 2024-12-16 at 10 05 14 PM" src="https://github.com/user-attachments/assets/468c2d62-202f-4601-8062-820193ae2447" />

The bar graph from the lasso regression showed that the coefficients increases by issue year. This trends show that loans issued in more recent year (2016-2018) had a stronger impact on the model's prediction. 

* <details> 
     <summary>Click to Expand Image</summary>
      <img width="553" alt="Screenshot 2024-12-16 at 10 06 38 PM" src="https://github.com/user-attachments/assets/9e3f8197-c67d-4318-b103-89c32fe5372c" />
This is the US interest rate, the trend shows that the rate remained around 0 from 2009 to around 2016, after which it began to rise gradually. ​This increase in interest rates in the US interest rate aligns with the upward trend that we find with the lasso regression. Some of the possible reasons could be that borrowers who seek out loans during periods of rising interest rates might have experienced greater financial strain.

* <details> 
     <summary>Click to Expand Image</summary>
      <img width="606" alt="Screenshot 2024-12-16 at 10 09 26 PM" src="https://github.com/user-attachments/assets/bc70fdc4-f4d9-4092-b61e-58e16d1f094f" />
Loans issued in August and September has the highest positive coefficients, while loans in February have the highest negative coefficients. ​
Some of the possible reasons for the trend could because that borrowers in August or September may display behaviors leading to better performance, such as financial stability or better repayment.​ What's more, LendingClub may have changed lending standards during these months, influencing loan performance

* <details> 
     <summary>Click to Expand Image</summary>
      <img width="335" alt="Screenshot 2024-12-16 at 10 12 38 PM" src="https://github.com/user-attachments/assets/379f54d3-8781-4ed9-b21f-743f50c0310d" />
      <img width="306" alt="Screenshot 2024-12-16 at 10 13 10 PM" src="https://github.com/user-attachments/assets/d40e6e14-5090-4a68-8a75-b6b59deab350" />
Findings from the standardized lasso regression showed that location have minor impact and purpose have significant impact.

### Findings from Gradient Boosting
Predictions are generally well-aligned with the red "perfect prediction line," indicating great accuracy. Some spread is visible at higher interest rate.
Based on the importance table, loan amount is the most significant feature, followed by annual income and revolving balance.
Train RMSE of GB is 2.4, which is lower than that of lasso (3.5), indicating that GB provides more accurate predictions.


## Discussion
The findings of our study are generally good with standardized lasso regression presenting a Test RMSE of 3.51%; and Gradient boosting presents a Test RMSE of 2.805%.The RMSE for both models are relatively low, suggesting relatively strong predictive power. ​The improvement from 3.51% to 2.805% between the linear and nonlinear models is significant, justifying the use of Gradient Boosting for better accuracy.  ​Previous research (Chang et al., 2022) also found that XGBoost was the most outstanding method with an accuracy around 88%.

## Conclusion
In this project, we applied a combination of supervised and unsupervised learning techniques to analyze asset-backed security (ABS) tranches within a peer-to-peer (P2P) lending platform. We identified  standardized lasso regression and gradient boosting to have the best predictive power. It suggests a large improvement from linear model to non-linear mode, reflecting the complexity of the dataset and the problem we want to address. While we use most methods learned in-class, we can adjust parameters in the future to get potentially better models. 

### Expected Return(including default and write off and exclusing current)

* <details> 
     <summary>Expected Return Visualization</summary>
      <img width="1181" alt="Screenshot 2024-12-16 at 10 50 38 PM" src="https://github.com/user-attachments/assets/8298fedd-9c46-4b0b-85a4-b96c37cbe01a" />
      <img width="742" alt="Screenshot 2024-12-16 at 10 48 34 PM" src="https://github.com/user-attachments/assets/05f9f39f-0f8a-4f6b-b3c7-c848af68f12b" />
      <img width="1000" alt="Screenshot 2024-12-16 at 10 52 12 PM" src="https://github.com/user-attachments/assets/b9df133c-7b9b-464d-a2f3-588ccd010665" />

When the grade of loans deteriorates, the distribution of returns becomes larger, which indicates that higher-grade loans have more stable returns. Additionally, as the grade of loans deteriorates, the percentage of charged-off loans increases. However, surprisingly, even with defaults and write-offs, there is a 50% recovery on the initial lending amount.

We ran a generalized linear model with lasso regularization to predict the expected return.
* <details> 
     <summary>Click to Expand Lasso</summary>
      <img width="579" alt="Screenshot 2024-12-16 at 11 32 19 PM" src="https://github.com/user-attachments/assets/8631766b-220b-455c-b4db-a93907c5ce6c" />

we get a RMSE of 0.2206, which is quite unaccurate. Then we run a gradient boosting model:
* <details> 
     <summary>Click to Expand Gradient Boosting</summary>
      <img width="555" alt="Screenshot 2024-12-17 at 12 14 28 AM" src="https://github.com/user-attachments/assets/186d5a10-bcab-491c-ab16-04202a43cdc0" />
     <summary>Click to Expand Gradient Boosting Importance</summary>
      <img width="244" alt="Screenshot 2024-12-17 at 12 14 54 AM" src="https://github.com/user-attachments/assets/c80ba758-dfcc-4da4-9be6-ac5004f82131" />

### ABS Tranches
* <details> 
     <summary>Click to Expand ABS Summary</summary>
      <img width="720" alt="Screenshot 2024-12-17 at 12 18 18 AM" src="https://github.com/user-attachments/assets/825cb339-ecff-44b7-8862-2a491fd43a97" />
     <summary>Click to Expand ABS Details</summary>
      <img width="1063" alt="Screenshot 2024-12-17 at 12 19 10 AM" src="https://github.com/user-attachments/assets/1b58c631-2e39-41a5-a13b-974113d2dd6c" />
      

## Reference
Chang, A. H., Yang, L. K., Tsaih, R. H., & Lin, S. K. (2022). Machine learning and artificial neural networks to construct P2P lending credit-scoring model: A case using Lending Club data. Quantitative Finance and Economics, 6(2), 303-325.​

Balyuk, Tetyana and Davydenko, Sergei, Reintermediation in FinTech: Evidence from Online Lending (January 20, 2023). Michael J. Brennan Irish Finance Working Paper Series Research Paper No. 18-17, 31st Australasian Finance and Banking Conference 2018, Available at SSRN: https://ssrn.com/abstract=3189236 or http://dx.doi.org/10.2139/ssrn.3189236





