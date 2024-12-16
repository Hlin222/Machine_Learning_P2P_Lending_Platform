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





