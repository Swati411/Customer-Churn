# Customer-Churn
An end-to-end machine learning project built to predict customer churn in the telecom sector. This repository covers everything from rigorous exploratory data analysis (EDA) and data cleaning to handling class imbalance, training multiple classification models, hyperparameter tuning, and cross-validation.

🚀 Project Overview
Customer churn is a critical challenge for subscription-based businesses like telecom, banking, and SaaS. Losing customers directly impacts recurring revenue and drives up customer acquisition costs. The goal of this project is to analyze historical customer behavior, uncover primary churn indicators, and build a robust classification pipeline to identify customers at risk of leaving.   

📂 Dataset & Features
The dataset contains 7,043 rows and 21 columns covering demographic details, account info, and service subscriptions:
Demographics: gender, SeniorCitizen, Partner, Dependents

Account Services: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, Total_Charges

Telecom Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Target Variable: Churn (Yes/No)   

🛠️ Methodology & Workflow
Data Cleaning & Preprocessing:

Handled missing values using appropriate strategies based on data distributions (e.g., mode filling for categorical features like gender and PaperlessBilling, mean for normally distributed MonthlyCharges, and median for right-skewed Total_Charges).   

Converted data types (type casting TotalCharges into numeric float values).   

Checked for duplicates (none found).   

Exploratory Data Analysis (EDA):

Univariate Analysis: Explored distributions of numerical features (tenure, MonthlyCharges, Total_Charges) and categorical attributes (Contract, InternetService, PaymentMethod, etc.).   

Bivariate Analysis: Investigated relationships between features and churn, revealing that month-to-month contracts, lack of tech support/online security, and lower tenure strongly correlate with higher churn.   

Multivariate Analysis: Checked feature correlations using heatmaps.   

Feature Engineering & Encoding:

Applied Label Encoding for binary categorical features (gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn).   

Applied One-Hot Encoding (pd.get_dummies) for multi-category features like Contract, InternetService, and PaymentMethod.   


Scaled numerical features using StandardScaler.   

Handling Class Imbalance:

Observed target class imbalance via count plots.

Utilized SMOTE (Synthetic Minority Over-sampling Technique) on the training set to balance the classes before model training.   




🤖 Model Training & Evaluation
Multiple classification algorithms were trained, evaluated, and tuned using GridSearchCV and RandomizedSearchCV:

Model	Accuracy	Recall	F1 Score	Notes
Logistic Regression	
79.4% 

75.0% 


0.659 

Best overall balance; ROC-AUC: 0.858 


SVC	
79.1% - 79.4% 


70.0% - 72.0% 


0.64 - 0.65 


Performance closely matches Logistic Regression 


Gradient Boosting	
77.7% - 78.8% 

74.0% - 78.0% 

0.65 


Reached the highest recall (78%) with lower precision 


XGBoost	
78.3% - 78.6% 

63.0% - 69.0% 

0.61 - 0.63 

Stable performance across tuning parameters 


Random Forest	
77.9% - 78.3% 


61.0% - 76.0% 


0.60 - 0.65 


Balanced metrics after hyperparameter adjustment 


KNN (k=7)	
73.2% 


75.0%

0.596 

Suffers from lower precision (0.50) 

Decision Tree	
72.6% - 74.9% 

54.0% - 62.0% 

0.51 - 0.57 

Lowest relative performance among tested models 

Cross-Validation: Verified model stability using 5-Fold Cross-Validation and Stratified K-Fold Cross-Validation (averaging ~84% validation performance on XGBoost).   

💡 Key Business Insights
Tenure & Contract Matters: Customers on short-term month-to-month contracts with lower tenure are significantly more likely to churn. Securing customers into annual or two-year commitments heavily reduces churn risk.   


Support Services Help Retention: Customers lacking technical support or online security tools exhibit higher churn rates. Promoting these add-on services can aid customer retention.   

Model Selection Rationale: While boosting models capture high recall, Logistic Regression offered the best harmonic balance between precision, recall (75%), F1-score (0.659), and interpretability—making it an ideal, practical choice for business deployment.
