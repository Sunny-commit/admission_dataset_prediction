🎓 Admission Prediction Based on Interest Levels
This project uses a dataset of university admissions to predict a student's Level of Interest using multiple academic and demographic features. The goal is to help analyze factors that influence how interested a student is in a particular university or program.

📁 Dataset
The dataset (Admission Data.csv) contains the following features (excluding university names):

GRE Score

TOEFL Score

SOP (Statement of Purpose)

LOR (Letter of Recommendation)

CGPA

Research Experience

Level of Interest (Target)

🧹 The column "Name of University" was removed for being non-numeric and less relevant to interest level prediction.

📊 Exploratory Data Analysis (EDA)
Summary statistics and data types inspection using .describe() and .info()

Distribution plots for all features using Seaborn to visualize how values are spread

Missing values check confirms data completeness

🧠 Machine Learning Model
The script uses a Linear Regression model from scikit-learn to predict the Level of Interest.

Pipeline:

Preprocessing

Feature-target split

Train-test split (80/20)

Training

LinearRegression() model trained on the dataset

Evaluation

Mean Squared Error (MSE)

R² Score (Goodness of fit)

🧪 Results
Example output from evaluation metrics:

yaml
Copy
Edit
Mean Square Error: 0.024
R2 Score: 0.88
Interpretation: The model performs reasonably well in predicting interest levels.

🛠️ Technologies Used
Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

▶️ How to Run
Ensure you have Admission Data.csv in the same directory or update the path.

Run admission_dataset.py in a Python environment or Google Colab.

Review the visualizations and model predictions.
