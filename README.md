# Predicting Insurance Charges

An insurance company that provides affordable health insurance to customers all across the USA. As the lead data scientist at the company, your responsibility is to create an automated system that calculates the annual medical expenditure for new customers. The system will use various pieces of information such as age, sex, BMI, number of children, smoking habits and region of residence to estimate the medical expenditure. The estimate provided by the system will be used to determine the annual insurance premium offered to the customer. It is important to be able to explain why the system outputs a certain prediction due to regulatory requirements.

The objectives are two folds:

1. Predict the annual expenses of the insurance to each customer (Prediction)
2. Explain the reasoning behind the prediction (interpretability of the model)

## Environment

You can reproduce my local environment using the spec-file.txt or enviroment.yaml files with conda using the command:
`conda create --name myenv --file spec-file.txt` or `conda env create -f environment.yaml`

## Steps:

### Data Acquisition

The data was taken from [download CSV file here](https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv). It is a labeled data of $1337$ customers. 

<figure>
<img src="summary.png" alt="data summary"/>
<figure-caption>Figure 1. Summary of the aata and its type.</figure-caption>
</figure>

The

### Exploratory Data Analysis (EDA)

to visualize and understand the data
