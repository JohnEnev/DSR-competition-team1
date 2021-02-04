# DSR Mini-Competition

**Team 1: John Enevoldsen, Sara Ghasemi, Mena Nasr**

# Setup



## Clone this repository:
`git clone https://github.com/JohnEnev/DSR-competition-team1.git`

## `cd` into the directory
`cd DSR-competition-team1`

## Install the required packages:
`pip install -r requirements.txt`

## Open the jupyter notebook
`jupyter notebook`


## Run the model notebook `final_model.ipynb`

Run each cell one by one.
Result will be in the last cell

# Thought process

We first performed exploratory data anlysis by determining how the data was organized and what variables likely accounted for predictability in sales. We found that there were siginificant portions of null values in the columns- "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceYear","PromoInterval". 
We performed a correlation matrix which showed negative correlation with sales with how far we were in the week. That is that later days in the week oproduced a drop in sales with Monday having the highest sales. We also found inuitively that sales correlated with Promotions, and to a much lesser degree School Holoidays as well. 
Next we dropped columns like Customers, and rows where there were no sales. 
Next we removed the rest of the null values  and ultimately we were left with around a little over 400,000 rows from over 600,000 initially. 
After cleaning we split the data into a training set which contained data from all of 2013  up until 01/05/2014, and a test set which ran from 01/05/2014-until July 2014. We then mean encoded the data frame, so that we could later run our models. 
To establish a baseline for us to evaluate against, we established an RMSPE of 31.68% Next we used a random forest model with 100 trees, and a maximum depth of 5, where we obtained an error of 22.59%. 
The next model we ran was the XGBoost regressor (with default settings in python) which further reduced the error to 20.5%. The third model we ran was the multivariate regression which had an error of 23.16%. All three model beat the baseline by approximately 10 percentage points.
