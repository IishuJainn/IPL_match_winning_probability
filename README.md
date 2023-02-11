# IPL_mach_winning_probability
A project that will tell the winning probability of the chasing team based on the given Target, Wickets down, Over finished, Venue, Chasing team and the Defending team.

https://user-images.githubusercontent.com/102272183/215305092-811c7f0e-64de-410b-bbd2-3abaaec1bd6d.mp4

# IPL-Match-Innings-Result-Prediction
This is a Machine Learning model to predict the result of a match in the Indian Premier League (IPL) based on various features at the end of the second innings.

## Prerequisites

pandas

numpy

sklearn

## Files

model.py contains the main code for building and evaluating the Machine Learning model.

matches.csv contains information about all the matches in the IPL.

deliveries.csv contains ball-by-ball information about all the matches in the IPL.

## Data Processing
The code in model.py reads in the matches.csv and deliveries.csv files, performs data cleaning and preparation, and calculates various features that are used as input to the Machine Learning model. The following steps are performed:

Merging the matches.csv and deliveries.csv files on the match ID.

Removing all matches where the Duckworth-Lewis method was applied.

Removing any matches where the team names have changed over time (e.g., Delhi Daredevils became Delhi Capitals).

Calculating the number of balls left, the number of runs left, the number of wickets left, and the current run rate (CRR) and required run rate (RRR) at the end of the second innings.
Encoding the categorical variables (batting team, bowling team, and city) using one-hot encoding.

## Model Building
The code then uses the processed data to build a Random Forest Classifier model, which is trained and evaluated on a 80/20 train/test split of the data. The accuracy of the model is printed at the end.

## How to run
To run the code, simply execute the following command in your terminal:

![image](https://user-images.githubusercontent.com/102272183/218274809-3dbf23d2-d4c3-4977-a749-dae327236131.png)
