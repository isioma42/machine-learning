# machine-learning
Northeastern University ALY6140 72241 Analytics Systems Technology 

### Vancouver Crime Prediction -ISOMA IDEMUDIA

_**From Crime Data to Classification with Gradient Boosted Trees: Supervised Machine Learning on Historical Data**_

---

The prediction of crimes is an important application of machine learning for societal benefit, but can also lead to dangerous biases. In this example, we demonstrate the first step in using historical data to predict vehicle theft based on 3 factors (features): Location, time of day, and month of the year. 

## Background
The Vancouver, British Columbia public dataset contains a variety of data for analysis of city activities and conditions. 
* http://data.vancouver.ca/datacatalogue/crime-data-details.htm
    
In addition, i need Vancouver census data to calculate the crime *rate* in each neighborhood to normalize our predictions to be independent of population. 
* http://vancouver.ca/your-government/2001---2011-census-local-area-profiles.aspx

I will train and test the model on the baseline data (vehicle thefts in 2006), and then test the models predict at a future time (2008).

