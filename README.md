# Machine-learning
Northeastern University, College of Proffessional Studies- Enterprise Artificial intelligence- ALY6140 72241 Analytics Systems Technology 

### Vancouver Crime Prediction -ISOMA IDEMUDIA

_**From Crime Data to Classification with Gradient Boosted Trees: Supervised Machine Learning on Historical Data**_
The prediction of crimes is an important application of machine learning for societal benefit, but can also lead to dangerous biases. In this example, we demonstrate the first step in using historical data to predict vehicle theft based on 3 factors (features): Location, time of day, and month of the year. 

## Background
The Vancouver, British Columbia public dataset contains a variety of data for analysis of city activities and conditions. 
* http://data.vancouver.ca/datacatalogue/crime-data-details.htm
    
In addition, i need Vancouver census data to calculate the crime *rate* in each neighborhood to normalize our predictions to be independent of population. 
* http://vancouver.ca/your-government/2001---2011-census-local-area-profiles.aspx

I will train and test the model on the baseline data (vehicle thefts in 2006), and then test the models predict at a future time (2008).

from sklearn.metrics import roc_curve, auc

colors = {
    'Multinomial Naive Bayes' : 'aqua',
    'Logistic Regression'     : 'cornflowerblue',
    'Random Forest'           : 'darkorange',
    'SVC'                     : 'green',
    'Majority Voting'         : 'red',
}

for model_name, model in classifiers.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    #y_preds = model.predict(X_test)
    y_preds = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test.values.ravel(), y_preds)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr,
             tpr,
             color=colors[model_name], lw=2,
             label='ROC curve of model {0} (area = {1:0.2f})'.format(model_name, roc_auc)
            )
    
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve - models comparison')
plt.legend(loc="lower right")
plt.show()
