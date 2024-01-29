# Introduction


In the ever-evolving automotive industry landscape various types of cars are launched every year but the intrinsic features of a car remains the same and acts as a core factor when it's up for sale. Moreover as technology keeps on improving and new features get added it has become quite normal for people to move on to a better car model by selling their old model. So in order for car dealerships to evaluate whether these second hand car models that they are being offered will lead them to profits in future, we are aiming to assist them at classifying the condition of these cars by using various machine learning algorithms. Here we are using the intrinsic features and other crucial information about a car as datasets to train and evaluate machine learning models to come to a decisive profitable conclusion so that the business doesn't get misled by fraud attempts.    


# Dataset description


Link: https://archive.ics.uci.edu/dataset/19/car+evaluation

The Dataset we are using is a car evaluation dataset that consists of 6 features and 1 label column. The feature columns are as follows: 'buying' as in the buying price of the car, 'maint' for price of the maintenance, 'doors' that consist of number of doors in a vehicle, 'persons' as the number of persons a vehicle can fit, luggage capacity as 'lug_boot' and 'safety' as estimated safety of the car. The target or the label column is called 'class' which consists of an evaluation level (unacceptable, acceptable, good, very good).

To reach our goal we are going to use popular classification models as our focus is to determine in what category of the evaluation level the offered second hand vehicles fall in. There are 1727 instances and 7 columns and each column consists of sub categorical values. 

Below is the correlation heatmap of our dataset.  


![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/260f7605-4e06-491a-9ba7-ee0eed14bf18)

















The target feature does have some degree of imbalance to it as sub categories such as the car being unacceptable is present in larger portions compared to other sub categories especially 'good' and 'vgood' are present in smaller quantities.


![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/0bd76eb7-7e09-4a9f-a910-8daa4cca29d3)  


  














# Dataset Pre-Processing:


The Dataset initially provided  had no null values so we altered it by implanting some null values in random column sub categories. In order to fix these null valued instances or columns we simply discarded them as all of our sub categories are categorical values so there was no scope of applying imputation methods involving mean or median.

As all of our features and target variables have sub categories with categorical values we had to encode them in two ways. Some categorical sub-categories were ordinal and some were nominal. Among the ordinals were 'maint', 'buying', 'safety', 'lug_boot' and also our target called 'class'. We determined them as Ordinal as each of them have sub-categories that follow rankings which impact the human perception when the vehicle is offered for sale. For this reason we have applied Ordinal encoding technique where we were able to assign a numerical value ranked following the sub features importance. From the figure below we can see what numbers we have assigned to which sub categories  

### Before Ordinal Encoding
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/8f0a5a51-e522-49ba-8605-11b6336b0b6a)


### After Ordinal Encoding
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/cd87d032-174c-4341-a67b-83aff3d03037)


Otherwise features such as 'doors' and 'persons' have sub categories that are mutually exclusive so we have used One-Hot encoding for them. In one-hot encoding the sub categories themselves are added as unique columns in the dataframe and they consist of binary 0 or 1 corresponding to the category of the original feature. So after applying this technique to  'doors' and 'persons' the figure below shows the result.  

![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/deddeae0-d529-4126-abd5-aaef66117637)







## Feature Scaling:

As all the features in our dataset are categorical we did not have to apply feature scaling

## Feature Selection:

![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/4ff73bcc-dc6b-4663-ba20-84b97bd6950d)


From the correlation heatmap above we can see that the correlation of all the doors features (“doors_2”, ”doors_3”, “door_4”, “doors_5more”) with the target “class” is -0.069068, 0.002992, 0.033404, 0.032678 respectively. Since they show very weak correlations, we have decided to drop “doors_2”, ”doors_3”, “door_4”, “doors_5more” columns from our dataframe.



# Model used for training & testing and Dataset Splitting:


- Logistic regression
- Support Vector machine
- Decision tree

The original dataset has been divided into two parts: training & testing. We are using 70% of the original dataset as our train set and 30% of it has been considered as our test set which is used to evaluate the performance of our trained model. 

Few things that we need to be clear about is that the instances that have been selected for the train set and test set are that they have not been selected randomly. However to fix the imbalance in our target feature we used stratify on that specific column so that the process of creating y_train and y_test uses optimal ratios of the sub categories present which makes the overall process much more consistent. When evaluating the models we will prioritize precision and recall over accuracy.


# Model Comparison Analysis:

| Model Name          | Accuracy | Precision | Recall | F1 score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.88     | 0.89      | 0.88   | 0.88     |
| Support Vector Machine | 0.93     | 0.93      | 0.93   | 0.93     |
| Random Forest       | 0.94     | 0.95      | 0.95   | 0.95     |



Here we can see the overall performance of our classification  models. The logistic Regression model is the only model with an accuracy of 0.88 whereas the other two exceed 0.90. Precision and recall are our main priority of assessing so that we can address the errors that can arise due to the imbalance in our target feature and also the higher the f1 score of the model the more balanced the precision and recall are for that specific model. According to our table we see that random forest performs the best in all aspects so if we have to finalize on one model then we will choose random forest. 



# Confusion Matrices:


- ### Logistic regression
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/9ba1e573-bf80-4a93-bac9-3d7654a72741)


- ### Random forest
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/6aad2e15-ac89-4eb7-9fc2-ad8655b0cbfe)

- ### SVM
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/e922a68f-73c8-4991-babb-9daed49d44f9)


## Interpretation:
![image](https://github.com/NafisAshraf/CarEvaluationModel/assets/134098048/2787d51e-26eb-400c-9a23-7054e37d9046)

Since we are making a model for car dealerships that buy used cars, our main concern is the False positive scenarios because if a car which is actually unacceptable gets predicted as acceptable/ very good/ good it can be very detrimental for the business. Here the recall value shows that among all the cars that were actually unacceptable how many were correctly predicted to be unacceptable.

# Conclusion:

Overall we can see that each classification model somewhat performs to a degree of accuracy where in a real life situation it is possible to obtain an optimal output.
