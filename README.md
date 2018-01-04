# Recommend-System

This is an off-line Recommend System by using collaborative filtering and dataset of moiveLen 100.

I implement two version, one is based on KNN, and the other is based on gradient descent. Both of them are using person similarity as the criterion.

Then I use MSE and MAE to meassure the performance. And find the recommend system combined with KNN and pearson would get better evaluation.

For more infomation about the whole project and summary, please see README, and report.pdf.

Just a simply introduction

Living in the time of information explosion, almost all of us would suffered how to make make good choices based on tons of information. The recommender system would help us especially when we do the shopping, watching, listening on line. In general, there are three designs for recommender system, content-based, knowledge-based and collaborative. The goal of this project is build the movie recommender system by using collaborative filtering algorithm. Collaborative filtering(item) is a method which can make the prediction for a user based on the historical preferences of this user. The specific steps are shown below, I implement it by using the item-based collaborative filtering based on knn model, and evaluate this system with MAE and RMSE
