import sys
import math
import numpy as np
from prettytable import PrettyTable

# 	This recommender system is based on KNN by using item-item collaborative filtering. I will outline the design and core
# 	funtion below:
# 
# 	Design:
# 		Step1: Initialise movie_user matrix
# 		Step2: Implement similarity metric equation
# 		Step3: Predict the rate of the movie by the specified user
# 		Step4: Compute top-N recommned movie list
# 		Step5: Compute the RMSE and MAE for the training dataset
# 
# 	Function:
# 		1. Display the movies the user rated
# 		2. Predict the rate of the specified movie by the specified user
# 		3. Recommend movie list for the specified user
# 		4. Evaluate system for the specified dataset


# Initialise the movie-user matrix
def init(path):
	file = open(path, 'r')
	userSet = set()
	# {tid: [(uid, rate), (),,,], ...}
	movieDict = {}
	lines = file.readlines()
	for line in lines:
		line_arr = line.split("\t")
		userSet.add(line_arr[0])

		if int(line_arr[1]) not in movieDict:
			movieDict[int(line_arr[1])] = [(int(line_arr[0]), int(line_arr[2]))]
		else:
			movieDict[int(line_arr[1])].append((int(line_arr[0]), int(line_arr[2])))

	matrix = np.zeros((1682, 943))

	for row in movieDict:
		for column in movieDict[row]:
			matrix[row - 1][column[0] - 1] = column[1]

	simMatrix = np.copy(matrix)
	sumArray = np.sum(matrix, axis = 1)

	avgArray = []

	for row in range(len(matrix)):
		if sumArray[row] != 0:
			avgArray.append(sumArray[row] / np.count_nonzero(matrix[row]))
		else:
			avgArray.append(0)

	for row in range(1682):
		for column in range(943):
			if simMatrix[row][column] != 0:
				simMatrix[row][column] -= avgArray[row]

	return matrix, simMatrix


# compute the similarity of two vector
def getSimilarity(destId, neighbor):

	denominator = np.sqrt(np.sum(destId ** 2) * np.sum(neighbor ** 2))
	if denominator == 0:
		return -1;
	return np.dot(destId, neighbor) / denominator


# find the top-N nearest neighbours
def getNeighbors(simMatrix, userId, destId, k):

	similarityList = []      
	## Note that, optimal part, which only compute similarity of the
	## movies which the user already rated
	###################Optimal Part#######################################
	for movieId in np.where(simMatrix[:, userId] != 0)[0]:
		if movieId == destId:
			continue
		similarityList.append([getSimilarity(simMatrix[destId], simMatrix[movieId]), movieId])

	similarityList.sort(reverse = True)

	if len(similarityList) > k:
		return similarityList[:k]
	else:
		return similarityList


# predict the rate of a movie specified by a user
def predictRate(matrix, simMatrix, userId, movieId, k = 15, cosSim = True):

	# [[0.58703950856427434, 5], [0.41403933560541256, 2]]
	if cosSim:
		neighbors = getNeighbors(matrix, userId, movieId, k)
	else:
		neighbors = getNeighbors(simMatrix, userId, movieId, k)

	# sum of weight * rate
	numerator = 0
	for neighbor in neighbors:
		numerator += neighbor[0] * matrix[neighbor[1]][userId]

	# sum of k nearest weights
	denominator = np.sum(neighbors, axis = 0)[0]

	if denominator == 0:
		return 0

	return numerator / denominator

# initialise test file
def initTest(path):

	user_dict = {}
	file = open(path, 'r')
	testData = file.readlines()

	for line in testData:
		line_arr = line.split("\t")
		userId = int(line_arr[0])
		pair = (int(line_arr[1]), int(line_arr[2]))

		if userId not in user_dict:
			user_dict[userId] = [pair]
		else:
			user_dict[userId].append(pair)

	return user_dict

# recommend the top-N movie list
# idea: predict all the movies the user not rated
#       sort them based on descending order, then get top n
def newRecommend(matrix, simMatrix, userId, N):

	ratingList = []
	returnList = []
	predictMovies = np.where(matrix[:, userId] == 0)[0]

	for predictMovie in predictMovies:
		ratingList.append([predictRate(matrix, simMatrix, userId, predictMovie), predictMovie])
	ratingList.sort(reverse = True)

	for i in range(N):
		returnList.append([ratingList[i][1], ratingList[i][0]])

	return returnList
	
# initialise movie file
def initMovie(path):

	movies = {}
	for movie in open(path, encoding='utf-8'):
		movie_arr = movie.split('|')
		movies[int(movie_arr[0])] = (movie_arr[1], movie_arr[2])

	return movies

# output the recommend movie in the table form
def recommend(matrix, simMatrix, userId, k, movies):
	# [movieId, movieId...]
	recommendList = newRecommend(matrix, simMatrix, userId, k)

	table = PrettyTable(["MovieId", "Movie Title", "Release", "Predict Rate"])
	table.align["MovieId"] = "l" # Left align city names
	table.padding_width = 5

	# Note that the key is the index of movie!!
	for key in recommendList:
		table.add_row([key[0] + 1, movies[key[0] + 1][0], movies[key[0] + 1][1], key[1]])
	print(table)

# compute the MAE and RMSE to measure the training dataset
def getEvaluation(user_dict, matrix, simMatrix, K, Cosine = True):

	count = 0
	allMaeErr = 0
	allRmseErr = 0

	print("Runing... Please wait about 1m...")
	for userId in user_dict:
		for movie in user_dict[userId]:
			rate = predictRate(matrix, simMatrix, userId - 1, movie[0] - 1, K, Cosine)
			allMaeErr += abs(movie[1] - rate)
			allRmseErr += (movie[1] - rate) * (movie[1] - rate)
			count += 1

	print("Err: ", allMaeErr, " count: ", count)
	MAE = allMaeErr / count
	RMSE = math.sqrt(allRmseErr / count)
	print("MAE: ", MAE, "  RMSE: ", RMSE)


# Note that both userId and movieId start from 0
# output the user rated record in the table form
def printUserRate(userId, matrix, movies):
	table = PrettyTable(["MovieId", "Movie Title", "Rate"])
	table.align["MovieId"] = "l" # Left align city names
	table.padding_width = 5

	for movieId in np.where(matrix[:, userId] != 0)[0]:
		table.add_row([movieId + 1, movies[movieId + 1][0], matrix[movieId][userId]])

	print(table)

#  simple UI
def printUI():
	print("~~~~~~~~~~~~~~~Welcome to Recommender System based on MovieLen 100k~~~~~~~~~~~~~~~")
	print("Action Menu: ")
	print("display 		Display user rates")
	print("predict 		Predict user's rating on movie")
	print("recommend 		Recommend List for user")
	print("evaluate		Evaluate Recommender System")
	print("quit			Quit")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# main function
def main():
	matrix, simMatrix = init("ml-100k/u.data")
	movies = initMovie("ml-100k/u.item")

	print("Initialise Success...")
	printUI()

	while(1):

		action = input("Please input action key:   (? for menu)\n")

		if action == '?':
			printUI()
		elif action == "display":
			userId = input("Please input userId:\n")
			printUserRate(int(userId) - 1, matrix, movies)

		elif action == "predict":
			userId, movieId = input("Please input userId and predict MovieId separated by space:\n").split()
			rate = predictRate(matrix, simMatrix, int(userId) - 1, int(movieId) - 1)

			print("Predict Rate: ", rate)

		elif action == "recommend":
			userId, N = input("Please input userId and topN of recommend list separated by space:\n").split()
			recommend(matrix, simMatrix, int(userId) - 1, int(N), movies)

		elif action == "evaluate":
			print("training SetId: [1|2|3|4|5]\t with Cosine similarity")

			train, n = input("Please input training setId, neigbours\n").split()
			trainPath = "ml-100k/u" + train + ".base"
			testPath = "ml-100k/u" + train + ".test"
			evaluateMatrix, evaluateAvgMatrix = init(trainPath)
			user_dict = initTest(testPath)

			getEvaluation(user_dict, evaluateMatrix, evaluateAvgMatrix, int(n), True)

		elif action == "quit":
			break

		else:
			print("action key error...")
			continue			

if __name__ == '__main__':
	main()



