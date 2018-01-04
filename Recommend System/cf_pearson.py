import math
import matplotlib.pyplot as plt

item_path = "data/u.item"
train_path = ["data/u1.base","data/u2.base","data/u3.base","data/u4.base","data/u5.base"]
test_path = ["data/u1.test","data/u2.test","data/u3.test","data/u4.test","data/u5.test"]
evaluation_path = [("data/u1.base","data/u1.test"),("data/u2.base","data/u2.test"),
					("data/u3.base","data/u3.test"),("data/u4.base","data/u4.test"),
					("data/u5.base","data/u5.test")]
evaluation_result = []

class rcsys:
	def __init__(self, k, item_file, train_file, test_file):
		self.item_file = item_file
		self.train_file = train_file
		self.test_file = test_file
		self.k = k
		self.get_data()

	def set_k_value(self, k):
		self.k = k
		print("k is set to ")
		print(k)

	def get_data(self):
		self.item_dic = {}
		for line in open(self.item_file):
			if line.strip():
				movie_id, movie_name = line.split("|")[0:2]
				self.item_dic[movie_id] = movie_name
		self.train_data = {}
		for line in open(self.train_file):
			user_id, movie_id ,movie_rating = line.split("\t")[0:3]
			self.train_data.setdefault(user_id, {})
			self.train_data[user_id][movie_id] = float(movie_rating)
		self.test_data = {}
		for line in open(self.test_file):
			user_id, movie_id ,movie_rating = line.split("\t")[0:3]
			self.test_data[(user_id,movie_id)] = float(movie_rating)

	def get_similarity(self):
		trans_dic = {}
		for user_id in self.train_data:
			for movie_id in self.train_data[user_id]:
				trans_dic.setdefault(movie_id,{})
				trans_dic[movie_id][user_id] = self.train_data[user_id][movie_id]
		#print("Now transfer the data")
		#print(trans_dic)
		list_item = list(trans_dic)
		list_item.sort()
		#print(list_item)
		self.sim_data = {}
		for i in range(len(list_item)):
			for j in range(i + 1, len(list_item)):
				move_A = list_item[i]
				move_B = list_item[j]
				#print(move_A)
				#print(move_B)
				value = self.similarity(trans_dic,move_A,move_B)
				#print(value)
				self.sim_data[(move_A, move_B)] = value
		#print(self.sim_data)
	# use pearson to get similarity
	def similarity(self, data, movie1, movie2):
		common_movies = {}
		for movie in data[movie1]:
			if movie in data[movie2]:
				common_movies[movie] = 1

		length = len(common_movies)
		if length == 0: return 0

		sum1 = sum(data[movie1][movie] for movie in common_movies)
		sum2 = sum(data[movie2][movie] for movie in common_movies)

		sum1_of_squares = sum([pow(data[movie1][movie], 2) for movie in common_movies])
		sum2_of_squares = sum([pow(data[movie2][movie], 2) for movie in common_movies])

		sum_of_cross = sum([data[movie1][movie] * data[movie2][movie] for movie in common_movies])

		num = sum_of_cross - (sum1 * sum2 / length)
		den = math.sqrt((sum1_of_squares - pow(sum1, 2) / length) * (sum2_of_squares - pow(sum2, 2) / length))

		if den == 0: return 0
		result = num / den

		if result > 1:
			result = 1.0
		if result < -1:
			result = -1.0
		result = round((result + 1) / 2, 4)
		return result
	def predict_movie_rating(self, user_id, movie_id):
		score = 0.0
		sumofsimilarity = 0.0
		movie_rating = self.train_data[user_id]
		if movie_id in movie_rating.keys():
			return movie_rating[movie_id]

		similarity_and_rating = []
		for (movie, rating) in movie_rating.items():
			pair = (movie, movie_id)
			if (pair not in self.sim_data):
				pair = (movie_id, movie)
			if (pair not in self.sim_data):
				continue
			similarity = self.sim_data[pair]
			similarity_and_rating.append((similarity, rating))

		similarity_and_rating.sort(reverse = True)
		if self.k != None:
			movie_rating = similarity_and_rating[0:self.k]
		for (similarity, rating) in movie_rating:
			score += similarity * rating
			sumofsimilarity += similarity
		if sumofsimilarity == 0.0:
			return 0.0
		return score / sumofsimilarity
	def recommend_movies(self, user_id):
		recommend_movies = []
		rated_movies = set(self.train_data[user_id].keys())
		unknown_rated_movies = set(self.item_dic.keys()) - rated_movies
		for movie_id in unknown_rated_movies:
			predicted_rating = self.predict_movie_rating(user_id,movie_id)
			recommend_movies.append((movie_id, predicted_rating))
		# make the list sorted by descending order
		recommend_movies.sort(reverse = True)
		# return the top-k recommended movies
		return recommend_movies[0:self.k]

	def compute_MSE_and_RMSE(self):
		error_list = []
		result = []
		for line in self.test_data:
			rating = self.test_data[line]
			predict_rating = self.predict_movie_rating(line[0],line[1])
			if predict_rating == 0:
				continue
			error = abs(predict_rating - rating)
			error_list.append(error)
		mean_square_error = sum(error_list) / len(error_list)
		root_mean_square_error = (sum(error ** 2 for error in error_list) /len(error_list))
		#print(mean_square_error)
		#print(root_mean_square_error)
		evaluation_result.append((mean_square_error, root_mean_square_error))
		print(evaluation_result)
		# return evaluation_result

def find_best_k_value(obj: rcsys):
	print("find_best_k_value")
	obj.get_similarity()
	obj.compute_MSE_and_RMSE()

def demo_mode():
	print("demo_mode enabled...")
	x = [10,15,20,30,40,50,60,70,80,90,100]
	y_axis_1 = []
	y_axis_2 = []
	y_axis_3 = []
	y_axis_4 = []
	y_axis_5 = []
	x_axis = []
	count = 1
	for i, j in evaluation_path:
		print("Now use the ")
		print(count)
		print(" training data")
		print(i)
		print(j)
		count += 1
		rc_sys = rcsys(k = 10, item_file = item_path, train_file = i, test_file = j)
		for k in x:
			print("Now the k is set to " + str(k))
			rc_sys.set_k_value(k)
			find_best_k_value(rc_sys)
			print(evaluation_result)
	plt.plot(x,evaluation_result)
	plt.xlabel("k")
	plt.ylabel("RMSE")
	plt.show()
	return 0

def show_recommended_movies(obj: rcsys):
	user_id = input("Give a user id, system will recommend some movies related to movies the user once watched")
	obj.get_similarity()
	recommended_movie_id_list = obj.recommend_movies(user_id)
	for (movie_id,predicted_rating) in recommended_movie_id_list:
		print(obj.item_dic[movie_id])
def normal_mode():
	print("normal_mode enabled...")
	rc_sys = rcsys(k = 50, item_file = item_path, train_file = train_path[0], test_file = test_path[0])
	show_recommended_movies(rc_sys)
def visualize_statistics():
	result = [(0.8992928337169805, 1.2689363972850065),
	(0.8978282345399572, 1.2704198090573593),
	(0.8794279835210467, 1.2131669758186274),
	(0.8830621292599697, 1.2241403857101534),
	(0.8766608829742324, 1.2078947223734502),
	(0.8716431117164675, 1.1880382581724527),
	(0.8686383583685856, 1.1848477944884026),
	(0.8502304298865849, 1.1325811675060153),
	(0.8557865277229795, 1.1446964442006022),
	(0.8528019239470244, 1.1373433370046824),
	(0.8566795389336412, 1.1475688336432983),
	(0.8533793555696216, 1.1398210456482285),
	(0.8351895022191397, 1.091622743675932),
	(0.8424939838084662, 1.1075585304381121),
	(0.8398683760160415, 1.1004407894112824),
	(0.8458108118813389, 1.1178726656773628),
	(0.8383228580398384, 1.099549945552994),
	(0.8220152707878309, 1.0571598230376262),
	(0.8299315668043739, 1.0734354709596692),
	(0.8282894862956731, 1.0681358596513737),
	(0.8414754453933853, 1.1086971595494424),
	(0.8310836181939552, 1.0827102755714666),
	(0.8168435210482196, 1.0444648328847173),
	(0.8241126982012116, 1.0591620111502966),
	(0.8240283347586528, 1.055710886803687),
	# (0.8394732149808556, 1.105665111664213),
	# (0.8280254809506606, 1.0755253415534725),
	# (0.8146364631492428, 1.0394171968461334),
	# (0.8222084986177903, 1.054222206966663),
	# (0.8222731155306221, 1.0508615821199614),
	(0.8387556204530909, 1.1041568907032229),
	(0.826767344096708, 1.072584763196778),
	(0.8134475572771839, 1.0372229286713575),
	(0.82054831588902, 1.0512466782862477),
	(0.8213333998515097, 1.0490967441262986),
	(0.8387952707578847, 1.104879230510297),
	(0.8266888433903614, 1.0724525818852788),
	(0.812549373580395, 1.0359090001097724),
	(0.8201830034401725, 1.0503299997673654),
	(0.8219923843130981, 1.0507193978781197),
	(0.8395920904415035, 1.1069234459543476),
	(0.8274359268441165, 1.0738597597147295),
	(0.8122508471786081, 1.0355259774737753),
	(0.8202184181598782, 1.0508642845625622),
	(0.8223891781822339, 1.051909331008799),
	(0.8401804243180429, 1.1083093342256567),
	(0.8277576553925935, 1.0744918843432918),
	(0.8128538484358467, 1.0371788004719134),
	(0.8204499408944187, 1.0516505716362847),
	(0.8228898586447756, 1.0532669172318903),
	(0.8406722701776749, 1.1095192499218531),
	(0.8282932555203154, 1.0754791766299188),
	(0.8134545899365401, 1.0384363472111218),
	(0.8206725391463102, 1.052631146880339),
	(0.8235320105856647, 1.054768257351579)]
	x_axis = [10,15,20,30,40,50,60,70,80,90,100]
	y_axis = []
	y_axis_mse = []
	y_axis_rmse = []
	count = 0
	MSE = 0
	RMSE = 0
	for i, j in result:
		MSE += i
		RMSE += j
		count += 1
		if count == 5:
			M_MSE = MSE/5
			M_RMSE = RMSE/5
			y_axis.append((M_MSE,M_RMSE))
			y_axis_mse.append(M_MSE)
			y_axis_rmse.append(M_RMSE)
			print("m_MAE")
			print(MSE)
			print("m_RMSE")
			print(RMSE)	
			MSE = 0
			RMSE = 0
			count = 0
	print(y_axis_mse)
	print(y_axis_rmse)
	plt.plot(x_axis,y_axis)
	plt.xlabel("K")
	plt.ylabel("MAE and RMSE using Pearson")
	plt.show()
	plt.plot(x_axis,y_axis_mse)
	plt.xlabel("K")
	plt.ylabel("MAE using Pearson")
	plt.show()
	plt.plot(x_axis,y_axis_rmse)
	plt.xlabel("K")
	plt.ylabel("RMSE using Pearson")
	plt.show()
	return 0

if __name__ == '__main__':
	mode = int(input("Choose 1: demo mode 2: normal mode 3: visualize statistics" + "\n"))
	if(mode == 1):
		demo_mode()
	elif(mode == 2):
		normal_mode()
	elif(mode == 3):
		visualize_statistics()
	exit(0)