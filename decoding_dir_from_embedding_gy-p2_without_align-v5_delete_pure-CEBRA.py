##
import os

import cebra
from cebra import CEBRA
import numpy as np

import cebra.datasets
import torch
import random
import sklearn.metrics
from sklearn.ensemble import IsolationForest

torch.backends.cudnn.benchmark = True
print('\n')

data_from_to = '20221103_20230402_delete'
filepath = '/data/data_hly/Multiscale_SNN_Code/Data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##
# data loading
bin_method = 'expect_bins'
expect_num_bins = 50
print('bin_method: ', bin_method)
print('expect_num_bins: ', expect_num_bins)

if bin_method == 'expect_bins':
	all_micro_spikes_concat = torch.load(filepath + '/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
	all_macro_conditions_concat = torch.load(filepath + '/all_macro_conditions_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
	
	load_data_1 = np.load(filepath + '/len_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.npy', allow_pickle=True)
	len_for_each_session_trial = load_data_1.tolist()
	
	load_data_2 = np.load(filepath + '/target_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.npy', allow_pickle=True)
	target_for_each_session_trial = load_data_2.tolist()
	
	# 1-8 ==> 0-7
	all_macro_conditions_concat = all_macro_conditions_concat - torch.tensor(1)
	for i in range(len(target_for_each_session_trial)):
		target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]) - 1).tolist()

##
turns = 8
output_acc_align = []
# seed_set = np.random.randint(31,55,10)
# seed_set = np.arange(41, 51)
seed_set = np.array([31, 44, 46, 49, 50, 54, 55, 59])
# seed_set = np.unique(np.random.randint(41,101,50))[:30]
for turn in range(turns):
	# seeds = np.random.randint(1, 100, 4)
	# random.seed(int(seeds[0]))
	# np.random.seed(int(seeds[1]))
	# torch.manual_seed(int(seeds[2]))
	# torch.cuda.manual_seed_all(int(seeds[3]))
	
	seeds = int(seed_set[turn])
	# seeds = int(np.random.randint(31, 50, 1))
	random.seed(seeds)
	np.random.seed(seeds)
	torch.manual_seed(seeds)
	torch.cuda.manual_seed_all(seeds)
	
	torch.backends.cudnn.deterministic = True
	
	print('turn: ', turn)
	# today = '2023-11-28-v1-p2-delete-pureCEBRA-turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to
	today = '2023-11-28-v2-p2-delete-pureCEBRA-turn_' + str(turn)  # train:pre 4 weeks, test: next all; standard embedding: Day_to
	## split_data and Load the model for training
	max_iterations = 10000  # default is 5000.
	output_dimension = 36  # here, we set as a variable for hypothesis testing below.
	
	print('output_dimension: ', output_dimension)
	
	if bin_method == 'expect_bins':
		train_data = 'days'
		print('train_data: ', train_data)
		if train_data == 'days':
			if data_from_to == '20221103_20230402_delete':
				day = {'0': 0, '1': 5, '2': 10, '3': 13, '4': 18, '5': 24, '6': 30, '7': 36, '8': 40, '9': 42, '10': 46,
				       '11': 48, '12': 53, '13': 55, '14': 58, '15': 62, '16': 63, '17': 66, '18': 68, '19': 70,
				       '20': 71,
				       '21': 74, '22': 81, '23': 86, '24': 88, '25': 90, '26': 92, '27': 94, '28': 96, '29': 98,
				       '30': 100,
				       '31': 101, '32': 103, '33': 106, '34': 110, '35': 114, '36': 118, '37': 120, '38': 122,
				       '39': 124, '40': 128, '41': 132, '42': 134, '43': 136, '44': 138, '45': 140, '46': 142,
				       '47': 143, '48': 146, '49': 149, '50': 152}
				
				Day_from = 1
				Day_to = 12
				test_Day = 50
				print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
				print('--------------------test_Day: {}------------------'.format(test_Day))
				
				
				def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):
					
					split_idx_start_beg = 0
					for i in range(day[str(Day_from - 1)]):
						split_idx_start_beg += sum(len_for_each_session_trial[i])
					split_idx_start_end = 0
					for i in range(day[str(Day_to)]):
						split_idx_start_end += sum(len_for_each_session_trial[i])
					
					split_idx_start = 0
					for i in range(day[str(test_Day - 1)]):
						split_idx_start += sum(len_for_each_session_trial[i])
					split_idx_end = 0  # 只预测第test_Day这一天的
					for i in range(day[str(test_Day)]):
						split_idx_end += sum(len_for_each_session_trial[i])
					
					neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
					label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
					
					neural_test = data_spike_train[split_idx_start:split_idx_end, :]
					label_test = data_label[split_idx_start:split_idx_end, :]
					return neural_train, neural_test, label_train, label_test
	
	# split data
	neural_train, neural_test, label_train, label_test = split_data(all_micro_spikes_concat,
	                                                                all_macro_conditions_concat,
	                                                                len_for_each_session_trial, Day_from,
	                                                                Day_to)  # direction
	
	add_num = 0
	add_num_to = 0
	# neural_train_add, _, label_train_add, _ = split_data(all_micro_spikes_concat, all_macro_conditions_concat,
	#                                                      len_for_each_session_trial, Day_from + add_num,
	#                                                      Day_to + add_num_to)  # direction
	print(f"add_num={add_num},add_num_to={add_num_to}")
	
	# neural_train = torch.vstack([neural_train, neural_train_add])
	# label_train = torch.vstack([label_train, label_train_add])
	print('neural_train_length: ', len(neural_train))
	print('split data...finished!')
	
	distance = 'euclidean'
	# distance = 'cosine'
	print('distance: ', distance)
	# model
	cl_dir_model = CEBRA(model_architecture='offset10-model',
	                     batch_size=512,
	                     learning_rate=3e-4,
	                     temperature=1,
	                     output_dimension=output_dimension,
	                     max_iterations=max_iterations,
	                     distance=distance,
	                     device='cuda_if_available',
	                     verbose=True)
	# cl_dir_model = CEBRA(model_architecture='offset10-model-mse',
	#                      batch_size=512,
	#                      learning_rate=3e-4,
	#                      temperature=1,
	#                      output_dimension=output_dimension,
	#                      max_iterations=max_iterations,
	#                      distance=distance,
	#                      device='cuda_if_available',
	#                      verbose=True)
	##
	if bin_method == 'expect_bins':
		if train_data == 'days':
			if not os.path.exists(filepath + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(
					max_iterations) + '_GY-p2_acrossdays_expect_bin_' + str(
				expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(
				Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				Day_to + add_num_to) + '_' + today + '_' + data_from_to + '.pt'):
				cl_dir_model.fit(neural_train, label_train)
				
				cl_dir_model.save(filepath + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(
					max_iterations) + '_GY-p2_acrossdays_expect_bin_' + str(
					expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(
					Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_' + data_from_to + '.pt')
				print('save model ...... finished!')
	
	## ## Load the models and get the corresponding embeddings
	if bin_method == 'expect_bins':
		if train_data == 'days':
			cl_dir_model = cebra.CEBRA.load(
				filepath + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(
					max_iterations) + '_GY-p2_acrossdays_expect_bin_' + str(expect_num_bins) + '_Day_from_' + str(
					Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_' + data_from_to + '.pt')
			cebra_dir_train = cl_dir_model.transform(neural_train)
			cebra_dir_test = cl_dir_model.transform(neural_test)
	print('Load the model ..... finished!')
	

	##
	# 定义超参数
	sequence_length = 50

	if data_from_to == '20221103_20230402_delete':
		week_all = {'1': 20221103, '2': 20221104, '3': 20221108, '4': 20221109, '5': 20221110, '6': 20221111,
		            '7': 20221114, '8': 20221115, '9': 20221116, '10': 20221117,
		            '11': 20221118, '12': 20221125, '13': 20221128, '14': 20221129, '15': 20221202, '16': 20221205,
		            '17': 20221208, '18': 20221209, '19': 20221213, '20': 20221214, '21': 20221215, '22': 20221219,
		            '23': 20221230, '24': 20230103, '25': 20230104, '26': 20230105, '27': 20230106, '28': 20230109,
		            '29': 20230111, '30': 20230112, '31': 20230113, '32': 20230116, '33': 20230208,
		            '34': 20230209, '35': 20230210, '36': 20230213, '37': 20230215, '38': 20230217,
		            '39': 20230227, '40': 20230303, '41': 20230306, '42': 20230308, '43': 20230310,
		            '44': 20230313, '45': 20230316, '46': 20230320, '47': 20230327,
		            '48': 20230328, '49': 20230329, '50': 20230402}
		week_name_from_to = str(week_all[str(Day_to)])
		
		print(f'0{week_name_from_to}')
	
	## decoding direction
	def decoding_dir(embedding_train, embedding_test, label_train):
		dir_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
		dir_decoder.fit(embedding_train, label_train)
		dir_pred = dir_decoder.predict(embedding_test)
		
		return dir_pred
	

	
	if not os.path.exists(
			filepath + '/deocder_dir_'+ 'distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
				expect_num_bins) + '_0' + week_name_from_to + '_Day_from_' + str(
				Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				Day_to + add_num_to) + '_'  + today + '_' + data_from_to + '.pt'):

		dir_pred = decoding_dir(torch.from_numpy(cebra_dir_train), torch.from_numpy(cebra_dir_test), label_train)
		
		dir_pred2 = dir_pred.reshape(-1, sequence_length)
		label_test2 = label_test[0:-1:sequence_length].view(-1).numpy()
		
		correct, correct2 = 0, 0
		for i in range(len(dir_pred2)):
			values, counts = np.unique(dir_pred2[i, :], return_counts=True)
			sorted_indices = np.argsort(-counts)
			
			if label_test2[i] in values[sorted_indices][:1]:
				correct += 1
			if label_test2[i] in values[sorted_indices][:2]:
				correct2 += 1
		
	
	
		print(f"seeds:{seeds}",
		      f"out_dim:{output_dimension}, "
		      f"Day_from_{Day_from}_to_{Day_to}, "
		      f"add_Day_from_{Day_from + add_num}_to_{Day_to + add_num_to}, "
		      f"TestDay:{test_Day}, "
		      f"Test Accuracy:{(correct / len(dir_pred2)) * 100}%,"
		      f"Test Accuracy-2:{(correct2 / len(dir_pred2)) * 100}%")
	
	output_acc_align.append([(correct / len(dir_pred2)) * 100, (correct2 / len(dir_pred2)) * 100, seeds])

np.save(filepath + '/outputs_acc_align_data_align_method_' + '_0' + week_name_from_to + '_GY-p2_acrossdays_Day_from_' + str(
	Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(
	Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_' + data_from_to + '.npy',
    output_acc_align)
# print(f"output_acc_align={output_acc_align}")
print(f"acc:top-1, top-2:={np.stack(output_acc_align)[:, :-1].mean(axis=0)}")
##

