##
import os

import cebra
from cebra import CEBRA
import numpy as np

import cebra.datasets
import torch
import datetime
import pickle
import random

from sklearn.ensemble import IsolationForest

torch.backends.cudnn.benchmark = True
print('\n')

data_from_to = '20221103_20230402_delete'

align_method = 2
print('align_method= ', align_method)
# today = datetime.date.today()
# today = str(today)
# today = '2023-07-11'
filepath = '/data/data_hly/Multiscale_SNN_Code/Data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
# import sys
# import time
#
# if len(sys.argv) != 2:
#     print("Usage: python your_script.py <variable>")
#     sys.exit(1)
#
# # 获取命令行参数
# variable = int(sys.argv[1])
# # print(f"Task for variable {variable} started.")
# # time.sleep(3)  # 模拟执行一些任务
# print(f"Task for variable {variable} completed.")
##
weather_align = True


def align_embedings_cross_days(cebra_pos_train, cebra_pos_test):
	if align_method == 4:
		return cebra_pos_test
	elif align_method == 5:
		Q_train, _ = torch.linalg.qr(cebra_pos_train, 'complete')
		cebra_pos_test_align = Q_train @ cebra_pos_test
		return cebra_pos_test_align
	
	cebra_pos_train_sample = cebra_pos_train
	
	# torch
	Q_train, R_train = torch.linalg.qr(cebra_pos_train_sample)
	Q_test, R_test = torch.linalg.qr(cebra_pos_test)
	U, S, V = torch.linalg.svd(Q_train.T @ Q_test)
	V = V.T
	
	if align_method == 1:
		cebra_pos_test_align = cebra_pos_test @ torch.linalg.pinv(R_test) @ V @ torch.linalg.inv(U) @ R_train
	elif align_method == 2:
		cebra_pos_test_align = Q_train @ U @ torch.linalg.pinv(V) @ R_test
	elif align_method == 3:
		cebra_pos_test_align = Q_train @ R_test
	elif align_method == 6:
		Q = torch.linalg.pinv(cebra_pos_test.T) @ cebra_pos_test.T @ cebra_pos_train @ torch.linalg.pinv(cebra_pos_test)
		cebra_pos_test_align = Q @ cebra_pos_test
		return cebra_pos_test_align
	
	return cebra_pos_test_align


##
# data loading
bin_method = 'expect_bins'
expect_num_bins = 50
print('bin_method: ', bin_method)
print('expect_num_bins: ', expect_num_bins)

if bin_method == 'expect_bins':
	all_micro_spikes_concat = torch.load(
		filepath + '/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
	all_macro_conditions_concat = torch.load(
		filepath + '/all_macro_conditions_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
	
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
	# today = '2023-11-28-v1-p2-delete-turn_' + str(turn)  # train:pre 4 weeks, test: next week
	# today = '2023-11-28-v2-p2-delete-turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to + 1
	# today = '2023-11-28-v3-p2-delete-turn_' + str(turn)  # train:1~21, test: 22~50; standard embedding: Day_to
	# today = '2023-11-28-v4-p2-delete-turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to
	# today = '2023-11-28-v5-p2-delete-turn_' + str(turn)  # train:1~12, test: 13~50; standard embedding: Day_to
	# today = '2024-01-01-v1-p2-delete-turn_' + str(turn)  # train:1~12, test: 13~50; standard embedding: Day_to
	# today = '2024-03-07-v1-p2-del-128_turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to
	today = '2024-03-07-v1-p2-del-64_turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to
	# today = '2024-03-07-v1-p2-del-36_turn_' + str(turn)  # train:pre 4 weeks, test: next week; standard embedding: Day_to
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
				       '11': 48, '12': 53, '13': 55, '14': 58, '15': 62, '16': 63, '17': 66, '18': 68, '19': 70, '20': 71,
				       '21': 74, '22': 81, '23': 86, '24': 88, '25': 90, '26': 92, '27': 94, '28': 96, '29': 98, '30': 100,
				       '31': 101, '32': 103, '33': 106, '34': 110, '35': 114, '36': 118, '37': 120, '38': 122,
				       '39': 124, '40': 128, '41': 132, '42': 134, '43': 136, '44': 138, '45': 140, '46': 142,
				       '47': 143, '48': 146, '49': 149, '50': 152}
				
				Day_from = 24
				Day_to = 35
				test_Day = 36
				# test_Day = variable
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
	# cl_dir_model = CEBRA(model_architecture='offset10-model',
	#                      batch_size=512,
	#                      learning_rate=3e-4,
	#                      temperature=1,
	#                      output_dimension=output_dimension,
	#                      max_iterations=max_iterations,
	#                      distance=distance,
	#                      device='cuda_if_available',
	#                      verbose=True)
	# cl_dir_model = CEBRA(model_architecture='offset10-model-mse',
	#                      batch_size=512,
	#                      learning_rate=3e-4,
	#                      temperature=1,
	#                      output_dimension=output_dimension,
	#                      max_iterations=max_iterations,
	#                      distance=distance,
	#                      device='cuda_if_available',
	#                      verbose=True)
	
	cl_dir_model = CEBRA(model_architecture='offset10-model',
	                     batch_size=512,
	                     learning_rate=3e-4,
	                     temperature=1,
	                     output_dimension=output_dimension,
	                     num_hidden_units= 64,
	                     max_iterations=max_iterations,
	                     distance=distance,
	                     device='cuda_if_available',
	                     verbose=True)
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
	decoder_method = 'gru_test_8_aligned'
	print('decoder_method: ', decoder_method)
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import numpy as np
	import torch.utils.data as Data
	
	# 定义超参数
	batch_size = 16
	learning_rate = 0.001
	num_epochs = 1000
	num_layers = 1
	
	sequence_length = 50
	hidden_size = 64
	num_classes = 8
	input_size = output_dimension * num_classes  # dim of embedding output
	
	weather_together = True
	
	if weather_together:
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
			# with open(filepath + '/cl_dir_train_stard_embeddings_' + week_name_from_to + '_gy-p2.pkl', 'rb') as f:
			# 	cebra_dir_train_stard_embeddings = pickle.load(f)
			# 	print('load standard embeddings ...... finished!')
			#
			# Save embeddings in current folder
			cebra_dir_train_stard_embeddings = dict()
			
			len_stand = int(len(label_train) * 0.2 // expect_num_bins) * expect_num_bins # 取训练的部分作为标准
			# len_stand = len(label_train)  # 取训练的所有作为标准
			cebra_dir_train_stard_embeddings['embeddings'] = cebra_dir_train[-len_stand:, :]
			
			# label_index = np.arange(len(label_train) - len_stand, len(label_train))
			label_index = np.arange(0, len_stand)
			id_target = [[] for _ in range(8)]
			for i, value in enumerate(label_train[-len_stand:, :]):
				id_target[int(value)].append(label_index[i])
			for id in range(len(id_target)):
				id_target[id] = np.stack(id_target[id])
			cebra_dir_train_stard_embeddings['id_target'] = id_target
		
		# 训练部分对齐
		if not os.path.exists(
				filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					expect_num_bins) + '_data_aligned_method_' + str(
					align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'):
			print('staring data aligned......')
			id_target_align_train = [[] for _ in range(8)]
			id_l, id_r = 0, 0
			images_train_align = torch.zeros((len(cebra_dir_train), input_size)).to(device)
			for i in range(day[str(Day_from - 1)], day[str(Day_to)]):
				id_target_align = [[] for _ in range(8)]
				for j in range(len(len_for_each_session_trial[i])):
					id_r += len_for_each_session_trial[i][j]
					id_target_align[target_for_each_session_trial[i][j]].append(
						torch.arange(id_l, id_r)
					)
					id_l = id_r
				
				for h in range(len(id_target_align)):
					id_target_align[h] = torch.cat(id_target_align[h])
					id_target_align_train[h].append(id_target_align[h])
					for m in range(8):
						
						print(f"i={i},j={j},h={h},m={m}")
						len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
						                cebra_dir_train_stard_embeddings['id_target'][m], :])
						len_test = len(id_target_align[h])
						if len_train >= len_test:
							nums = len_train // len_test
							temp_align = []
							images_align = []
							
							for k in range(nums):
								idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]
								
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m], :][
									             idx, :]).double().to(device),
									torch.tensor(cebra_dir_train[id_target_align[h], :]).double().to(device)
								))
							images_align.append(torch.stack(temp_align, dim=0).mean(0))
							images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
								torch.stack(images_align, dim=0).mean(0).float()
						else:
							nums = len_test // len_train
							temp_align = []
							images_align = []
							
							for k in range(nums):
								idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]
								
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m], :]).double().to(
										device),
									torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
								))
							if len_test % len_train > 0:
								idx = torch.arange(len_test)[nums * len_train:]
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m][
									             :len_test % len_train],
									             :]).double(
									).to(device),
									torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
								))
							images_align.append(torch.cat(temp_align, dim=0).mean(0))
							images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
								torch.stack(images_align, dim=0).mean(0).float()
			
			images_train_align = images_train_align.cpu().data
			torch.save(images_train_align.cpu().data,
			           filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
				           expect_num_bins) + '_data_aligned_method_' + str(
				           align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
				           Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				           Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt')
		else:
			images_train_align = torch.load(
				filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					expect_num_bins) + '_data_aligned_method_' + str(
					align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt')
		
		print('train data aligned......finished!')
		# 测试部分对齐
		if not os.path.exists(
				filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					expect_num_bins) + '_data_aligned_test_method_' + str(
					align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(
					test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'):
			
			# for idx in range(8):
			# 	id_target_align_train[idx] = torch.cat(id_target_align_train[idx])
			
			id_l, id_r = 0, 0
			images_test_align = torch.zeros((len(cebra_dir_test), input_size)).to(device)
			for i in range(day[str(test_Day - 1)], day[str(test_Day)]):
				id_target_align_test = [[] for _ in range(8)]
				for j in range(len(len_for_each_session_trial[i])):
					id_r += len_for_each_session_trial[i][j]
					id_target_align_test[target_for_each_session_trial[i][j]].append(
						torch.arange(id_l, id_r)
					)
					id_l = id_r
				
				for h in range(len(id_target_align_test)):
					id_target_align_test[h] = torch.cat(id_target_align_test[h])
					for m in range(8):
						print(f"i={i},j={j},h={h},m={m}")
						# len_train = len(images_train_align[id_target_align_train[h],
						#                 m * output_dimension:(m + 1) * output_dimension])
						len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
						                cebra_dir_train_stard_embeddings['id_target'][m], :])
						len_test = len(id_target_align_test[h])
						if len_train >= len_test:
							nums = len_train // len_test
							temp_align = []
							images_align = []
							
							if nums > 10:
								nums = 10
							
							for k in range(nums):
								idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]
								
								# temp_align.append(align_embedings_cross_days(
								# 	images_train_align[
								# 	id_target_align_train[h], m * output_dimension:(m + 1) * output_dimension][
								# 	idx, :].double().to(device),
								# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(device)
								# ))
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m], :][
									             idx, :]).double().to(device),
									torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(device)
								))
							images_align.append(torch.stack(temp_align, dim=0).mean(0))
							images_test_align[id_target_align_test[h],
							m * output_dimension:(m + 1) * output_dimension] = \
								torch.stack(images_align, dim=0).mean(0).float()
						else:
							nums = len_test // len_train
							temp_align = []
							images_align = []
							for k in range(nums):
								idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]
								
								# temp_align.append(align_embedings_cross_days(
								# 	images_train_align[
								# 	id_target_align_train[h],
								# 	m * output_dimension:(m + 1) * output_dimension].double().to(device),
								# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
								# ))
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m], :]).double().to(
										device),
									torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
								))
							if len_test % len_train > 0:
								idx = torch.arange(len_test)[nums * len_train:]
								# temp_align.append(align_embedings_cross_days(
								# 	torch.tensor(images_train_align[
								# 	             id_target_align_train[h][:len_test % len_train],
								# 	             m * output_dimension:(m + 1) * output_dimension]).double().to(device),
								# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
								# ))
								temp_align.append(align_embedings_cross_days(
									torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
									             cebra_dir_train_stard_embeddings['id_target'][m][
									             :len_test % len_train],
									             :]).double().to(device),
									torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
								))
							images_align.append(torch.cat(temp_align, dim=0).mean(0))
							images_test_align[id_target_align_test[h],
							m * output_dimension:(m + 1) * output_dimension] = \
								torch.stack(images_align, dim=0).mean(0).float()
			
			images_test_align = images_test_align.cpu().data
			torch.save(images_test_align.cpu().data,
			           filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
				           expect_num_bins) + '_data_aligned_test_method_' + str(
				           align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
				           Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(
				           test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				           Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt')
		
		else:
			images_test_align = torch.load(
				filepath + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					expect_num_bins) + '_data_aligned_test_method_' + str(
					align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(
					test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
					Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt')
		print('test data aligned......finished!')
	
	## covert the train and test data
	cebra_dir_train_gru = images_train_align.float().reshape(-1, sequence_length, input_size)
	label_train_gru = label_train[0:-1:sequence_length].view(-1)
	# label_train_gru = label_train_gru[train_index]
	
	cebra_dir_test_gru = images_test_align.float().reshape(-1, sequence_length, input_size)
	label_test_gru = label_test[0:-1:sequence_length].view(-1)
	# label_test_gru = label_test_gru[test_index]
	
	train_dataset = Data.TensorDataset(cebra_dir_train_gru, label_train_gru)
	train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
	                               num_workers=2,
	                               drop_last=True)
	
	test_dataset = Data.TensorDataset(cebra_dir_test_gru, label_test_gru)
	test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, )
	
	train_method = 1
	print('train_method:', train_method)
	
	
	class Decoder_GRU(nn.Module):
		def __init__(self, input_size, hidden_size, num_layers, num_classes):
			super(Decoder_GRU, self).__init__()
			self.hidden_size = hidden_size
			self.num_layers = num_layers
			self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
			for p in self.gru.parameters():
				nn.init.normal_(p, 0.0, 0.0001)
			
			# two linear
			# self.fc1 = nn.Linear(sequence_length * hidden_size, 8)
			# self.fc2 = nn.Linear(sequence_length * hidden_size, 512)
			# self.fc4 = nn.Linear(512, 8)
			# self.theat = nn.Parameter(torch.ones(num_classes))
			self.fc3 = nn.Linear(hidden_size, num_classes, bias=True)
		
		def forward(self, x):
			# x = F.normalize(x, dim=2)
			h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 初始化隐藏状态
			
			# print(x.dtype, h0.dtype)
			out, _ = self.gru(x, h0)  # 输入RNN模型
			
			if train_method == 1:
				out = out[:, -1, :]
				out = self.fc3(out)
				out = torch.sigmoid(out)
			
			return out
	
	
	if not os.path.exists(
			filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
				expect_num_bins) + '_align_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
				Day_from) + '_to_' + str(
				Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				Day_to + add_num_to) + '_' + 'lr_' + str(
				learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(hidden_size) + '_method_' + str(
				train_method) + '_together_' + str(
				weather_together) + '_' + today + '_' + data_from_to + '.pt'):
		# 初始化模型和优化器
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = Decoder_GRU(input_size, hidden_size, num_layers, num_classes).to(device)
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		# criterion = nn.CrossEntropyLoss().to(device)
		criterion = nn.MSELoss().to(device)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.995)
		
		# 训练模型
		total = 0
		correct = 0
		total_loss = 0
		total_steps = len(train_loader)
		temp_acc = 0
		temp_loss = []
		for epoch in range(num_epochs):
			for step, (images, labels) in enumerate(train_loader):
				if not weather_together:
					images = images_train_align[step].unsqueeze(0)
				else:
					images = images.to(device)
				
				# #####
				labels2 = labels.to(device)
				# images = images.to(device)
				temp_label = torch.zeros((batch_size, num_classes))
				for kk in range(batch_size):
					temp_label[kk, int(labels[kk])] = 1
				# temp_label[, labels] = 1
				labels = temp_label
				labels = labels.to(device)
				
				# 前向传播和计算损失
				outputs = model(images)
				
				loss = criterion(outputs, labels)
				
				# 反向传播和优化
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				# lr_scheduler.step()
				total_loss += loss.item()
				
				# _, trained = torch.max(torch.stack(temp_predict, dim=1), 1)
				_, trained = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (trained == labels2).sum().item()
			
			if (epoch + 1) % 1 == 0:
				print(
					f"Turn: {turn}, Epoch [{epoch + 1}/{num_epochs}], Loss: {np.round(total_loss / total_steps, 8)}, Train_Acc: {np.round(correct / total * 100, 8)}%, lr: {np.round(optimizer.param_groups[0]['lr'], 8)}")
			
			if (epoch + 1) % 10 == 0 and correct / total > temp_acc:
				torch.save(model,
				           filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					           expect_num_bins) + '_align_method_' + str(
					           align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					           Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(
					           Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + 'lr_' + str(
					           learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(
					           hidden_size) + '_method_' + str(
					           train_method) + '_together_' + str(
					           weather_together) + '_' + today + '_' + data_from_to + '.pt')
				temp_acc = correct / total
			
			if total_loss / total_steps < 0.0001 or (correct / total > 0.97 and correct / total > temp_acc):
				torch.save(model,
				           filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
					           expect_num_bins) + '_align_method_' + str(
					           align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
					           Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(
					           Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + 'lr_' + str(
					           learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(
					           hidden_size) + '_method_' + str(
					           train_method) + '_together_' + str(
					           weather_together) + '_' + today + '_' + data_from_to + '.pt')
				
				print(f'stopped at epoch: {epoch}')
				break
			total_loss = 0
			total = 0
			correct = 0
	else:
		model = torch.load(
			filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_acrossdays_expect_bin_' + str(
				expect_num_bins) + '_align_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(
				Day_from) + '_to_' + str(
				Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(
				Day_to + add_num_to) + '_' + 'lr_' + str(
				learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(hidden_size) + '_method_' + str(
				train_method) + '_together_' + str(
				weather_together) + '_' + today + '_' + data_from_to + '.pt')
	
	# 在测试集上评估模型
	model.eval()
	with torch.no_grad():
		outputs_data = dict(pred_value=[],
		                    pred_label=[],
		                    truth_label=[],
		                    outputs=[])
		correct = 0
		correct2 = 0
		correct3 = 0
		total = 0
		for step, (images, labels) in enumerate(test_loader):
			if not weather_together:
				images_test_align = torch.zeros((1, sequence_length, input_size)).to(device)
				
				for i in range(8):
					# i = int(labels)
					# images = images.reshape(sequence_length, input_size)
					
					len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
					                cebra_dir_train_stard_embeddings['id_target'][i], :])
					len_test = sequence_length
					nums = len_train // len_test
					temp_align = []
					images_align = []
					kk = torch.arange(0, nums, nums // 2)
					for k in kk:
						idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]
						
						temp_align.append(align_embedings_cross_days(
							torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
							             cebra_dir_train_stard_embeddings['id_target'][i], :][idx,
							             :]).double().to(device),
							images[0].double().to(device),
						)
						)
					images_align.append(torch.stack(temp_align, dim=0).mean(0))
					images_test_align[0][:, i * output_dimension:(i + 1) * output_dimension] = torch.stack(
						images_align, dim=0).mean(0)
				
				images = images_test_align
			else:
				# images = images_train_align[step].unsqueeze(0)
				images = images.to(device)
			
			labels = labels.to(device)
			# labels2 = labels
			outputs = model(images)
			
			pred_value, predicted = torch.max(outputs.data, 1)
			
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			if labels in torch.topk(outputs.data, k=2).indices:
				correct2 += 1
			if labels in torch.topk(outputs.data, k=3).indices:
				correct3 += 1
			
			outputs_data['pred_value'].append(pred_value.cpu().data.numpy())
			outputs_data['pred_label'].append(predicted.cpu().data.numpy())
			outputs_data['truth_label'].append(labels.cpu().data.numpy())
			outputs_data['outputs'].append(outputs.cpu().data.numpy())
		
		np.save(filepath + '/outputs_data_align_method_' + str(
			align_method) + '_0' + week_name_from_to + '_GY-p2_acrossdays_Day_from_' + str(
			Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(
			Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + 'lr_' + str(
			learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(hidden_size) + '_method_' + str(
			train_method) + '_together_' + str(weather_together) + '_' + today + '_' + data_from_to + '.npy',
		        outputs_data)
		print(f"decoder_method:{decoder_method},"
		      f"seeds:{seeds}",
		      f"align_method:{align_method},"
		      f"out_dim:{output_dimension}, "
		      f"Day_from_{Day_from}_to_{Day_to}, "
		      f"add_Day_from_{Day_from + add_num}_to_{Day_to + add_num_to}, "
		      f"TestDay:{test_Day}, "
		      f"Test Accuracy:{(correct / total) * 100}%,"
		      f"Test Accuracy-2:{(correct2 / total) * 100}%,"
		      f"Test Accuracy-3:{(correct3 / total) * 100}%,"
		      f"train_method:{train_method},"
		      f"lr:{learning_rate},"
		      f"weather_together:{weather_together}")
	
	output_acc_align.append([(correct / total) * 100, (correct2 / total) * 100, (correct3 / total) * 100, seeds])

np.save(filepath + '/outputs_acc_align_data_align_method_' + str(
	align_method) + '_0' + week_name_from_to + '_GY-p2_acrossdays_Day_from_' + str(
	Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(
	Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + 'lr_' + str(
	learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(
	hidden_size) + '_' + today + '_' + data_from_to + '.npy',
        output_acc_align)
# print(f"output_acc_align={output_acc_align}")
print(f"acc:top-1, top-2, top-3:={np.stack(output_acc_align)[:, :-1].mean(axis=0)}")
##

