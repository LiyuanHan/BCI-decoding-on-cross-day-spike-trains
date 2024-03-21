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
import sys
import time

if len(sys.argv) != 2:
    print("Usage: python your_script.py <variable>")
    sys.exit(1)

# 获取命令行参数
variable = int(sys.argv[1])
# print(f"Task for variable {variable} started.")
# time.sleep(3)  # 模拟执行一些任务
print(f"Task for variable {variable} completed.")
##
weather_align = True
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
	
	load_data_1 = np.load(filepath + '/len_for_each_session_trial_expect_' + str(
		expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.npy', allow_pickle=True)
	len_for_each_session_trial = load_data_1.tolist()
	
	load_data_2 = np.load(filepath + '/target_for_each_session_trial_expect_' + str(
		expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.npy', allow_pickle=True)
	target_for_each_session_trial = load_data_2.tolist()
	
	# 1-8 ==> 0-7
	all_macro_conditions_concat = all_macro_conditions_concat - torch.tensor(1)
	for i in range(len(target_for_each_session_trial)):
		target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]) - 1).tolist()

##

turns = 8
output_acc_without_align = []
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
	# today = '2024-01-25-v1-p2-delete_pureMLP_single_turn_' + str(turn)  # train:80%, test: 20%; standard embedding: Day_from (no shuffle)
	today = '2024-01-25-v2-p2-delete_pureMLP_single_turn_' + str(turn)  # train:80%, test: 20%; standard embedding: Day_from (shuffled)
	# # split_data and Load the model for training
	max_iterations = 10000  # default is 5000.
	output_dimension = 36  # here, we set as a variable for hypothesis testing below.
	distance = 'euclidean'
	print('output_dimension: ', output_dimension)
	
	if bin_method == 'expect_bins':
		train_data = 'days'
		print('train_data: ', train_data)
		if train_data == 'days':
			if data_from_to == '20221103_20230402_delete':
				day = {'0': 0, '1': 5, '2': 10, '3': 13, '4': 18, '5': 24, '6': 30, '7': 36, '8': 40, '9': 42, '10': 46,
				       '11': 48, '12': 53, '13': 55, '14': 58, '15': 62, '16': 63, '17': 66, '18': 68, '19': 70,
				       '20': 71, '21': 74, '22': 81, '23': 86, '24': 88, '25': 90, '26': 92, '27': 94, '28': 96,
				       '29': 98,
				       '30': 100, '31': 101, '32': 103, '33': 106, '34': 110, '35': 114, '36': 118, '37': 120,
				       '38': 122,
				       '39': 124, '40': 128, '41': 132, '42': 134, '43': 136, '44': 138, '45': 140, '46': 142,
				       '47': 143, '48': 146, '49': 149, '50': 152}
				# Day_from = 1
				Day_from = variable
				Day_to = Day_from
				test_Day = Day_from
				
				ratio = 0.8
				print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
				print('--------------------test_Day: {}------------------'.format(test_Day))
				
				
				def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):
					
					split_idx_start_beg = 0
					for i in range(day[str(Day_from - 1)]):
						split_idx_start_beg += sum(len_for_each_session_trial[i])
					split_idx_start_end = 0
					for i in range(day[str(Day_to)]):
						split_idx_start_end += sum(len_for_each_session_trial[i])
					
					data_neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
					data_label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
					
					random_indices = np.arange(0, len(data_label_train), expect_num_bins)
					np.random.shuffle(random_indices)
					
					len_train = (int(len(data_neural_train) * ratio) // expect_num_bins)
					
					random_indices_augmented_train = []
					for num in random_indices[:len_train]:
						augmented_indices = [num + i for i in range(expect_num_bins)]
						random_indices_augmented_train.extend(augmented_indices)
					
					random_indices_augmented_test = []
					for num in random_indices[len_train:]:
						augmented_indices = [num + i for i in range(expect_num_bins)]
						random_indices_augmented_test.extend(augmented_indices)
					
					neural_train = data_neural_train[np.stack(random_indices_augmented_train), :]
					label_train = data_label_train[np.stack(random_indices_augmented_train), :]
					
					neural_test = data_neural_train[np.stack(random_indices_augmented_test), :]
					label_test = data_label_train[np.stack(random_indices_augmented_test), :]
					
					return neural_train, neural_test, label_train, label_test, random_indices
				
	
	# split data
	neural_train, neural_test, label_train, label_test, random_indices = split_data(all_micro_spikes_concat,
	                                                                all_macro_conditions_concat,
	                                                                len_for_each_session_trial, Day_from,
	                                                                Day_to)  # direction
	
	add_num = 0
	add_num_to = 0
	print(f"add_num={add_num},add_num_to={add_num_to}")
	
	print('neural_train_length: ', len(neural_train))
	print('split data...finished!')
	
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
	hidden_size = 2048  # 64
	num_classes = 8
	input_size = neural_train.shape[1]  # dim of embedding output
	
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
	## covert the train and test data
	cebra_dir_train_gru = neural_train.reshape(-1, sequence_length, input_size)
	label_train_gru = label_train[0:-1:sequence_length].view(-1)
	
	cebra_dir_test_gru = neural_test.reshape(-1, sequence_length, input_size)
	label_test_gru = label_test[0:-1:sequence_length].view(-1)
	
	train_dataset = Data.TensorDataset(cebra_dir_train_gru, label_train_gru)
	train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
	                               num_workers=0,
	                               drop_last=True)
	
	test_dataset = Data.TensorDataset(cebra_dir_test_gru, label_test_gru)
	test_loader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, )
	
	train_method = 1
	print('train_method:', train_method)
	
	
	class Decoder_MLP(nn.Module):
		def __init__(self, input_size, hidden_size, num_classes):
			super(Decoder_MLP, self).__init__()
			self.fc1 = nn.Linear(input_size, hidden_size)
			self.relu = nn.ReLU()
			self.fc2 = nn.Linear(hidden_size, num_classes)
		
		def forward(self, x):
			out = self.fc1(x)
			out = self.relu(out)
			out = self.fc2(out)
			return out
	
	
	if not os.path.exists(
			filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_single_bin_' + str(
				expect_num_bins) + '_align_method_' + str(align_method) + '_0' + week_name_from_to + '_testDay_' + str(
				test_Day) + '_' + 'lr_' + str(
				learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(hidden_size) + '_method_' + str(
				train_method) + '_together_' + str(
				weather_together) + '_' + today + '_' + data_from_to + '.pt'):
		# 初始化模型和优化器
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = Decoder_MLP(input_size * sequence_length, hidden_size, num_classes).to(device)
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
				images = images.to(device)
				images = images.reshape(-1, sequence_length * input_size)
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
				           filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_single_bin_' + str(
					           expect_num_bins) + '_align_method_' + str(
					           align_method) + '_0' + week_name_from_to + '_testDay_' + str(
					           test_Day) + '_' + 'lr_' + str(
					           learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(
					           hidden_size) + '_method_' + str(
					           train_method) + '_together_' + str(
					           weather_together) + '_' + today + '_' + data_from_to + '.pt')
				temp_acc = correct / total
			
			if total_loss / total_steps < 0.0001 or (correct / total > 0.97 and correct / total > temp_acc):
				torch.save(model,
				           filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_single_bin_' + str(
					           expect_num_bins) + '_align_method_' + str(
					           align_method) + '_0' + week_name_from_to + '_testDay_' + str(
					           test_Day) + '_' + 'lr_' + str(
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
			filepath + '/deocder_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_single_bin_' + str(
				expect_num_bins) + '_align_method_' + str(align_method) + '_0' + week_name_from_to + '_testDay_' + str(
				test_Day) + '_' + 'lr_' + str(
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
			images = images.to(device)
			images = images.reshape(1, sequence_length * input_size)
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
		
		np.save(filepath + '/outputs_data_without_align' + '_GY-p2_single' + '_testDay_' + str(test_Day) + '_' + 'lr_' + str(
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
	
	output_acc_without_align.append(
		[(correct / total) * 100, (correct2 / total) * 100, (correct3 / total) * 100, seeds])

np.save(filepath + '/outputs_acc_without_align_data_0' + week_name_from_to + '_GY-p2_single' + '_testDay_' + str(test_Day) + '_' + 'lr_' + str(
	learning_rate) + '_batch_' + str(batch_size) + '_hidden_' + str(
	hidden_size) + '_' + today + '_' + data_from_to + '.npy',
        output_acc_without_align)
# print(f"output_acc_align={output_acc_align}")
print('saved finished!....!')
print(f"acc:top-1, top-2, top-3:={np.stack(output_acc_without_align)[:, :-1].mean(axis=0)}")
##

