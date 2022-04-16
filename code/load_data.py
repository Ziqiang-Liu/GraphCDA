import csv
import torch
import random
from train import train
import numpy as np
  
    
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def dataset(args):
    dataset = dict()

    dataset['c_d'] = read_csv(args.dataset_path + '/c_d.csv')
    

    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].size(0)):
        for j in range(dataset['c_d'].size(1)):
            if dataset['c_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['c_d'][i][j] >= 1:
                one_index.append([i, j, 1])
   
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index

    dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}

    cc_matrix = read_csv(args.dataset_path + '/c_c.csv')
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['cc'] = {'data_matrix': cc_matrix, 'edges': cc_edge_index}

    return dataset, cd_pairs


def feature_representation(model, args, dataset):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model = train(model, dataset, optimizer, args)
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(dataset)
    cir_fea = cir_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, cir_fea, dis_fea


def new_dataset(cir_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
            
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])
    
    
    
    print("--------------------")
    print(cir_fea.shape,dis_fea.shape)
    print("--------------------")
    print(len(unknown_pairs), len(known_pairs))
    
    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = cir_fea[unknown_pairs[i][0],:].tolist() + dis_fea[unknown_pairs[i][1],:].tolist()+[0,1]
        nega_list.append(nega)
        
    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0],:].tolist() + dis_fea[known_pairs[j][1],:].tolist()+[1,0]
        posi_list.append(posi)
    
    samples = posi_list + nega_list
    
    random.shuffle(samples)
    samples = np.array(samples)
    return samples

def C_Dmatix(cd_pairs,trainindex,testindex):
    c_dmatix = np.zeros((585,88))
    for i in trainindex:
        if cd_pairs[i][2]==1:
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]]=1
    
    
    dataset = dict()
    cd_data = []
    cd_data += [[float(i) for i in row] for row in c_dmatix]
    cd_data = torch.Tensor(cd_data)
    dataset['c_d'] = cd_data
    
    train_cd_pairs = []
    test_cd_pairs = []
    for m in trainindex:
        train_cd_pairs.append(cd_pairs[m])
    
    for n in testindex:
        test_cd_pairs.append(cd_pairs[n])



    return dataset['c_d'],train_cd_pairs,test_cd_pairs