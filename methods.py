import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from copy import copy

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

class Network(nn.Module):
    def __init__(self, input_dim, hidden_layers, droupout_prob, fold):
        super().__init__()
        layers = []
        if len(hidden_layers) == 0:
            raise Exception("There must be atleast one layer in the hidden layers")

        hidden_layers.insert(0, input_dim)

        for i, layer in enumerate(hidden_layers[fold:][:-1]):
            layers.append(nn.Linear(hidden_layers[fold+i], hidden_layers[fold+i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=droupout_prob))

        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = torch.sigmoid(self.model(x))
        return logits

# Stacked Denoising Autoencoder class
class SDAE(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(SDAE, self).__init__()
        hidden_layers.insert(0, input_dim)

        encoder_layers = []
        for i, layer in enumerate(hidden_layers[:-1]):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        hidden_layers_rev = hidden_layers[::-1]
        for i, layer in enumerate(hidden_layers_rev[:-1]):
            decoder_layers.append(nn.Linear(hidden_layers_rev[i], hidden_layers_rev[i+1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def run_complete_mixture(seed, folds, epochs, data, input_dim, hidden_layers, droupout_prob, 
                         batch_size_train, batch_size_test, alpha: int = 0.01, beta: int = 0.9, lambd1: int = 0.001, lambd2: int = 0.001):
    print('Running Complete Mixture Learning')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)


        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        auroc.append(roc_auc_score(all_targets, all_predicted))
        # print(f'AUROC for fold {fold}: {auroc[fold]}')
        # print('--------------------------------')        

    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()


def run_mixture_1(seed, folds, epochs, data, source_domain, input_dim, hidden_layers, droupout_prob, 
                  batch_size_train, batch_size_test, alpha: int = 0.01, beta: int = 0.9, lambd1: int = 0.001, lambd2: int = 0.001):

    print('Running Mixture 1')
    # print('--------------------------------')

    information = dict(data)
    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        #Getting a subset of test_ids, essentially the one where race matches source_domain
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids[information['Race'][test_ids] == source_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold, but only on the source_domain
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        auroc.append(roc_auc_score(all_targets, all_predicted))
        # print(f'AUROC for fold {fold}: {auroc[fold]}')
        # print('--------------------------------')        

    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()


def run_mixture_2(seed, folds, epochs, data, target_domain, input_dim, hidden_layers, droupout_prob,
                  batch_size_train: int = 20, batch_size_test: int = 4, alpha: int = 0.01, beta: int = 0.9, lambd1: int = 0.001, lambd2: int = 0.001):
    print('Running Mixture 2')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    information = dict(data)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        #Getting a subset of test_ids, essentially the one where race matches target_domain
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids[information['Race'][test_ids] == target_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold, but only on the source_domain
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        try:
            auroc.append(roc_auc_score(all_targets, all_predicted))
            # print(f'AUROC for fold {fold}: {auroc[fold]}')
            # print('--------------------------------')        
        except:
            print("There weren't enough samples in test to cover both classes in some fold")
    
    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()


def run_independent_major(seed, folds, epochs, data, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2):
    print('Running Independent on Majority')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        auroc.append(roc_auc_score(all_targets, all_predicted))
        # print(f'AUROC for fold {fold}: {auroc[fold]}')
        # print('--------------------------------')        

    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()

def run_independent_minor(seed, folds, epochs, data, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2):
    print('Running Indepedent on Minority')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        #there may be a chance that for some fold, that there may be just 1 class in y_true for minority, so excluding that case completely
        try:
            auroc.append(roc_auc_score(all_targets, all_predicted))
            # print(f'AUROC for fold {fold}: {auroc[fold]}')
            # print('--------------------------------')        
        except:
            print("There weren't enough samples in test to cover both classes in some fold")
    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()



def run_naive(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2):
    print('Running Naive Transfer')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    information = dict(data)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
        
        # Sample elements randomly from a given list of ids, no replacement.
        #Getting a subset of train_ids, essentially the ones where race matches source_domain
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids[information['Race'][train_ids] == source_domain])
        #Getting a subset of test_ids, essentially the one where race matches target_domain
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids[information['Race'][test_ids] == target_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_train, sampler=train_subsampler, drop_last=False)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_test, sampler=test_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)

        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold, but only on the source_domain
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        try:
            auroc.append(roc_auc_score(all_targets, all_predicted))
            # print(f'AUROC for fold {fold}: {auroc[fold]}')
            # print('--------------------------------')        
        except:
            print("There weren't enough samples in test to cover both classes in some fold")
    
    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()



def run_supervised(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob,
                   batch_size_pretrain, batch_size_finetune, batch_size_finetunetest, alpha, beta, lambd1, lambd2, alpha_finetune):
    print('Running Supervised Transfer')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    information = dict(data)
    
    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')

        #PRE TRAINING on entire majority

        # Sample elements randomly from a given list of ids, no replacement.
        #Getting a subset of train_ids, essentially the ones where race matches source_domain
        train_subsampler = torch.utils.data.SubsetRandomSampler(np.arange(len(information['X']))[information['Race']==source_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_pretrain, sampler=train_subsampler, drop_last=False)

        # Init the neural network
        network = Network(input_dim, hidden_layers, droupout_prob, fold)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=alpha,
                                nesterov=True, momentum=beta)


        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloader, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = 0.001 * torch.norm(all_params, 1)
                l2_regularization = 0.001 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer.step()

                current_loss += loss.item()

        #FINE TUNING on target_domain
        #Getting a subset of train_ids, essentially the ones where race matches target_domain
        train_subsamplerminority = torch.utils.data.SubsetRandomSampler(train_ids[information['Race'][train_ids] == target_domain])
        #Getting a subset of test_ids, essentially the one where race matches target_domain
        test_subsamplerminority = torch.utils.data.SubsetRandomSampler(test_ids[information['Race'][test_ids] == target_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloaderminority = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_finetune, sampler=train_subsamplerminority, drop_last=False)
        testloaderminority = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_finetunetest, sampler=test_subsamplerminority, drop_last=False)

        # Initialize optimizer
        optimizer2 = torch.optim.SGD(network.parameters(), lr=alpha_finetune,
                                nesterov=True, momentum=beta)


        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0

            #Iterating over train dataloader
            for i, data in enumerate(trainloaderminority, 0):
                #Get inputs from data
                inputs, targets = data.double()[:, :-1], data.double()[:, -1]
                
                #Zero the gradients
                optimizer2.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization

                #Perform Backward Pass
                loss.backward()

                #Perform Optimization
                optimizer2.step()

                current_loss += loss.item()


        #Training done in this fold over all epochs using Mini-Batch

        #Testing Model in this fold, but only on the source_domain
        correct, total = 0, 0
        all_targets, all_predicted = [], []

        with torch.no_grad():
            for i, data in enumerate(testloaderminority, 0):
                # Get inputs
                inputs, targets = data[:, :-1], data[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        try:
            auroc.append(roc_auc_score(all_targets, all_predicted))
            # print(f'AUROC for fold {fold}: {auroc[fold]}')
            # print('--------------------------------')        
        except:
            print("There weren't enough samples in test to cover both classes in some fold")
    
    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()

# Custom corruption function to add noise to the input data
def corrupt_input(input_data, corruption_level=0.3):
    mask = (torch.rand_like(input_data) < corruption_level).float()
    corrupted_data = input_data + mask * torch.randn_like(input_data)
    return corrupted_data

def run_unsupervised(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob, num_epochs_sdae, batch_size_sdae_pretrain, batch_size_sdae_finetune, batch_size_sdae_finetunetest, beta, lambd1, lambd2, alpha_sdae, alpha_sdae_finetune):
    print('Running Unsupervised Transfer')
    # print('--------------------------------')

    #Creating Dataset
    dataset = np.concatenate((data['X'], data['Y'].reshape(-1, 1)), axis=1)
    information = dict(data)
    hidden_layer_info = copy(hidden_layers)

    #To store results
    auroc = []
    results = {}
    
    #Initializing Loss Function
    loss_function = nn.BCELoss()
    
    #setting seeds
    torch.manual_seed(seed)

    #defining K-Fold Validator
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)

    #For each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(f'FOLD {fold}')
    
        #Creating an object of autoencoder and training it
        autoencoder = SDAE(input_dim, copy(hidden_layer_info))
        # print(autoencoder)
        #Defining loss function and optimiser for training the autoencoder
        criterion_sdae = nn.MSELoss()
        optimizer_sdae = torch.optim.SGD(autoencoder.parameters(), lr=alpha_sdae)
    
        #Defining dataloader over entire training data
        subsampler = torch.utils.data.SubsetRandomSampler(np.arange(len(data['X'])).reshape(-1,))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_sdae_pretrain, sampler=subsampler, drop_last=False)
    
        #Training the autoencoder for 500 epochs
        for epoch in range(num_epochs_sdae):
            for i, source_batch in enumerate(dataloader, 0):
                source_batch = source_batch.double()[:, :-1]
                source_batch = corrupt_input(source_batch)
                optimizer_sdae.zero_grad()
                _, reconstructed = autoencoder(source_batch.type(torch.FloatTensor))
                loss_sdae = criterion_sdae(reconstructed, source_batch.float())
                loss_sdae.backward()
                optimizer_sdae.step()
        #FINE TUNING on target_domain
        #Transfering weights from autoencoder to model to be fine tuned
    
        #Init the neural network
        network = Network(input_dim, copy(hidden_layer_info), droupout_prob, 0)
        network.apply(reset_weights)

        # print(network)

        #Copy weights and biases from trained autoencoder
        i = 0
        j = 0
        for _ in hidden_layers:
            # print(fold, i, j)
            network.model[i].weight.data = autoencoder.encoder[j].weight.data
            network.model[i].bias.data = autoencoder.encoder[j].bias.data
            i += 3
            j += 2

        # print("weight transfer done")
        #Getting a subset of train_ids, essentially the ones where race matches target_domain
        train_subsamplerminority = torch.utils.data.SubsetRandomSampler(train_ids[information['Race'][train_ids] == target_domain])
        #Getting a subset of test_ids, essentially the one where race matches target_domain
        test_subsamplerminority = torch.utils.data.SubsetRandomSampler(test_ids[information['Race'][test_ids] == target_domain])
        
        # Define data loaders for training and testing data in this fold
        trainloaderminority = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size_sdae_finetune, sampler=train_subsamplerminority, drop_last=False)
        testloaderminority = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size_sdae_finetunetest, sampler=test_subsamplerminority, drop_last=False)
    
        # Initialize optimizer
        optimizer2 = torch.optim.SGD(network.parameters(), lr=alpha_sdae_finetune, nesterov=True, momentum=beta)
    
        #Iterating through epochs
        for epoch in range(0, epochs):
            current_loss = 0.0
    
            #Iterating over train dataloader
            for i, dataf in enumerate(trainloaderminority, 0):
                #Get inputs from data
                inputs, targets = dataf.double()[:, :-1], dataf.double()[:, -1]
                
                #Zero the gradients
                optimizer2.zero_grad()
                
                #Forward pass
                outputs = network(inputs.type(torch.FloatTensor))
                
                #Compute BCELoss + L1 + L2
                if targets.shape != outputs.reshape(-1).shape:
                    continue
                loss = loss_function(outputs.reshape(-1), targets.float())
                all_params = torch.cat([x.view(-1) for x in network.parameters()])
                l1_regularization = lambd1 * torch.norm(all_params, 1)
                l2_regularization = lambd2 * torch.norm(all_params, 2)
                loss += l1_regularization + l2_regularization
    
                #Perform Backward Pass
                loss.backward()
    
                #Perform Optimization
                optimizer2.step()
    
                current_loss += loss.item()
    
        #Training done in this fold over all epochs using Mini-Batch
    
        #Testing Model in this fold, but only on the target_domain
        correct, total = 0, 0
        all_targets, all_predicted = [], []
    
        with torch.no_grad():
            for i, dataf in enumerate(testloaderminority, 0):
                # Get inputs
                inputs, targets = dataf[:, :-1], dataf[:, -1]
            
                # Generate outputs
                outputs = network(inputs.type(torch.FloatTensor))
                
                # Store targets and predictions for AUROC calculation
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(outputs.data.cpu().numpy())
    
        # Calculate AUROC
        try:
            auroc.append(roc_auc_score(all_targets, all_predicted))
            # print(f'AUROC for fold {fold}: {auroc[fold]}')
            # print('--------------------------------')        
        except:
            print("There weren't enough samples in test to cover both classes in some fold")
    
    print(f'Mean AUROC: {np.array(auroc).mean()}')
    print('--------------------------------')        
    
    return np.array(auroc).mean()


def run_mixture_tab():
    pass