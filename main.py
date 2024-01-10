from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import torch
import argparse
from numpy.random import seed
import random as rn
import time
from torch import nn
from torch.utils.data import DataLoader
from data import get_protein_data, get_m_or_micro_rna_data, get_methylation_data
from methods import run_complete_mixture, run_mixture_1, run_mixture_2, run_independent_major, run_independent_minor, run_naive, run_supervised, run_unsupervised
import os
import matplotlib.pyplot as plt
import time
import openpyxl

if __name__ == "__main__":
    start = time.time()

    #Other Parameters
    hidden_layers = [128, 64]
    droupout_prob = 0.5
    alpha = 0.01
    beta = 0.9
    lambd1 = 0.001
    lambd2 = 0.001
    folds = 3
    epochs = 100
    n = 20 #Number of independent runs

    #General for most cases, for some special cases the value is specified in the function call
    batch_size_train = 20
    batch_size_test = 20
    batch_size_minority = 4

    #Batch Sizes for Supervised Transfer
    batch_size_pretrain = 20
    batch_size_finetune = 10
    batch_size_finetunetest = 10
    alpha_finetune = 0.002

    #Batch Sizes for Unsupervised Transfer
    batch_size_sdae_pretrain = 32
    batch_size_sdae_finetune = 10
    batch_size_sdae_finetunetest = 10
    num_epochs_sdae = 500
    alpha_sdae = 0.01
    alpha_sdae_finetune = 0.002

    results_path = 'Results/'
    try:
        os.mkdir(results_path)
        print(f"Directory '{results_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{results_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print()
    
    #Taking Arguments from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("cancer_type", type=str, help="Cancer Type")
    parser.add_argument("datatype", type=str, help="Data Type")
    parser.add_argument("target", type=str, help="Clinical Outcome Endpoint")
    parser.add_argument("years", type=int, help="Event Time Threhold (Years)")
    parser.add_argument("target_domain", type=str, help="Target Group")
    args = parser.parse_args()
    
    #Getting data from arguments
    cancer_type = args.cancer_type
    datatype = args.datatype
    target = args.target
    years = args.years
    source_domain = 'WHITE'
    target_domain = args.target_domain
    groups = (source_domain,target_domain)
    
    #Data Paths
    if datatype == "Protein":
        data_path = 'Dataset/ProteinData/Protein.txt'
        genetic_data_path = 'Dataset/Genetic_Ancestry.xlsx'
        clinical_outcome_path = 'Dataset/TCGA-CDR-SupplementalTableS1.xlsx'
        input_dim = 189
    elif datatype == "mRNA":
        data_path = "Dataset/mRNAData/mRNA.mat"
        genetic_data_path = 'Dataset/Genetic_Ancestry.xlsx'
        clinical_outcome_path = 'Dataset/TCGA-CDR-SupplementalTableS1.xlsx'
        input_dim = 200
    elif datatype == "MicroRNA":
        data_path = "Dataset/MicroRNAData/MicroRNA-Expression.mat"
        genetic_data_path = 'Dataset/Genetic_Ancestry.xlsx'
        clinical_outcome_path = 'Dataset/TCGA-CDR-SupplementalTableS1.xlsx'
        input_dim = 200
    elif datatype == "Methylation":
        data_path = "Dataset/MethylationData/Methylation.mat"
        genetic_data_path = 'Dataset/MethylationData/MethylationGenetic.xlsx'
        clinical_outcome_path = 'Dataset/MethylationData/MethylationClinInfo.xlsx'
        input_dim = 200
    else:
        print("Only allowed data types are Protein, mRNA, MicroRNA, Methylation")
        exit()

    TaskName = 'TCGA-'+cancer_type+'-'+datatype+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)+'YR'
    out_file_name = './Result/' + TaskName + '.xlsx'

    #Getting Dataset
    if datatype == "Protein":
        data = get_protein_data(cancer_type=cancer_type,target=target,groups=groups, data_path=data_path, genetic_data_path=genetic_data_path,clinical_outcome_path=clinical_outcome_path, years = years)
    elif datatype == "mRNA":
        data = get_m_or_micro_rna_data(cancer_type=cancer_type,target=target,groups=groups, data_path=data_path, genetic_data_path=genetic_data_path,clinical_outcome_path=clinical_outcome_path, years = years, input_dim = input_dim, datatype = datatype)
    elif datatype == "MicroRNA":
        data = get_m_or_micro_rna_data(cancer_type=cancer_type,target=target,groups=groups, data_path=data_path, genetic_data_path=genetic_data_path,clinical_outcome_path=clinical_outcome_path, years = years, input_dim = input_dim, datatype = datatype)
    elif datatype == "Methylation":
        data = get_methylation_data(cancer_type=cancer_type,target=target,groups=groups, data_path=data_path, genetic_data_path=genetic_data_path,clinical_outcome_path=clinical_outcome_path, years = years, input_dim = input_dim)

    #Getting race wise seperated data 
    dataminority = {}
    dataminority['X'] = data['X'][data['Race']==target_domain]
    dataminority['Time'] = data['Time'][data['Race']==target_domain]
    dataminority['Y'] = data['Y'][data['Race']==target_domain]
    dataminority['SampleID'] = data['SampleID'][data['Race']==target_domain]

    datamajority = {}
    datamajority['X'] = data['X'][data['Race']=='WHITE']
    datamajority['Time'] = data['Time'][data['Race']=='WHITE']
    datamajority['Y'] = data['Y'][data['Race']=='WHITE']
    datamajority['SampleID'] = data['SampleID'][data['Race']=='WHITE']


    #Making sure atleast 5 samples present for both categories for both prognosis categories
    if sum(dataminority['Y'] == 1) < 5:
        raise ValueError('Insufficient Positive Prognosis Samples for Minority Class')
    if sum(dataminority['Y'] == 0) < 5:
        raise ValueError('Insufficient Negative Prognosis Samples for Minority Class')
    if sum(datamajority['Y'] == 1) < 5:
        raise ValueError('Insufficient Positive Prognosis Samples for Majority Class')
    if sum(datamajority['Y'] == 0) < 5:
        raise ValueError('Insufficient Negative Prognosis Samples for Majority Class')

    #Creating lists to store results
    complete_mixture = []
    mixture_1 = []
    mixture_2 = []
    independent_major = []
    independent_minor = []
    naive = []
    supervised = []
    unsupervised = []

    # It's-a model time
    for i in range(n):
        print(f'Independent Set {i+1}/{n}')
        print('--------------------------------')
        seed = i
        complete_mixture.append(run_complete_mixture(seed, folds, epochs, data, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2))
        mixture_1.append(run_mixture_1(seed, folds, epochs, data, source_domain, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2))
        mixture_2.append(run_mixture_2(seed, folds, epochs, data, target_domain, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test = batch_size_minority, alpha= alpha, beta=beta, lambd1= lambd1, lambd2 = lambd2))
        independent_major.append(run_independent_major(seed, folds, epochs, datamajority, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test, alpha, beta, lambd1, lambd2))
        independent_minor.append(run_independent_minor(seed, folds, epochs, dataminority, input_dim, hidden_layers, droupout_prob, batch_size_train = batch_size_minority, batch_size_test = batch_size_minority, alpha = alpha, beta = beta, lambd1 = lambd1, lambd2 = lambd2))
        naive.append(run_naive(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob, batch_size_train, batch_size_test = batch_size_minority, alpha= alpha, beta=beta, lambd1= lambd1, lambd2 = lambd2))
        supervised.append(run_supervised(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob, batch_size_pretrain, batch_size_finetune, batch_size_finetunetest, alpha, beta, lambd1, lambd2, alpha_finetune))        
        unsupervised.append(run_unsupervised(seed, folds, epochs, data, source_domain, target_domain, input_dim, hidden_layers, droupout_prob, num_epochs_sdae, batch_size_sdae_pretrain, batch_size_sdae_finetune, batch_size_sdae_finetunetest, beta, lambd1, lambd2, alpha_sdae, alpha_sdae_finetune))
        print()

    # Creating Box Plot & Saving Results
    results = {
        'Mixture_0': complete_mixture,
        'Mixture_1': mixture_1,
        'Mixture_2': mixture_2,
        f'Independent_{source_domain}': independent_major,
        f'Independent_{target_domain}': independent_minor,
        'Naive Transfer': naive,
        'Supervised Transfer': supervised,
        'Unsupervised Transfer': unsupervised
        }

    resultsdf = pd.DataFrame(results)
    resultsdf.to_excel(f'Results/{cancer_type}-PROTEIN-WHITE-{target_domain}-{target}-{years}YR.xlsx')

    plt.boxplot(resultsdf, vert=False)
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8], ['Mixture_0', 'Mixture_1', 'Mixture_2', 'Independent_1', 'Independent_2', 'Naive', 'Supervised Transfer', 'Unsupervised Transfer'])
    plt.savefig(f'Results/{cancer_type}-PROTEIN-WHITE-{target_domain}-{target}-{years}YR.png', dpi=1000, bbox_inches='tight')

    # Saving calculated performance gaps
    excelfile = f'Results/{cancer_type}-PROTEIN-WHITE-{target_domain}-{target}-{years}YR.xlsx'
    # Load the workbook
    wb = openpyxl.load_workbook(excelfile)
    # Select the sheet by name
    sheet = wb["Sheet1"]
    
    sheet["K6"].value = "Performance Gaps"
    
    sheet["K7"].value = "G"
    sheet["K8"].value = "(G\u0303) Using Supervised Transfer"
    sheet["K9"].value = "(G\u0303) Using Unsupervised Transfer"

    sheet["L7"].value = ((np.median(mixture_1) + np.median(independent_major))/2) - ((np.median(mixture_2) + np.median(independent_minor))/2)
    sheet["L8"].value = ((np.median(mixture_1) + np.median(independent_major))/2) - np.median(supervised)
    sheet["L9"].value = ((np.median(mixture_1) + np.median(independent_major))/2) - np.median(unsupervised)

    # Save the workbook
    wb.save(excelfile)
    
    end = time.time()
    print(f'Time Taken By Script: {(end-start)/60} minutes')