import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import torch
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest

### Getting the Protein Dataset ###

def get_protein_data(cancer_type, target, groups, data_path, genetic_data_path, clinical_outcome_path, years):
    
    """
        This function takes in input the cancer_type, target outcome, the groups to consider, data_path for the protein file, genetic_data_path for the racial information, clinical_outcome_path for the clinical outcome file, and the number of years under consideration
    
        It returns a dictionary containing C (survival or not overall), Race (Racial Information), Time (Time of death if not survived within mentioned years else time of last contact), E (Death or not death), SampleID, Features, Y (Target evaluated based on Time and C), X (Standardized Feature Dataset)
    """
        
    #Reading the data_path file
    df = pd.read_csv(data_path, sep='\t', index_col = 'SampleID')
    #Dropping all samples with na values
    df = df.dropna(axis=1)
    
    #Getting the Tumors from the cancer type
    
    #Some already defined cancer types with their cooreponding tumor types
    CancerTumor_Map = {'GBMLGG': ['GBM', 'LGG'], 'COADREAD': ['COAD', 'READ'], 'KIPAN': ['KIRC', 'KICH', 'KIRP'], 'STES': ['ESCA', 'STAD'], 'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'], 'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'], 'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'], 'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']}
    
    #If not a prespecified cancer type, or a tumor type input, add a new entry to the dictionary with that name
    if cancer_type not in CancerTumor_Map:
        CancerTumor_Map[cancer_type] = [cancer_type]
    
    tumors = CancerTumor_Map[cancer_type]
    
    #Now from the protein dataframe get the cooresponding samples with the specified tumor types and dropping the TumorType Column since no longer useful
    df = df[df['TumorType'].isin(tumors)]
    df = df.drop(columns=['TumorType'])
    
    #Getting the patient ID from the sample ID
    df.index = [sampleid[:12] for sampleid in df.index.values]
    
    
    
    #Now to add the race column from Genetic Ancestry Dataset
    #Opening the xlsx genetic ancestry file, and getting only the data for the selected tumortypes
    races_df = pd.concat([pd.read_excel(genetic_data_path, tumor, usecols='A,E', index_col='Patient_ID', keep_default_na=False) for tumor in tumors])
    
    #Dropping Unknown Ethnicites
    races_df = races_df[races_df['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    
    #Renaming Ethnicities
    races_df.loc[:, 'race'] = races_df.loc[:, 'EIGENSTRAT']
    races_df.loc[races_df['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    races_df.loc[races_df['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    races_df.loc[races_df['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    races_df.loc[races_df['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    races_df.loc[races_df['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    races_df = races_df.drop(columns=['EIGENSTRAT'])
    
    #Joining Race df to Main df
    df = df.join(races_df, how='inner')
    df = df[df['race'].isin(groups)]
    df = df.dropna(axis='columns')
    
    
    # Now to add the data for the clinical outcome
    #First chossing the column number based on input target
    if target == 'OS':
        cols = 'B,Z,AA'
    elif target == 'DSS':
        cols = 'B,AB,AC'
    elif target == 'DFI':
        cols = 'B,AD,AE'
    elif target == 'PFI':
        cols = 'B,AF,AG'
    else:
        raise ValueError('Incorrect Input Target, Choose from OS, DSS, DFI, PFI')
    #Reading from file
    df_C_T = pd.read_excel(clinical_outcome_path, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
    #Changing Names of columns
    #The dataset contains Death (E)
    df_C_T.columns = ['E', 'T']
    df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    #Changing Death to Survival (C)
    df_C_T['C'] = 1 - df_C_T['E']
    #Joining to Main DF
    df = df.join(df_C_T, how='inner')
    
    #Adding target column Y
    df['Y'] = 1
    #We will only take the people who either died or whose time of survival is greater than 365 * years days
    #Essentially we will not take the people who lived, but last contact time is less than 365*years days
    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    #Now for all these people, set Y to 0 if time of survival is less than 365*years days
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    
    #Returning Complete data as a dictionary
    data = {}
    data['C'] = np.asarray(df['C'].tolist(), dtype=np.int32)
    data['Race'] = np.asarray(df['race'].tolist())
    data['Time'] = np.asarray(df['T'].tolist(), dtype=np.float32)
    data['E'] = np.asarray(df['E'].tolist(), dtype= np.int32)
    data['SampleID'] = df.index.values
    data['Features'] = list(df)
    data['Y'] = np.asarray(df['Y'].tolist(), dtype=np.int32)
    data['X'] = df.drop(columns=['C', 'race', 'T', 'E', 'Y']).values.astype('float32')
    #Standardizing Features
    scaler = StandardScaler()
    data['X'] = scaler.fit_transform(data['X'])

    return data

def get_m_or_micro_rna_data(cancer_type, target, groups, data_path, genetic_data_path, clinical_outcome_path, years, input_dim, datatype):
    
    """
        This function takes in input the cancer_type, target outcome, the groups to consider, data_path for the protein file, genetic_data_path for the racial information, clinical_outcome_path for the clinical outcome file, and the number of years under consideration
    
        It returns a dictionary containing C (survival or not overall), Race (Racial Information), Time (Time of death if not survived within mentioned years else time of last contact), E (Death or not death), SampleID, Features, Y (Target evaluated based on Time and C), X (Standardized Feature Dataset)
    """
    
    #Getting the Tumors from the cancer type
    
    #Some already defined cancer types with their cooreponding tumor types
    CancerTumor_Map = {'GBMLGG': ['GBM', 'LGG'], 'COADREAD': ['COAD', 'READ'], 'KIPAN': ['KIRC', 'KICH', 'KIRP'], 'STES': ['ESCA', 'STAD'], 'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'], 'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'], 'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'], 'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']}
    
    #If not a prespecified cancer type, or a tumor type input, add a new entry to the dictionary with that name
    if cancer_type not in CancerTumor_Map:
        CancerTumor_Map[cancer_type] = [cancer_type]
    
    tumors = CancerTumor_Map[cancer_type]

    A = loadmat(data_path)

    if datatype == "mRNA":
        X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    elif datatype == "MicroRNA":
        X, Y, GeneName, SampleName = A['X'].astype('float32'), A['CancerType'], A['FeatureName'][0], A['SampleName']
    
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    df_Y = df_Y[df_Y['Disease'].isin(tumors)]
    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns=['Disease'])

    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')



    
    #Now to add the race column from Genetic Ancestry Dataset
    #Opening the xlsx genetic ancestry file, and getting only the data for the selected tumortypes
    races_df = pd.concat([pd.read_excel(genetic_data_path, tumor, usecols='A,E', index_col='Patient_ID', keep_default_na=False) for tumor in tumors])
    
    #Dropping Unknown Ethnicites
    races_df = races_df[races_df['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    
    #Renaming Ethnicities
    races_df.loc[:, 'race'] = races_df.loc[:, 'EIGENSTRAT']
    races_df.loc[races_df['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    races_df.loc[races_df['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    races_df.loc[races_df['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    races_df.loc[races_df['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    races_df.loc[races_df['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    races_df = races_df.drop(columns=['EIGENSTRAT'])
    
    #Joining Race df to Main df
    df = df.join(races_df, how='inner')
    df = df[df['race'].isin(groups)]
    df = df.dropna(axis='columns')
    
    
    # Now to add the data for the clinical outcome
    #First chossing the column number based on input target
    if target == 'OS':
        cols = 'B,Z,AA'
    elif target == 'DSS':
        cols = 'B,AB,AC'
    elif target == 'DFI':
        cols = 'B,AD,AE'
    elif target == 'PFI':
        cols = 'B,AF,AG'
    else:
        raise ValueError('Incorrect Input Target, Choose from OS, DSS, DFI, PFI')
    #Reading from file
    df_C_T = pd.read_excel(clinical_outcome_path, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
    #Changing Names of columns
    #The dataset contains Death (E)
    df_C_T.columns = ['E', 'T']
    df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    #Changing Death to Survival (C)
    df_C_T['C'] = 1 - df_C_T['E']
    #Joining to Main DF
    df = df.join(df_C_T, how='inner')
    
    #Adding target column Y
    df['Y'] = 1
    #We will only take the people who either died or whose time of survival is greater than 365 * years days
    #Essentially we will not take the people who lived, but last contact time is less than 365*years days
    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    #Now for all these people, set Y to 0 if time of survival is less than 365*years days
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    
    #Returning Complete data as a dictionary
    data = {}
    data['C'] = np.asarray(df['C'].tolist(), dtype=np.int32)
    data['Race'] = np.asarray(df['race'].tolist())
    data['Time'] = np.asarray(df['T'].tolist(), dtype=np.float32)
    data['E'] = np.asarray(df['E'].tolist(), dtype= np.int32)
    data['SampleID'] = df.index.values
    data['Features'] = list(df)
    data['Y'] = np.asarray(df['Y'].tolist(), dtype=np.int32)
    data['X'] = df.drop(columns=['C', 'race', 'T', 'E', 'Y']).values.astype('float32')
    #Standardizing Features
    scaler = StandardScaler()
    data['X'] = scaler.fit_transform(data['X'])

    #Selecting K Best features
    data['X'] = SelectKBest(k=200).fit_transform(data['X'], data['Y'])

    return data




def get_methylation_data(cancer_type, target, groups, data_path, genetic_data_path, clinical_outcome_path, years, input_dim):
        #Getting the Tumors from the cancer type
    
    #Some already defined cancer types with their cooreponding tumor types
    CancerTumor_Map = {'GBMLGG': ['GBM', 'LGG'], 'COADREAD': ['COAD', 'READ'], 'KIPAN': ['KIRC', 'KICH', 'KIRP'], 'STES': ['ESCA', 'STAD'], 'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'], 'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'], 'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'], 'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']}
    
    #If not a prespecified cancer type, or a tumor type input, add a new entry to the dictionary with that name
    if cancer_type not in CancerTumor_Map:
        CancerTumor_Map[cancer_type] = [cancer_type]
    
    tumors = CancerTumor_Map[cancer_type]

    # Loading Data Matrix
    MethylationData = loadmat(data_path)
    
    # extracting input combinations data...
    X, Y, GeneName, SampleName = MethylationData['X'].astype('float32'), MethylationData['CancerType'], MethylationData['FeatureName'][0], MethylationData['SampleName']

    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    
    MethylationData_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    MethylationData_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    MethylationData_Y = MethylationData_Y[MethylationData_Y['Disease'].isin(tumors)]
    MethylationData_in = MethylationData_X.join(MethylationData_Y, how='inner')
    MethylationData_in = MethylationData_in.drop(columns=['Disease'])
    
    index = MethylationData_in.index.values
    index_new = [row[:12] for row in index]
    MethylationData_in.index = index_new
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    #print('The shape of fetched methylation data is:')
    #print(MethylationData_in.shape)
    
    # fetching race info from MethylationGenetic.xlsx
    MethyAncsData = [pd.read_excel(genetic_data_path,
                         disease, usecols='A,B',
                         index_col='bcr_patient_barcode',
                         keep_default_na=False)
           for disease in tumors]
    MethyAncsData_race = pd.concat(MethyAncsData)
    race_groups = ['WHITE',
              'BLACK OR AFRICAN AMERICAN',
              'ASIAN',
              'AMERICAN INDIAN OR ALASKA NATIVE',
              'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(race_groups)]
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'WHITE', 'race'] = 'WHITE'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'BLACK OR AFRICAN AMERICAN', 'race'] = 'BLACK'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'ASIAN', 'race'] = 'ASIAN'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'AMERICAN INDIAN OR ALASKA NATIVE', 'race'] = 'NAT_A'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race'] = 'OTHER'
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(groups)]
    #print('The shape of race methylation data is:')
    #print(MethyAncsData_race.shape)
    
    #print(MethyCIDataPath)
    if target=='OS':
        cols = 'A,D,Y,Z'
    elif target == 'DSS':
        cols = 'A,D,AA,AB'
    elif target == 'DFI': # this info is not/very few in methylation data.
        cols = 'A,D,AC,AD'
    elif target == 'PFI':
        cols = 'A,D,AE,AF'
    OutcomeData_M = pd.read_excel(clinical_outcome_path,
                                usecols=cols,dtype={'OS': np.float64},
                                index_col='bcr_patient_barcode')
    
    # adding clinical outcome endpoints data...
    OutcomeData_M.columns = ['G', 'E', 'T']
    OutcomeData_M = OutcomeData_M[OutcomeData_M['E'].isin([0, 1])]
    OutcomeData_M = OutcomeData_M.dropna()
    OutcomeData_M['C'] = 1 - OutcomeData_M['E']
    OutcomeData_M.drop(columns=['E'], inplace=True)
    #print('The shape of outcome methylation data is:')
    #print(OutcomeData_M.shape)
    
    # Keep patients with race information
    MethylationData_in = MethylationData_in.join(MethyAncsData_race, how='inner')
    MethylationData_in = MethylationData_in.dropna(axis='columns')
    #print('The shape of MethylationData_in after race addition is:')
    #print(MethylationData_in.shape)
    
    MethylationData_in = MethylationData_in.join(OutcomeData_M, how='inner')
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    #print('The shape of patients with race and outcome in methylation data is:')
    #print(MethylationData_in.shape)
    
    Data = MethylationData_in

    #Adding target column Y
    Data['Y'] = 1
    #We will only take the people who either died or whose time of survival is greater than 365 * years days
    #Essentially we will not take the people who lived, but last contact time is less than 365*years days
    Data = Data[~((Data['T'] < 365 * years) & (Data['C'] == 1))]
    #Now for all these people, set Y to 0 if time of survival is less than 365*years days
    Data.loc[Data['T'] <= 365 * years, 'Y'] = 0

    
    C = Data['C'].tolist()
    R = Data['race'].tolist()
    G = Data['G'].tolist()
    T = Data['T'].tolist()
    Y = np.asarray(Data['Y'].tolist(), dtype=np.int32)
    E = [1 - c for c in C]
    Data = Data.drop(columns=['C', 'race', 'T', 'G', 'Y'])
    X = Data.values
    X = X.astype('float32')
    PackedData = {'X': X,
                  'Y': Y,
                  'Time': np.asarray(T, dtype=np.float32),
                  'C': np.asarray(C, dtype=np.int32),
                  'E': np.asarray(E, dtype=np.int32),
                  'Race': np.asarray(R),
                  'G': np.asarray(G),
                  'SampleID': Data.index.values,
                  'Features': list(Data)}

    # Feature Selection
    PackedData['X'] = SelectKBest(k=200).fit_transform(PackedData['X'], PackedData['Y'])
    
    return PackedData
