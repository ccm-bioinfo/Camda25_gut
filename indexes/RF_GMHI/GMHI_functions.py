#Librerias que se utilizarán
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

import multiprocessing
##############################################################################################################################################

def get_fH_fN_dH_dN(meta,tax,sample_name='SampleID',control='Healthy',disease_name='Diagnosis'):
    ######### Recibe los data frames de los metadatos y la taxonomia.
    

    # Obtener IDs saludables y no saludables
    healthy_id = meta[meta[disease_name] == control][sample_name]
    no_healthy_id = meta[meta[disease_name] != control][sample_name]

    # Subset de taxonomía
    tax_healthy = tax[healthy_id]
    tax_no_healthy = tax[no_healthy_id]

    # Cota para evitar divisiones por 0
    lower = 1e-5

    # Presencia absoluta por especie (conteo de columnas con abundancia > lower)
    abs_pres_H = (tax_healthy > lower).sum(axis=1)
    abs_pres_N = (tax_no_healthy > lower).sum(axis=1)

    # Número total de muestras en cada grupo
    total_H = tax_healthy.shape[1]
    total_N = tax_no_healthy.shape[1]

    # Prevalencias PH y PN (con np.maximum para aplicar la cota mínima)
    PH = np.maximum(abs_pres_H / total_H, lower)
    PN = np.maximum(abs_pres_N / total_N, lower)

    # Cálculo vectorizado de las métricas
    metrics = pd.DataFrame({
    'f_H': PH / PN,
    'f_N': PN / PH,
    'd_H': PH - PN,
    'd_N': PN - PH,
    'PH': PH,
    'PN': PN
}, index=tax.index)

    return metrics

######### Regresa un DataFrame en el que para cada especie se obtienen sus metricas f_H,f_N,d_H y d_N



###______________________________________________________


def get_MH_MN(metrics,theta_f_H,theta_d_H,theta_f_N,theta_d_N):
    ######### Recibe el conjunto de metricas para cada especie y los parámetros de comparación
    
    
    #Se obtienen las especies beneficiosas que son mayores a los parametros theta_f y theta_d
    health_species_pass_theta_f=set(metrics[metrics['f_H']>=theta_f_H].index)
    health_species_pass_theta_d=set(metrics[metrics['d_H']>=theta_d_H].index)
    
    #Se obtienen las especies dañinas que son mayores a los parametros theta_f y theta_d
    no_health_species_pass_theta_f=set(metrics[metrics['f_N']>=theta_f_N].index)
    no_health_species_pass_theta_d=set(metrics[metrics['d_N']>=theta_d_N].index)
    
    #Se definen los conjuntos de las especies beneficiosas y dañinas que superan ambos parámetros
    MH=health_species_pass_theta_f & health_species_pass_theta_d
    MN=no_health_species_pass_theta_f & no_health_species_pass_theta_d
    
    return MH,MN

######### Regresa los conjuntos de especies identificadas beneficiosas MH y dañinas MN, de acuerdo a los parámetros


###______________________________________________________


def get_all_GMHI(tax, MH, MN,train=False):
    if len(MH) == 0 or len(MN) == 0:
        return pd.Series(data=0, index=tax.columns, name='GMHI', dtype='float64')

    
    taxonomy=tax.copy()
    
    if train=False:
        taxonomy=taxonomy/taxonomy.sum()
        taxonomy=taxonomy[taxonomy.gt(0.01).any(axis=1)]


    def vectorized_get_Psi(tax_sub, M):
        # Subconjunto de abundancias para especies en M
        tax_M = tax_sub.loc[M]
        
        # Máscara binaria: presencia de especie (> 0)
        present_mask = tax_M > 0

        # R_M: número de especies de M presentes en cada muestra / |M|
        R_M = present_mask.sum(axis=0) / len(M)

        # n * log(n), con log seguro (ignora ceros)
        nlogn = np.abs(tax_M * np.log(np.where(tax_M > 0, tax_M, 1)))  # log(1) = 0

        sum_nlogn = nlogn.sum(axis=0)

        # Psi = (R_M / |M|) * sum(n*log(n))
        Psi = (R_M / len(M)) * sum_nlogn

        # Evitar ceros
        Psi = np.maximum(Psi, 1e-5)
        return Psi

    Psi_MH = vectorized_get_Psi(taxonomy, list(MH))
    Psi_MN = vectorized_get_Psi(taxonomy, list(MN))

    GMHI = np.log10(Psi_MH / Psi_MN)
    GMHI.name = 'GMHI'
    return GMHI


######### Se regresa la serie con el índice GMHI de cada muestra


###______________________________________________________


def get_accuracy(GMHI,meta,sample_name='SampleID',control='Healthy',disease_name='Diagnosis'):
    #Se recibe el GMHI obtenido y los metadatos para comparar la efectividad de la predicción
    if GMHI.max()==0 and GMHI.min()==0:
        return [0,0]

#     #En diagnosis se almacenan los diagnosticos de cada muestra
    diagnosis=meta[disease_name].copy(deep=False)
    diagnosis.index=meta[sample_name]

#     #Se evaluan los verdaderos positivos y los verdaderos negativos. Posteriormente se obtiene el accuracy
    true_positive=GMHI[diagnosis==control][GMHI>0].count()
    true_negative=GMHI[diagnosis!=control][GMHI<0].count()
    accuracy=np.divide(true_positive+true_negative,GMHI.count())
    
    
    #return balanced_accuracy_score(['Unhealthy' if x != 'Healthy' else 'Healthy' for x in meta['Diagnosis']], ['Unhealthy' if x < 0 else 'Healthy' for x in list(GMHI)])
    return accuracy, balanced_accuracy_score(['Unhealthy' if x != control else control for x in meta[disease_name]], ['Unhealthy' if x < 0 else control for x in list(GMHI)])

######### Se regresa el accuracy obtenido



###______________________________________________________


def train_GMHI(meta,tax,theta_f_H,theta_d_H,theta_f_N,theta_d_N,sample_name='SampleID',control='Healthy',disease_name='Diagnosis'):

    taxonomy=tax.copy()
    taxonomy=taxonomy/taxonomy.sum()
    taxonomy=taxonomy[taxonomy.gt(0.01).any(axis=1)]
    
    metrics=get_fH_fN_dH_dN(meta,taxonomy,sample_name,control,disease_name)
    MH,MN=get_MH_MN(metrics,theta_f_H,theta_d_H,theta_f_N,theta_d_N)

    if len(MH)==0 or len(MN)==0:
        return pd.Series(data=0,index=tax.index,name='GMHI',dtype='float64'),set(),set(),[0,0]
    
    
    GMHI=get_all_GMHI(taxonomy,MH,MN,train=True)
    accuracy=get_accuracy(GMHI,meta,sample_name,control,disease_name)
    return GMHI,MH,MN,accuracy

######### Devuelve 



###______________________________________________________ 




def model_per_param(params):
    p1,p2,p3,p4=params[2:6]
    p1_p2=str(p1)+','+str(p2)
    p3_p4=str(p3)+','+str(p4)
    index_name=p1_p2+','+p3_p4
    
    df_metrics=params[-1]

    if df_metrics[df_metrics['p1,p2']==p1_p2].size!=0 and df_metrics[df_metrics['p3,p4']==p3_p4].size!=0:
        
        
        MH=df_metrics[df_metrics['p1,p2']==p1_p2]['MH_MN_GMHI'].values[0][0]
        MN=df_metrics[df_metrics['p3,p4']==p3_p4]['MH_MN_GMHI'].values[0][1]
        #####REVISAR
        GMHI=get_all_GMHI(params[1],MH,MN)
        metrics=get_accuracy(GMHI,params[0],params[6],params[7],params[8])
    else:
        df_metrics[df_metrics['p1,p2']==p1_p2].size==0 and df_metrics[df_metrics['p3,p4']==p3_p4].size==0
        #print(params[0],params[1],p1,p2,p3,p4,params[6],params[7],params[8])
        GMHI,MH,MN,metrics=train_GMHI(params[0],params[1],p1,p2,p3,p4,params[6],params[7],params[8])

    p1_p2=str(p1)+','+str(p2)
    p3_p4=str(p3)+','+str(p4)
    row=[p1_p2,p3_p4]
    
    row.extend([metrics[0],metrics[1],[MH,MN,GMHI]])
    
    df_metrics.loc[p1_p2+','+p3_p4,]=row
    return p1_p2+','+p3_p4, row



def gmhi_pairwise_gridsearch(meta, tax, sample_name, diagnosis_col_name,p1_lower=0,p2_lower=0,p3_lower=0,p4_lower=0,p1_upper=0.7,p2_upper=0.7,p3_upper=0.7,p4_upper=0.7,grid_len=7,only_pair=False):

    all_diagnosis=meta[diagnosis_col_name].unique()
    cols_and_index = np.append(all_diagnosis, 'All')
    df_accuracy = pd.DataFrame(index=cols_and_index, columns=cols_and_index, data=np.nan)
    df_f1 = pd.DataFrame(index=cols_and_index, columns=cols_and_index, data=np.nan)

    df_GMHI_accuracy = pd.DataFrame(index=tax.columns)
    df_GMHI_f1 = pd.DataFrame(index=tax.columns)

    dict_best_sets_accuracy = {}
    dict_best_sets_f1 = {}

    list_pairs = []
    for first_condition in all_diagnosis:
        for second_condition in all_diagnosis:
                

            pair=set([first_condition,second_condition])

            if pair in list_pairs:
                continue

            if first_condition == second_condition:
                if only_pair==True:
                    continue
                vs_name = f'{first_condition} vs All'
                meta_first_and_second_condition = meta.copy()
                tax_first_and_second_condition = tax.copy()
                second_condition = 'All'
            else:
                vs_name=first_condition+' vs '+second_condition
                meta_first_and_second_condition=meta[meta[diagnosis_col_name].isin([first_condition,second_condition])].copy()
                tax_first_and_second_condition=tax[meta_first_and_second_condition[sample_name]].copy()

            list_pairs.append(pair)

            df_metrics = pd.DataFrame(columns=['p1,p2', 'p3,p4', 'accuracy', 'f1', 'MH_MN_GMHI'])

            param_grid = [[meta_first_and_second_condition, tax_first_and_second_condition, p1, p2, p3, p4,
                           sample_name, first_condition, diagnosis_col_name, df_metrics]
                          for p1 in np.linspace(p1_lower, p1_upper, num=grid_len)
                          for p2 in np.linspace(p2_lower, p2_upper, num=grid_len)
                          for p3 in np.linspace(p3_lower, p3_upper, num=grid_len)
                          for p4 in np.linspace(p4_lower, p4_upper, num=grid_len)]

            for params in param_grid:
                row=model_per_param(params)
                df_metrics.loc[row[0],]=row[1]

            best_param_for_accuracy = df_metrics['accuracy'].idxmax()
            best_param_for_f1 = df_metrics['f1'].idxmax()

            print(f'Los mejores parámetros para {vs_name} son '
                  f'accuracy: {best_param_for_accuracy} | f1: {best_param_for_f1}')

            best_GMHI_sets_for_accuracy = df_metrics.loc[best_param_for_accuracy, 'MH_MN_GMHI'][:2]
            best_GMHI_sets_for_f1 = df_metrics.loc[best_param_for_f1, 'MH_MN_GMHI'][:2]

            df_accuracy.loc[first_condition, second_condition] = df_metrics.loc[best_param_for_accuracy, 'accuracy']
            df_f1.loc[first_condition, second_condition] = df_metrics.loc[best_param_for_f1, 'f1']

            df_accuracy.loc[second_condition, first_condition] = df_accuracy.loc[first_condition, second_condition]
            df_f1.loc[second_condition, first_condition] = df_f1.loc[first_condition, second_condition]

            df_GMHI_accuracy[vs_name] = df_metrics.loc[best_param_for_f1, 'MH_MN_GMHI'][2]
            df_GMHI_f1[vs_name] = df_metrics.loc[best_param_for_accuracy, 'MH_MN_GMHI'][2]

        
            dict_best_sets_accuracy[vs_name]={first_condition:best_GMHI_sets_for_accuracy[0],second_condition:best_GMHI_sets_for_accuracy[1]}
            dict_best_sets_f1[vs_name]={first_condition:best_GMHI_sets_for_f1[0],second_condition:best_GMHI_sets_for_f1[1]}


            if only_pair==True:
                return df_accuracy.loc[first_condition, second_condition],df_f1.loc[first_condition, second_condition],df_GMHI_accuracy[vs_name],df_GMHI_f1[vs_name],dict_best_sets_accuracy[vs_name],dict_best_sets_f1[vs_name]


    

    return df_accuracy, df_f1, df_GMHI_accuracy, df_GMHI_f1, dict_best_sets_accuracy, dict_best_sets_f1
