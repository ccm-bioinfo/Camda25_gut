#Librerias que se utilizarán
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

##############################################################################################################################################

def get_fH_fN_dH_dN(meta,tax,sample_name='SampleID',control='Healthy',disease_name='Diagnosis'):
    ######### Recibe los data frames de los metadatos y la taxonomia.
    
    #Se obtienen los id's de las muestras saludables identificadas en los metadatos y después 
    #observamos la taxonomia de las muestras saludables
    healthy_id = meta[meta[disease_name]==control][sample_name]
    tax_healthy = tax[healthy_id]
    
    #Se obtienen los id's de muestras no saludables y despues se observa la taxonmia de estas muestras
    no_healthy_id = meta[meta[disease_name]!=control][sample_name]
    tax_no_healthy = tax[no_healthy_id]
    
    #Se obtienen todas las especies de todas las muestras
    species = tax.index
    
    #Definimos lower para establecer una cota y evitar divisiones entre 0
    lower=1e-05
    
    #Se crea un Data Frame que tendrá las metricas como columnas y a las especies como index
    metrics=pd.DataFrame(index=species,columns=['f_H','f_N','d_H','d_N','PH','PN'])
    
    #Este ciclo obtiene para cada especie m las prevalencias en las muestras saludables p_H y no saludables P_N
    #Posteriormente se  agregan f_H,f_N, d_H y d_N al data frame metric
    for specie in species:
        
        #Se localiza la especie en todas las muestras healthy y se obtiene su presencia absoluta
        specie_in_H=tax_healthy.loc[specie,:]
        abs_pres_H=len(specie_in_H[specie_in_H>lower])
        
        #Se localiza la especie en todas las muestras no-healthy y se obtiene su presencia absoluta
        specie_in_N=tax_no_healthy.loc[specie,:]
        abs_pres_N=len(specie_in_N[specie_in_N>lower])
        
        #Se obtiene PH y PN de la especie, tomando en cuenta que si el resultado es 0, entonces se intercambia por la cota 1e-05
        PH=np.divide(abs_pres_H,len(specie_in_H),out=np.asanyarray(lower),where=(abs_pres_H!=0))
        PN=np.divide(abs_pres_N,len(specie_in_N),out=np.asanyarray(lower),where=(abs_pres_N!=0))
        metrics.loc[specie,:]=[np.divide(PH,PN),np.divide(PN,PH),PH-PN,PN-PH,PH,PN]
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
    
    print('|MH|=', len(MH) )
    print('|MN|=', len(MN))        
    return MH,MN

######### Regresa los conjuntos de especies identificadas beneficiosas MH y dañinas MN, de acuerdo a los parámetros


###______________________________________________________


def get_Psi(set_M,sample):
    ######### Recibe el conjunto M_H o M_N y la muestra con la presencia relativa de cada especie
    
    
    #M_in_sample es el conjunto M_H o M_N intersección las especies presentes en la muestra i
    M_in_sample=set(sample[sample!=0].index) & set_M
    
    #Se calcula la R_M
    R_M_sample=np.divide(len(M_in_sample),len(set_M))
    
    #Se obtiene el array n, que contiene las abundanicas relativas de las especies presentes de M en la muestra i
    #Posteriormente se calcula el logaritmo y la suma
    n=sample[sample!=0][list(M_in_sample)]
    log_n=np.log(n)
    sum_nlnn=np.sum(np.absolute(n*log_n))
    
    #Finalmente se recupera Psi para la muestra i y el conjunto M
    Psi=np.divide(R_M_sample,len(set_M))*sum_nlnn
    
    #Se evita que el caso Psi sea igual a 0 para evitar división entre 0 en la siguiente función. 
    if Psi==0:
        Psi=1e-05
    return Psi

######### Regresa el número Psi asociado a la muestra i y  al conjunto M_H o M_N. 


###______________________________________________________


def get_all_GMHI(tax,MH,MN):
    ######### Se ingresa la taxonomia, el conjunto de especies MH y MN.
    

    #Se crea la variable GMHI, una serie de pandas que tiene como indice el nombre de la muestra y como información su indice GMHI.
    #Esta serie se llenará con un ciclo for, que recorre todas las especies
    samples=tax.columns 
    GMHI=pd.Series(index=samples,name='GMHI',dtype='float64')
    for sample in samples:
        
        #Se obtiene Psi_MH y Psi_MN con la función get_Psi
        Psi_MH=get_Psi(MH,tax[sample])
        Psi_MN=get_Psi(MN,tax[sample])
        
        #Se hace el cociente y se evalua en el logaritmo base 10. Posteriormente se agrega la información a la serie GMHI
        GMHI_sample=np.log10(np.divide(Psi_MH,Psi_MN))
        GMHI[sample]=GMHI_sample
        
    return GMHI 

######### Se regresa la serie con el índice GMHI de cada muestra


###______________________________________________________


def get_accuracy(GMHI,meta,sample_name='SampleID',control='Healthy',disease_name='Diagnosis'):
    #Se recibe el GMHI obtenido y los metadatos para comparar la efectividad de la predicción
    
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
    GMHI=get_all_GMHI(taxonomy,MH,MN)
    accuracy=get_accuracy(GMHI,meta,sample_name,control,disease_name)
    print(accuracy)
    return GMHI,MH,MN,accuracy

######### Devuelve 



###______________________________________________________


def get_MH():
    return {'Alistipes_communis',
 'Alistipes_ihumii',
 'Alistipes_indistinctus',
 'Alistipes_senegalensis',
 'Alistipes_shahii',
 'Alistipes_sp_AF17_16',
 'Anaerococcus_SGB53821',
 'Anaerosacchariphilus_sp_NSJ_68',
 'Anaerotruncus_rubiinfantis',
 'Bacteroidales_bacterium_ph8',
 'Bacteroides_nordii',
 'Bifidobacterium_adolescentis',
 'Bifidobacterium_angulatum',
 'Bifidobacterium_catenulatum',
 'Blautia_glucerasea',
 'Blautia_luti',
 'Blautia_massiliensis',
 'Butyricicoccus_sp_AM29_23AC',
 'Butyricimonas_SGB15260',
 'Butyricimonas_faecihominis',
 'Butyricimonas_paravirosa',
 'Butyrivibrio_crossotus',
 'Candidatus_Borkfalkia_ceftriaxoniphila',
 'Candidatus_Gastranaerophilales_bacterium',
 'Candidatus_Pararuminococcus_gallinarum',
 'Catenibacterium_SGB4425',
 'Christensenellaceae_bacterium_NSJ_63',
 'Clostridiaceae_bacterium_AF18_31LB',
 'Clostridiaceae_bacterium_Marseille_Q4143',
 'Clostridiaceae_bacterium_Marseille_Q4145',
 'Clostridiaceae_bacterium_Marseille_Q4149',
 'Clostridiaceae_unclassified_SGB15090',
 'Clostridiaceae_unclassified_SGB4771',
 'Clostridiales_Family_XIII_bacterium_BX16',
 'Clostridium_SGB4648',
 'Clostridium_sp_1001270H_150608_G6',
 'Clostridium_sp_AF12_28',
 'Clostridium_sp_AF27_2AA',
 'Clostridium_sp_AF34_13',
 'Clostridium_sp_AF36_4',
 'Clostridium_sp_AM33_3',
 'Clostridium_sp_AM49_4BH',
 'Collinsella_SGB14861',
 'Coprobacter_fastidiosus',
 'Coprococcus_SGB4669',
 'Coprococcus_eutactus',
 'Enterocloster_SGB14313',
 'Eubacteriaceae_bacterium',
 'Eubacteriales_unclassified_SGB15145',
 'Eubacterium_sp_AF16_48',
 'Eubacterium_ventriosum',
 'Faecalibacterium_SGB15315',
 'Faecalicatena_fissicatena',
 'Firmicutes_bacterium_AF16_15',
 'GGB13404_SGB14252',
 'GGB2653_SGB3574',
 'GGB2970_SGB3952',
 'GGB2982_SGB3964',
 'GGB2998_SGB3988',
 'GGB2998_SGB3989',
 'GGB3109_SGB4643',
 'GGB3175_SGB4191',
 'GGB3278_SGB4328',
 'GGB32900_SGB53446',
 'GGB33469_SGB15236',
 'GGB33469_SGB15237',
 'GGB33512_SGB15201',
 'GGB33586_SGB53517',
 'GGB33928_SGB15225',
 'GGB3570_SGB4777',
 'GGB3612_SGB4882',
 'GGB3614_SGB4886',
 'GGB3619_SGB4894',
 'GGB3653_SGB4964',
 'GGB3730_SGB5060',
 'GGB3815_SGB5180',
 'GGB4566_SGB6305',
 'GGB4571_SGB6317',
 'GGB47687_SGB2286',
 'GGB51269_SGB5062',
 'GGB51884_SGB49168',
 'GGB52130_SGB14966',
 'GGB79630_SGB13983',
 'GGB79734_SGB15291',
 'GGB79973_SGB14341',
 'GGB80011_SGB15267',
 'GGB80140_SGB15224',
 'GGB9062_SGB13981',
 'GGB9063_SGB13982',
 'GGB9189_SGB14128',
 'GGB9296_SGB14253',
 'GGB9342_SGB14306',
 'GGB9345_SGB14311',
 'GGB9495_SGB14892',
 'GGB9512_SGB14909',
 'GGB9524_SGB14924',
 'GGB9531_SGB14932',
 'GGB9534_SGB14937',
 'GGB9602_SGB15028',
 'GGB9602_SGB15031',
 'GGB9611_SGB15045',
 'GGB9614_SGB15049',
 'GGB9616_SGB15051',
 'GGB9616_SGB15052',
 'GGB9635_SGB15103',
 'GGB9635_SGB15106',
 'GGB9699_SGB15216',
 'GGB9707_SGB15229',
 'GGB9708_SGB15233',
 'GGB9708_SGB15234',
 'GGB9713_SGB15249',
 'GGB9747_SGB15356',
 'GGB9758_SGB15368',
 'GGB9759_SGB15370',
 'GGB9760_SGB15373',
 'GGB9760_SGB15374',
 'GGB9775_SGB15395',
 'Gemmiger_SGB15295',
 'Gemmiger_SGB15299',
 'Gemmiger_formicilis',
 'Lachnospira_sp_NSJ_43',
 'Lachnospiraceae_bacterium_8_1_57FAA',
 'Lachnospiraceae_bacterium_AM48_27BH',
 'Lachnospiraceae_bacterium_Marseille_Q4251',
 'Lachnospiraceae_bacterium_OM04_12BH',
 'Lawsonibacter_sp_NSJ_51',
 'Mediterraneibacter_butyricigenes',
 'Methylobacterium_SGB15164',
 'Nitrosopumilus_SGB14899',
 'Oscillibacter_sp_ER4',
 'Oscillospiraceae_bacterium',
 'Oscillospiraceae_bacterium_Marseille_Q3528',
 'Oscillospiraceae_unclassified_SGB15256',
 'Parabacteroides_goldsteinii',
 'Parabacteroides_johnsonii',
 'Pseudoruminococcus_massiliensis',
 'Rheinheimera_SGB14999',
 'Roseburia_sp_AF02_12',
 'Roseburia_sp_AM16_25',
 'Roseburia_sp_AM59_24XD',
 'Roseburia_sp_BX1005',
 'Ruminococcus_SGB4421',
 'Ruminococcus_bromii',
 'Ruminococcus_callidus',
 'Ruminococcus_sp_BSD2780120874_150323_B10',
 'Streptococcus_SGB14137',
 'Sutterella_wadsworthensis',
 'Vescimonas_coprocola',
 'Victivallis_lenta'}



def get_MN():
    return {'Anaerostipes_caccae',
 'Anaerotruncus_colihominis',
 'Atopobium_parvulum',
 'Bifidobacterium_dentium',
 'Blautia_hansenii',
 'Blautia_producta',
 'Citrobacter_freundii',
 'Clostridiales_bacterium_1_7_47FAA',
 'Clostridium_asparagiforme',
 'Clostridium_bolteae',
 'Clostridium_butyricum',
 'Clostridium_citroniae',
 'Clostridium_clostridioforme',
 'Clostridium_hathewayi',
 'Clostridium_nexile',
 'Clostridium_paraputrificum',
 'Clostridium_ramosum',
 'Clostridium_spiroforme',
 'Clostridium_symbiosum',
 'Eggerthella_lenta',
 'Enterocloster_aldenensis',
 'Enterocloster_clostridioformis',
 'Erysipelatoclostridium_ramosum',
 'Erysipelotrichaceae_bacterium_2_2_44A',
 'Escherichia_marmotae',
 'Faecalimonas_umbilicata',
 'Flavonifractor_plautii',
 'Fusobacterium_nucleatum',
 'Gemella_morbillorum',
 'Gemella_sanguinis',
 'Granulicatella_adiacens',
 'Holdemania_filiformis',
 'Klebsiella_pneumoniae',
 'Lachnospiraceae_bacterium_1_4_56FAA',
 'Lachnospiraceae_bacterium_2_1_58FAA',
 'Lachnospiraceae_bacterium_3_1_57FAA_CT1',
 'Lachnospiraceae_bacterium_5_1_57FAA',
 'Lachnospiraceae_bacterium_9_1_43BFAA',
 'Lactobacillus_salivarius',
 'Mediterraneibacter_glycyrrhizinilyticus',
 'Peptostreptococcus_stomatis',
 'Proteus_mirabilis',
 'Ruminococcaceae_bacterium_D16',
 'Ruminococcus_gnavus',
 'Solobacterium_moorei',
 'Streptococcus_anginosus',
 'Streptococcus_australis',
 'Streptococcus_gordonii',
 'Streptococcus_infantis',
 'Streptococcus_mitis_oralis_pneumoniae',
 'Streptococcus_sanguinis',
 'Streptococcus_vestibularis',
 'Subdoligranulum_sp_4_3_54A2FAA',
 'Subdoligranulum_variabile',
 'Tyzzerella_nexilis',
 'Veillonella_atypica',
 'Veillonellaceae_bacterium',
 'candidate_division_TM7_single_cell_isolate_TM7c'}