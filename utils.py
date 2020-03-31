import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def ordered_dict_values(dictionary):
    """
    A function to obtain unique values from a dictionary
    
    Parameters
    ----------
    dictionary: dict
        Dictionary to obtain unique values from 
    
    Returns
    -------
    lst: list
        A list of unique values in the dictionary by order of appearance
    """
    lst = []
    for v in dictionary.values():
        if v not in lst:
            lst.append(v)
    return lst

def get_dtype(df):
    """
    A function to get the datatype for each column in a dataframe
    
    Parameters
    ----------
    df: pandas.DataFrame
        A dataframe to extract datatypes from 
    
    Returns
    -------
    dtype: dict
        A dictionary containing 2 lists: column and dtype - datatype 
    """
    dtype = {'column':[],'dtype':[]}
    for column in df.columns:
        dtype['column'].append(column)
        dtype['dtype'].append(df[column].dtype)
    dtype = pd.DataFrame(dtype)
    return dtype

def clean_df(df,schema,debug=False):
    """
    A function to do some basic data cleaning using a provided schema.
    
    The following steps are performed in order:
    - variable names are convert to lowercase
    - spacing replaced with '_'
    - miscellaneous replacements of categorical variable names/missing values that don't work with scehma
    - loop through each variable name:
    a. exclude variables which are specified in the schema doc
    b. expand any nested lists
    c. replace missing values with specified values
    d. if value is less than minimum value, set to missing
    e. if value is greater than maximum value, set to missing
    f. enforce variable type for string/categorical variables
    g. enforce datetime variable type
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to be cleaned
    schema: pandas.DataFrame
        A dataframe containing schema information including: variable name, supposed dtype, missing value indicator, max and min ranges
    debug: bool
        A flag used for debugging
        
    Returns
    -------
    df: pandas.DataFrame
        The cleaned dataframe
    
    """
    #Clean the names
    df.columns = [name.lower() for name in df.columns]
    df.columns = [name.replace(' ','_') for name in df.columns]
    df = df[[name for name in df.columns if 'unnamed' not in name]]
    df['tropi'] = df['tropi'].replace({999.:np.nan})
    df['stenttype'] = df['stenttype'].replace({999.0:'4.0'})
    for var in schema.varname:
        if var in df.columns:
            index = schema['varname']==var
            series = df[var]

            if schema['type'][index].values[0] == 'exclude':
                df = df.drop(var,axis=1)
            elif schema['dtype'][index].values[0] == 'nested':
                expanded = series.str.split(',',expand=True)
                expanded.columns = [var+str(col) for col in expanded.columns]
                df = df.drop(var,axis=1).join(expanded)
            else:
                series = series.replace({schema['missing_code'][index].values[0]:np.nan,999:np.nan})
                series = series.fillna(schema['impute_value'][index].values[0])

                if schema['dtype'][index].values[0] in ['float64','numeric','timeto','category']:
                    series = pd.to_numeric(series,errors='coerce')

                if schema['min'][index].values[0] == schema['min'][index].values[0]:
                    series[series < schema['min'][index].values[0]] = np.nan

                if schema['max'][index].values[0] == schema['max'][index].values[0]:
                    series[series > schema['max'][index].values[0]] = np.nan

                if schema['dtype'][index].values[0] in ['category','object','str','freetext']:
                    series = series.apply(lambda row: str(row) if row==row else np.nan)
                elif schema['dtype'][index].values[0] == 'datetime':
                    series = pd.to_datetime(series,errors='coerce')

                df[var] = series            
    
    return df

def get_data():
    """
    A wrapper function to read csv datasets, perform cleaning and replaced categorical levels using the var_dict
    
    Variables which are commented out are removed intentionally from the analysis
    
    Parameters
    ----------
    None
    
    Returns
    -------
    combined: pandas.DataFrame
        A dataframe containing the dataset
    var_list: list
        A list of strings containing variable names
    cat_features_list: list
        A list of strings containing categorical variables
    cat_order: 
        A dictionary of lists indicating the order of appearance for each variable - used for Table 1
    """
    
    schema = pd.read_csv('raw_data/schema.csv')
    nstemi = pd.read_csv('raw_data/nstemi.csv')
    nstemi = clean_df(nstemi,schema)
    nstemi['acs_type'] = 'NSTEMI'
    stemi = pd.read_csv('raw_data/stemi.csv')
    stemi = clean_df(stemi,schema)
    stemi['acs_type'] = 'STEMI'
    combined = pd.concat([nstemi,stemi],axis=0)
    combined = combined.reset_index(drop=True)

    print(f'Cohort Size: {len(combined)}')

    valve_dict = {'0':'Absent','1':'Present','2':'Present','3':'Present','4':'Present','5':'Present'}
    var_dict = {'age':{'display':'Age, years'},
                'gender':{'display':'Sex','replace':{'0':'Male','1':'Female'}},
                'height':{'display':'Height, cm'},
                'weight':{'display':'Weight, kg'},
                'bmi':{'display':'Body Mass Index'},
                #'race':{'display':'Ethnicity','replace':{'0':'Chinese','1':'Malay','2':'Indian','3':'Others'}},
                #'smoking':{'display':'Smoking','replace':{'0.0':'No','1.0':'Yes'}},
                #'alcohol':{'display':'Alcohol Use','replace':{'0':'No Alcohol Use','1':'Previous Alcohol Use','2':'Current Alcohol Use'}},
                ##Past Medical History
                #'af':{'display':'Atrial Fibrillation','replace':{'0':'No','1':'Yes'}},
                #'prevaccstatus':{'display':'Existing Anticoagulation Use'},
                #'prevacctype':{'display':'Existing Anticoagulation Before Diagnosis Of LV Thrombus','replace':{'0':'No Existing Anticoagulation','1':'Warfarin','2':'Novel Oral Anticoagulant'}},
                #'htn':{'display':'Hypertension','replace':{'0':'No','1':'Yes'}},
                #'hld':{'display':'Hyperlipidemia','replace':{'0':'No','1':'Yes'}},
                'dm':{'display':'Diabetes Mellitus/Prediabetes','replace':{'0':'No','1':'Yes','2':'Yes','3':'Yes'}},
                'ckd':{'display':'Chronic Kidney Disease','replace':{'0':'No','1':'Yes','2':'Yes','3':'Yes'}},
                #'cld':{'display':'Chronic Liver Disease','replace':{'0':'No','1':'Yes'}},
                #'pvd':{'display':'Peripheral Vascular Disease','replace':{'0':'No Peripheral Vascular Disease','1':'Intermittent Claudication','2':'Critical Limb Ischemia'}},
                'vte':{'display':'Venous Thromboembolism','replace':{'0':'No','1':'Yes'}},
                'stroke':{'display':'Cerebrovascular Accident/Transient Ischemic Attack','replace':{'0':'No','1':'Yes','2':'Yes'}},
                #'asthma':{'display':'Asthma','replace':{'0':'No','1':'Yes'}},
                #'copd':{'display':'Chronic Obstructive Pulmonary Disease','replace':{'0':'No','1':'Yes'}},
                #'malignancy':{'display':'Malignancy','replace':{'0':'No','1':'Yes'}},
                #'prevacs':{'display':'Prior Acute Coronary Syndrome','replace':{'0':'No Previous ACS','1':'Non-STE ACS','2':'STEMI','3':'Other Ischemic Heart Disease'}},
                'heartfailure':{'display':'Heart Failure','replace':{'0':'No','1':'Yes'}},
                ##Post MI Complications
                #'postmiarrhythmia':{'display':'Post-AMI Arrhythmia','replace':{'0':'No','2':'Pulseless Ventricular Tachycardia/Ventricular Fibrillation','5':'Pulseless Electrical Activity','1':'Ventricular Tachycardia','3':'Complete Heart Block','4':'Bradycardia'}},
                'newaf':{'display':'Post-AMI Atrial Fibrillation','replace':{'0':'No','1':'Yes'}},
                'cardiogenic_shock':{'display':'Post-AMI Cardiogenic Shock','replace':{'0':'No','1':'Yes'}},
                'cpr':{'display':'Cardiopulmonary Resuscitation','replace':{'0':'No','1':'Yes','0.0':'No','1.0':'Yes'}},
                ##Labs at point of diagnosis 
                'tropi':{'display':'Peak Troponin I, ng/dL'},
                'hb':{'display':'Hemoglobin, g/dL'},
                'tw':{'display':'White Blood Cell Count, 10^9/L'},
                'lymphocyte':{'display':'Lymphocyte Count, 10^9/L'},
                'neutrophil':{'display':'Neutrophil Count, 10^9/L'},
                'plt':{'display':'Platelet Count, 10^9/dL'},
                'pt':{'display':'Prothrombin Time, seconds'},
                'inr':{'display':'International Normalized Ratio'},
                'aptt':{'display':'Activated Partial Thromboplastin Time, seconds'},
                #'ast':{'display':'Aspartate Aminotransferase, U/L'},
                #'alt':{'display':'Alanine Aminotransferase, U/L'},
                #'alp':{'display':'Alkaline Phosphatase, U/L'},
                #'egfr':{'display':'Estimated Glomerular Filtration Rate'},
                'creatinine':{'display':'Creatinine, mmol/L'},
                #MI Characteristics
                'areaofinfarct':{'display':'ACS Type','replace':{'0':'NSTEMI',
                                                                 '1':'STEMI',
                                                                 '2':'STEMI',
                                                                 '3':'STEMI',
                                                                 '4':'STEMI',
                                                                 '5':'STEMI',
                                                                 '6':'STEMI',
                                                                 '7':'STEMI',
                                                                 '8':'STEMI',
                                                                 '9':'STEMI'}},
                                                                        #'1':'Anterior',
                                                                        #'2':'Anterolateral',
                                                                        #'3':'Anteroseptal',
                                                                        #'4':'Anteroinferior',
                                                                        #'5':'Lateral',
                                                                        #'6':'Inferior',
                                                                        #'7':'Inferoposterior',
                                                                        #'8':'Inferolateral',
                                                                        #'9':'Posterior'}},
                #'anterior':{'display':'Anterior Infarct'},
                #'septal':{'display':'Septal Infarct'},
                #'lateral':{'display':'Lateral Infarct'},
                #'posterior':{'display':'Posterior Infarct'},
                #'inferior':{'display':'Inferior Infarct'},
                #Echo characteristics
                'ef':{'display':'Visual Ejection Fraction, %'},
                'lvidd/mm':{'display':'Left Ventricle Internal Diameter At End-diastole, mm'},
                'lvids/mm':{'display':'Left Ventricle Internal Diameter At End-systole, mm'},
                'lvotsize/mm':{'display':'Left Ventricle Outflow Tract, mm'},
                #'mr':{'display':'Mitral Regurgitation','replace':valve_dict},
                #'ms':{'display':'Mitral Stenosis','replace':valve_dict},
                #'ar':{'display':'Aortic Regurgitation','replace':valve_dict},
                #'as':{'display':'Aortic Stenosis','replace':valve_dict},
                #'tr':{'display':'Tricuspid Regurgitation','replace':valve_dict},
                #'ts':{'display':'Tricuspid Stenosis','replace':valve_dict},
                #'pr':{'display':'Pulmonary Regurgitation','replace':valve_dict},
                #'ps':{'display':'Pulmonary Stenosis','replace':valve_dict},
                'wall_motion_abn_(absent_=_0,_regional_=_1,_global_=_2)':{'display':'Wall Motion Abnormality','replace':{'0.0':'None','1.0':'Regional','2.0':'Global','1':'Regional','2':'Global'}},
                #'wall_affected_(apex_=_1,_anterior_=_2,_septal_=_3,_inferior_=_4,_lateral_=_5)':{'display':'Apical Wall Motion Abnormality','replace':{'1':'Yes','2':'No','3':'No','4':'No','5':'No'}},
                'lvaneurysm':{'display':'Left Ventricular Aneurysm','replace':{'0':'No','1':'Yes'}},
                'mobility':{'display':'LV Thrombus Mobility','replace':{'0':'No','1':'Yes'}},
                'protrusion':{'display':'Protrusion','replace':{'0':'No','1':'Yes'}},
                #'lvtdiameter':{'display':'LV Thrombus Maximal Diameter, cm'},
                #MI Treatment
                'aspirin':{'display':'Aspirin Use','replace':{'0.0':'No','1.0':'Yes'}},
                '2ndantiplatelet':{'display':'Second Antiplatelet Agent','replace':{'0.0':'No','1.0':'Yes','2.0':'Yes','3.0':'Yes','4.0':'Yes'}},
                #'coronaryangiogram':{'display':'Coronary Angiogram Performed','replace':{'0':'No','1':'Yes'}},
                'cad':{'display':'Coronary Artery Disease','replace':{'0.0':'No Vessel Disease','1.0':'Single Vessel Disease','2.0':'Double Vessel Disease','3.0':'Triple Vessel Disease'}},    
                'n_of_culprit_a':{'display':'Number of Culprit Arteries','replace':{0.0:'None',1.0:'One',2.0:'Two',3.0:'Three'}},
                'revascularisation':{'display':'Revascularization Procedure','replace':{'0':'No','1':'Yes','2':'Yes','3':'Yes'}},
                #'stenttype':{'display':'Type Of Stent Used','replace':{'4.0':'None','0.0':'POBA','1.0':'Drug-eluting Stent','2.0':'Bare Metal Stent','3.0':'Bioabsorbable Vascular Stent'}},
                #'chadsvasc':{'display':'CHADS-Vasc Score'},
                #'hasbled':{'display':'HASBLED score'},
                ##Tx for LV thrombus
                #'heparinbridging':{'display':'Heparin Bridging Therapy','replace':{'0.0':'No Heparin Bridging','1.0':'Low Molecular Weight Heparin','2.0':'Intravenous Heparin',
                #                                                                   '0':'No Heparin Bridging','1':'Low Molecular Weight Heparin','2':'Intravenous Heparin'}},
                #'acc_@time_of_lvt_dx_(0_=_no,_1_=_yes)':{'display':'Anticoagulation Clinic Followup At Time of Diagnosis'}
                'anticoagulation':{'display':'Anticoagulation After LV Thrombus Diagnosis','replace':{'3.0':'No Anticoagulation','0.0':'Warfarin','1.0':'Novel Oral Anticoagulant','2.0':'Heparin'}},
                'followupduration':{'display':'Followup Duration, days'}}

    def tidy(df,var_dict):
        """
        Subfunction to extract variable names, categorical variables and get order of display of categorical levels for Table 1
        
        Parameters
        ----------
        df: pandas.DataFrame
            The dataset to be tidied
        var_dict:
            A nested dictionary containing original variable names as keys and a dictionary of display name and dictionary to replace categorical values
        """
        var_list = ['lvtstatus','lvtrecurrence','dateofdeath','repeat_scan_date','finalscandate']
        cat_features = []
        cat_order = {}
        for varname in var_dict:
            display_name = var_dict[varname].get('display')
            replace_dict = var_dict[varname].get('replace',None)
            if replace_dict is not None:
                try:
                    df[varname] = df[varname].apply(str)
                    df[varname] = df[varname].replace(replace_dict)
                    df[varname] = df[varname].replace({'nan':np.nan})
                except:
                    print(varname)
                    raise 
                cat_features.append(display_name)
                cat_order[display_name] = ordered_dict_values(replace_dict)
            df = df.rename({varname:display_name},axis=1)
            var_list.append(display_name)
        df = df[var_list]
        return df,var_list,cat_features, cat_order
    
    combined,var_list,cat_features_list,cat_order = tidy(combined,var_dict)
    
    return combined, var_list,cat_features_list,cat_order

def apply_exclusions(combined,var_list,cat_features_list,cat_order,exclude_death):
    """
    A function used to apply exclusion criteria to the dataset
    
    Parameters
    ----------
    combined: pandas.DataFrame
        The dataset
    var_list: list
        List of variable names in the dataset
    cat_features: list
        List of variable names for categorical features
    cat_order: list
        Dictionary with variable names as keys and values of lists indicating order of appearance for categorical features
    exclude_death: bool
        A flag to indicate if patietns who died should be excluded
    
    Returns
    -------
    combined: pandas.DataFrame
        Dataset after applying exclusion criteria
    var_list: list
        List of variable names after applying exclusion criteria
    cat_features: list
        List of categorical features after applying exclusion criteria
    cat_order: list
        Dictionary with variable names as keys and values of lists indicating order of appearance for categorical features 
        after applying exclusion criteria
    """
    
    
    if exclude_death == True:
        print('Processing dataset excluding patients who died:')
        outcome_string = 'Unresolved LVT'
        combined['ddeath'] = pd.to_datetime(combined['dateofdeath'],errors='coerce')
        combined['repeat_scan_date'] = pd.to_datetime(combined['repeat_scan_date'],errors='coerce')
        combined['finalscandate'] = pd.to_datetime(combined['finalscandate'],errors='coerce')
        combined['diedbeforerepeatscan'] = combined.apply(lambda row: pd.isnull(row['ddeath'])==False and (pd.isnull(row['repeat_scan_date']) and pd.isnull(row['finalscandate'])),axis=1)
        print(f'Died before any repeat scan: {sum(combined["diedbeforerepeatscan"])}')
        combined = combined[combined['diedbeforerepeatscan']==False]
    else:
        print('Processing dataset including patients who died...')
        outcome_string = 'Unresolved LVT/Death'
    
    print(f'No anticoagulation: {sum(combined["Anticoagulation After LV Thrombus Diagnosis"] == "No Anticoagulation")}')
    combined = combined[combined['Anticoagulation After LV Thrombus Diagnosis'] != 'No Anticoagulation']
    #combined['Peak Troponin I, ng/dL'] = combined['Peak Troponin I, ng/dL'].replace({999.:np.nan})
    combined['lvtrecurrence'][~combined['lvtrecurrence'].isin(['0.0','1.0','2.0'])] = np.nan
    combined['lvtstatus'] = combined['lvtrecurrence'].replace({'0.0':'Resolved LVT','1.0':outcome_string,'2.0':outcome_string})
    #combined['lvtstatus'] = combined['lvtstatus'].replace({'0.0':'Resolved LVT','1.0':'Unresolved LVT','2.0':'Unresolved LVT','3.0':'Resolved LVT'})
    print(f'Unknown outcome: {sum(combined["lvtstatus"].isna())}')
    combined = combined[combined['lvtstatus'].isna()==False]
    combined = combined.drop('lvtrecurrence',axis=1)
    combined = combined.drop(['repeat_scan_date','finalscandate','dateofdeath','Anticoagulation After LV Thrombus Diagnosis'],axis=1)
    try:
        combined = combined.drop(['ddeath','diedbeforerepeatscan'],axis=1)
    except:
        pass
    var_list = [n for n in var_list if n not in ['lvtrecurrence','dateofdeath','repeat_scan_date','finalscandate','diedbeforerepeatscan','Anticoagulation After LV Thrombus Diagnosis']]
    cat_features_list = [cat for cat in cat_features_list if cat != 'Anticoagulation After LV Thrombus Diagnosis']
    combined = combined.reset_index(drop=True)
    print(f'Final cohort size:{len(combined)}')
    print()
    return combined,var_list,cat_features_list, cat_order


    