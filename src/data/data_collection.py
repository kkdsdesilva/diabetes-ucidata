# import libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo
import warnings

# ignore warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# function for import data
def import_data(kind=None):
    """Import data from various sources.
    kind: str
        The kind of data to import. Options are 'basic', 'tests', 'drugs', 'other', 'targets', or None.    

        Returns: 
            data: pandas dataframe
                The data to be used for analysis.
        """
    
    # fetch dataset 
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
    
    # data (as pandas dataframes) 
    data_1 = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
    data_2 = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 

    if kind==None:
        # merge dataframes
        data = pd.concat([data_1, data_2], axis=1)

        # return data
        return data
    
    elif kind=='basic':
        col = ['race', 'gender', 'age', 'weight', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'payer_code', 'medical_specialty', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
       'number_diagnoses']
        
        # return basic data
        return data_1[col]
    
    elif kind=='tests':
        col = ['max_glu_serum', 'A1Cresult']

        # return tests data
        return data_1[col]

    
    elif kind=='drugs':
        col = ['metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
        
        # return drugs data
        return data_1[col]
    
    elif kind=='other':
        col = ['change', 'diabetesMed']

        # return other data
        return data_1[col]
    
    elif kind=='targets':
        col = ['readmitted']

        # return targets data
        return data_2[col]
    
    else:
        raise ValueError('Invalid kind: %s' % kind)