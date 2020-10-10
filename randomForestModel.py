import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dateutil.relativedelta import relativedelta
import joblib
import dbConnection

#set features
#featuresを選り抜く
selected_best_features = [
'd_BirthDate',
's_Par1RelationCode00',
's_Par1RelationCode01',
's_Par1RelationCode02',
's_Par1RelationCode03',
's_FaRelation00',
's_FaRelation01',
'd_FaBirthDate',
's_FaHousehold00',
's_FaHousehold01',
's_FaHousehold02',
's_MoRelation00',
's_MoRelation02',
'd_MoBirthDate',
's_MoHousehold00',
's_MoHousehold01',
's_MoHousehold02',
's_Oth1Relation00',
's_Oth1Relation03',
's_Oth1Relation07',
's_Oth1Relation08',
's_Oth1Relation11',
's_Oth1Relation12',
's_Oth1Relation90',
'd_Oth1BirthDate',
's_Oth1Household00',
's_Oth1Household01',
's_Oth1Household02',
's_Oth2Relation00',
's_Oth2Relation07',
's_Oth2Relation11',
's_Oth2Relation14',
's_Oth2Relation18',
's_Oth2Relation90',
'd_Oth2BirthDate',
's_Oth2Household00',
's_Oth2Household01',
's_Oth2Household02',
's_Oth3Relation00',
's_Oth3Relation14',
's_Oth3Relation15',
's_Oth3Relation18',
's_Oth3Relation25',
's_Oth3Relation90',
'd_Oth3BirthDate',
's_Oth3Household00',
's_Oth3Household01',
's_Oth3Household02',
's_Oth4Relation00',
's_Oth4Relation14',
's_Oth4Relation25',
'd_Oth4BirthDate',
's_Oth4Household00',
's_Oth4Household01',
's_Oth4Household02',
's_Oth5Relation00',
'd_Oth5BirthDate',
's_Oth5Household00',
's_Oth5Household01',
's_Oth5Household02',
's_Oth6Relation00',
'd_Oth6BirthDate',
's_Oth6Household02',
's_Oth7Relation00',
'd_Oth7BirthDate',
'd_Oth8BirthDate',
's_RootCode02',
's_RootCode03',
's_RootCode07',
's_RootCode12',
's_RootCode15',
's_RootCode17',
's_RootCode21',
's_RootCode22',
's_RootCode23',
's_RootCode24',
's_RootCode25',
's_RootCode26',
's_RootCode27',
's_RootCode99',
'i_MiFg',
's_FamilyCode00',
's_FamilyCode01',
's_FamilyCode02',
's_FamilyCode03',
's_FamilyCode05',
's_FamilyCode99',
'i_ChildSex1',
'i_ChildSex2',
's_SupCode00',
's_SupCode01',
's_SupCode02',
's_SupCode04',
's_SupCode08',
's_SupCode11',
's_SupCode14',
'i_ChiAbuNotiFg',
'i_CardFg',
's_AbuDtlCode01',
's_AbuDtlCode02',
's_AbuDtlCode03',
's_AbuDtlCode04',
's_AbuCode01',
's_AbuCode02',
's_AbuCode03',
's_AbuCode04',
's_AbuCode99',
'i_ConAbuFg',
'i_DvFg',
's_MentalDtlCode00',
's_MentalDtlCode01',
's_MentalDtlCode02',
's_MentalDtlCode03',
's_MentalDtlCode04',
's_SafetyChkCode00',
's_SafetyChkCode01',
's_SafetyChkCode07',
's_SafetyChkCode12',
's_SafetyChkCode13',
's_SafetyChkCode15',
's_SafetyChkCode16',
's_SafetyChkCode17',
's_AbuExistCode00',
's_AbuExistCode01',
's_AbuExistCode02',
's_AbuExistCode03',
'i_Parent1FgDiv0',
'i_Parent1FgDiv1',
'i_Parent2FgDiv0',
'i_Parent2FgDiv1',
'i_Parent6FgDiv0',
'i_Parent6FgDiv1',
'i_Parent7FgDiv0',
'i_Parent7FgDiv1',
'i_Parent10FgDiv0',
'i_Parent10FgDiv1',
'i_Person1FgDiv0',
'i_Person1FgDiv1',
'i_Person2FgDiv0',
'i_Person2FgDiv1',
'i_Person4FgDiv0',
'i_Person4FgDiv1',
'i_Person5FgDiv0',
'i_Person5FgDiv1',
'i_Family1FgDiv0',
'i_Family1FgDiv1']

#pass data to preprocessor
#データの捕録
def get_children_data(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
    df = dbConnection.get_children_by_id(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo)
    return df

#cleaning data
#データを綺麗する
def preprocesing_data(df):
    #convert data type to datetime
    df['d_BirthDate'] = pd.to_datetime(df['d_BirthDate'], format='%Y-%m-%d')
    df['d_ConsultDay'] = pd.to_datetime(df['d_ConsultDay'], format='%Y-%m-%d')

    dateColumnsWithNaNValue = ['d_FaBirthDate', 'd_MoBirthDate', 'd_Oth1BirthDate', 'd_Oth2BirthDate', 'd_Oth3BirthDate', 'd_Oth4BirthDate', 'd_Oth5BirthDate', 'd_Oth6BirthDate', 'd_Oth7BirthDate', 'd_Oth8BirthDate', 'd_Oth9BirthDate', 'd_Oth10BirthDate', 'd_Oth11BirthDate', 'd_Oth12BirthDate', 'd_Oth13BirthDate', 'd_Oth14BirthDate', 'd_Oth15BirthDate', 'd_Oth16BirthDate', 'd_Oth17BirthDate', 'd_Oth18BirthDate']
    # fill nan value with current date (transform it to zero)
    for col_name in dateColumnsWithNaNValue:
        df[col_name] = pd.to_datetime(df[col_name].fillna(df['d_ConsultDay']), format='%Y-%m-%d')

def calculate_age(df):
    # transform date features to time difference between consultDay
    dateColumns = ['d_BirthDate', 'd_FaBirthDate', 
                   'd_MoBirthDate', 'd_Oth1BirthDate', 
                   'd_Oth2BirthDate', 'd_Oth3BirthDate', 
                   'd_Oth4BirthDate', 'd_Oth5BirthDate', 
                   'd_Oth6BirthDate', 'd_Oth7BirthDate', 
                   'd_Oth8BirthDate', 'd_Oth9BirthDate', 
                   'd_Oth10BirthDate', 'd_Oth11BirthDate', 
                   'd_Oth12BirthDate', 'd_Oth13BirthDate', 
                   'd_Oth14BirthDate', 'd_Oth15BirthDate', 
                   'd_Oth16BirthDate', 'd_Oth17BirthDate', 
                   'd_Oth18BirthDate']

    for col_name in dateColumns:
        df[col_name] = df.apply(lambda x: relativedelta(x['d_ConsultDay'], x[col_name]).years, axis=1)


def delete_unused_columns(df):
    # delete unused columns
    df = df.drop(['s_Cas_OffCode', 's_Cas_Year', 's_Cas_ChildNo', 's_Cas_SeqNo', 's_Kind1Code', 's_ForeignTxt', 's_OtherText', 's_CityCode', 'd_ConsultDay', 'd_CloseDay', 'Temporary_Protection'], axis=1)
    return df

#make prediction using randomforest
#予測を作る
def make_prediction(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
    df = get_children_data(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo)
    df_question = df
    preprocesing_data(df_question)
    calculate_age(df_question)
    df_question = delete_unused_columns(df_question)
    
    # set dataframe using new features
    X_question = df_question[selected_best_features]

    # load the pretrained model
    rfs = joblib.load('C:\\Users\\JISOAI\\randomForestModel.pkl')

    # evaluate the ML accuracy with RandomForestClassifier algorithm
    RFS_prediction = rfs.predict(X_question)

    # convert numpy array to pandas series
    RFS_prediction_df = pd.DataFrame(RFS_prediction, columns = ['Prediction'])

    RFS_probability = rfs.predict_proba(X_question)

    RFS_probability_df = pd.DataFrame(RFS_probability[:, 1], columns = ['Probability'])
    RFS_probability_df['Probability'] = round(RFS_probability_df['Probability'], 3) * 100

    # join question data with the prediction value
    result = pd.concat([df.loc[:, df.columns != 'Temporary_Protection'], RFS_prediction_df], axis=1)
    result = pd.concat([result, RFS_probability_df], axis=1)
    
    # xtest1 = result.iloc[0].s_Cas_OffCode
    # xtest2 = result.iloc[0].s_Cas_Year
    # xtest3 = result.iloc[0].s_Cas_ChildNo
    # xtest4 = result.iloc[0].s_Cas_SeqNo
    # xtest5 = str(result.iloc[0].Prediction)
    # xtest6 = str(result.iloc[0].Probability)

    prediction_result = [{
        's_Cas_OffCode': result.iloc[0].s_Cas_OffCode,
        's_Cas_Year': result.iloc[0].s_Cas_Year,
        's_Cas_ChildNo': result.iloc[0].s_Cas_ChildNo,
        's_Cas_SeqNo':  result.iloc[0].s_Cas_SeqNo, 
        'Prediction':  str(result.iloc[0].Prediction), 
        'Probability': str(result.iloc[0].Probability)
    }]
    return prediction_result