from models.t_mixdata import random_forest_model as rfm_t_mixdata
from models.t_mixdata import nearest_neighbor_model as knn_t_mixdata
from models.t_mixdata_ts import random_forest_model as rfm_t_mixdata_ts
from models.t_mixdata_ts import nearest_neighbor_model as knn_t_mixdata_ts
import pandas as pd

# t_mixdata version
def make_prediction_t_mixdata(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
    result = []
    # get random forest result
    result, df = rfm_t_mixdata.make_prediction(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo)
    # get nearest neighbor result
    result['similar_data'] = knn_t_mixdata.make_prediction(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo, df)
    return result

# t_mixdata_ts version
def make_prediction_t_mixdata_ts(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
    result = []
    # get random forest result
    result, df = rfm_t_mixdata_ts.make_prediction(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo)
    # get nearest neighbor result
    result['similar_data'] = knn_t_mixdata_ts.make_prediction(id, s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo, df)
    return result
