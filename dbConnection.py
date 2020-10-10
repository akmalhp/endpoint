import pyodbc
import pandas as pd
import numpy as np  

#connect to database 
#データベースへと繋ぐ
sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER=jisonext1;DATABASE=JisoNextDB;UID=jisouser;PWD=jisonext1db$$;Trusted_Connection=yes') 
cursor = sql_conn.cursor()

# def get_top10data():
#     query = """SELECT TOP 10 * FROM T_MIXDATA"""
#     df = pd.read_sql(query, sql_conn)
#     return df.to_json()

#obtain data by ID
#データの捕録
def get_children_by_id(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
    sql = """EXEC T_MIXDATA_BACKUP_GetDataById ?, ?, ?, ?;"""
    values = (s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo)
    cursor.execute(sql, (values))
    rows = cursor.fetchall()
    #to convert data from database to dataframe
    df = pd.DataFrame((tuple(x) for x in rows))
    #to set column header
    df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    return df

# def get_children_by_id2(s_Cas_OffCode, s_Cas_Year, s_Cas_ChildNo, s_Cas_SeqNo):
#     sql = "EXEC T_MIXDATA_BACKUP_GetDataById '"+s_Cas_OffCode+"','"+s_Cas_Year+"','"+s_Cas_ChildNo+"','"+s_Cas_SeqNo+"';"
#     df = pd.read_sql(sql, sql_conn)
#     return df.to_json()