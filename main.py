from fastapi import FastAPI 
from pydantic import BaseModel
from typing import List
import db_connect
import randomForestModel
import nearestNeighborModel
import uvicorn
# return both random forest and nearest neighbor results
import controller

#in-memory database
db = []

class Children(BaseModel):
    id:int
    s_Cas_OffCode: str
    s_Cas_Year: str
    s_Cas_ChildNo: str
    s_Cas_SeqNo: str

app = FastAPI()

#decorator
@app.get('/')
def index():
    return {'key' : 'value'}

# @app.get('/mixdata')
# def get_mixdata():
#     result = dbConnection.get_top10data()
#     return result

#randomforest
@app.post('/predict')
def get_prediction(child: Children):
    result = randomForestModel.make_prediction(child.s_Cas_OffCode, child.s_Cas_Year, child.s_Cas_ChildNo, child.s_Cas_SeqNo)
    return result

@app.post('/similardata')
def similar_data(child: Children):
    result = nearestNeighborModel.make_prediction(child.s_Cas_OffCode, child.s_Cas_Year, child.s_Cas_ChildNo, child.s_Cas_SeqNo)
    return result

# ai model with t_mixdata
@app.post('/t_mixdata/airesult')
def get_ai_result_t_mixdata(children: List[Children]):
    result = []

    for child in children: 
        data = controller.make_prediction_t_mixdata(child.id, child.s_Cas_OffCode, child.s_Cas_Year, child.s_Cas_ChildNo, child.s_Cas_SeqNo)
        result.append(data)

    return result

# ai model with t_mixdata_ts
@app.post('/t_mixdata_ts/airesult')
def get_ai_result_t_mixdata_ts(children: List[Children]):
    result = []

    for child in children: 
        data = controller.make_prediction_t_mixdata_ts(child.id, child.s_Cas_OffCode, child.s_Cas_Year, child.s_Cas_ChildNo, child.s_Cas_SeqNo)
        result.append(data)

    return result

#set URL for backend interface
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
