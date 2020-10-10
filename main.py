from fastapi import FastAPI 
from pydantic import BaseModel
import dbConnection
import randomForestModel
import nearestNeighbors
import uvicorn

#in-memory database
db = []

class Children(BaseModel):
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

@app.post('/predict2')
def get_prediction2(child: Children):
    result = nearestNeighbors.make_prediction(child.s_Cas_OffCode, child.s_Cas_Year, child.s_Cas_ChildNo, child.s_Cas_SeqNo)
    return result

#set URL for backend interface
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
