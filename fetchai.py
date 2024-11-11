from uagents import Agent, Bureau, Context, Model
from typing import Any, Dict 
import time
import joblib
import pandas as pd


class Message(Model):
    text: str

al="agent1qww3ju3h6kfcuqf54gkghvt2pqe8qp97a7nzm2vp8plfxflc0epzcjsv79t"
bo="agent1q0mau8vkmg78xx0sh8cyl4tpl4ktx94pqp2e94cylu6haugt2hd7j9vequ7"

alice = Agent(name="alice", seed="alice recovery phrase")
bob = Agent(name="bob", seed="bob recovery phrase")


@alice.on_event("startup")
async def send_message(ctx: Context):
    
    dict={}
    user_input = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    for i in user_input:
        dict[i]=float(input(f"enter value for{i}:"))
        
    data=pd.DataFrame([dict])
    model=joblib.load("C:/Users/MY PC/Downloads/ML-hackathon-example/ML-hackathon-example/heart_new.pkl")
    prediction=model.predict(data)
    
    if(prediction[0]==1):
        msg = "heart attack patient"
    else:
        msg="No heart attack"
        
    await ctx.send(bob.address, Message(text=msg))
    
    
@bob.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"{msg.text}")
    


bureau = Bureau()
bureau.add(alice)
bureau.add(bob)


def main():
    bureau.run()

if __name__ == "__main__":
    data_dict={}
    main()


