from MNIST import MNIST
from data_loader import load_data
from model_parser import parse
import datetime
'''
Steps : 
1. We load the configuration file
2. load the model
3. Predict
'''

def main():
    conf=parse('../configurations/config.conf')
    net = MNIST()
    _,_,x_test,y_test = load_data()
    net.restore_model(conf['PREDICTION']['trained_model'])
    x = x_test[:128]
    x = x/255.
    x = x.reshape(-1,28,28,1)
    n = 100
    #print("ground truth : ",y_test[0])
    #print("predicted label :",net.predict(x).numpy())
    t=0.   
    for i in range(n):
        t1 = datetime.datetime.now()
        net.predict(x)
        t += (datetime.datetime.now() - t1).total_seconds()
    print('Avg time : ', t/n)
    


if __name__== "__main__":
    main()