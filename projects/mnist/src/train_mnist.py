from MNIST import MNIST
from data_loader import get_mnist_dataset
from model_parser import parse

'''
Steps : 
1. We load the configuration file
2. Create the model or load the model
3. Initialize the optimizers
4. Train the network on the data
5. Save the model
'''

def main():
    conf=parse('../configurations/config.conf')
    net = MNIST()
    dset_train_x,dset_train_y,dset_test_x,dset_test_y = get_mnist_dataset(batch_size=int(conf['MODEL']['batch_size']))
    if(conf['MODEL']['to_restore_model']=='true'):
        net.restore_model(conf['MODEL']['restore_path'])
    else:
        net.create_model()
    net.compile_model()
    net.train_model(dset_train_x,dset_train_y,dset_test_x,dset_test_y,int(conf['MODEL']['batch_size']),int(conf['MODEL']['epochs']))
    net.save_model(conf['MODEL']['save_path'])


if __name__== "__main__":
    main()