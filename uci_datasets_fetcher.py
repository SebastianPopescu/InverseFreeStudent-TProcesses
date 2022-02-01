import numpy as np
from collections import defaultdict
import pandas
from sklearn.cluster import  KMeans
from sklearn import preprocessing
import os
import gzip
import numpy as np
from keras.datasets import cifar10
from scipy.io import arff

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
        '%s-labels-idx1-ubyte.gz'
        % kind)
    images_path = os.path.join(path,
        '%s-images-idx3-ubyte.gz'
        % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
        offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 784)

    return images, labels

def one_hot_encoder(input):

    unique_classes = len(np.unique(input))
    object = np.zeros(shape=(input.shape[0], unique_classes))
    for i in range(input.shape[0]):
        object[i,int(input[i])] = 1.0
    return object

def get_kmeans(X, num_inducing):

    km = KMeans(n_clusters=num_inducing).fit(X)
    k_mean_output = km.cluster_centers_
    print('we have passed the extraction of k-means')
    k_mean_output = k_mean_output.astype('float32')

    return k_mean_output

def normalize(X):

    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    
    return (X - X_mean) / X_std

def preprocess_data(X):
    
    #### to be used for regression tasks ####
    X = normalize(X)

    return X

def split_into_sets(seed, prop, X,Y):

    ind = np.arange(X.shape[0])

    np.random.seed(seed)
    np.random.shuffle(ind)

    n = int(X.shape[0] * prop)

    X_train = X[ind[:n]]
    Y_train = Y[ind[:n]]

    X_test = X[ind[n:]]
    Y_test = Y[ind[n:]]

    return X_train, Y_train, X_test, Y_test

def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def fetch_dataset(name):

    ###############################
    ###### Classification part ####
    ###############################

    if name =='MNIST':

        data_train = np.loadtxt('/vol/biomedic2/sgp15/data/uci_datasets/mnist_train.csv',delimiter=',')
        data_test = np.loadtxt('/vol/biomedic2/sgp15/data/uci_datasets/mnist_test.csv',delimiter=',')
        Y_training = data_train[:,0]
        Y_testing = data_test[:,0]
        X_training = np.delete(data_train,[0],axis=1)
        X_testing = np.delete(data_test,[0],axis=1)
        X_training = X_training.reshape(60000, 784)
        X_testing = X_testing.reshape(10000, 784)
        X_training = X_training.astype('float32')
        X_testing = X_testing.astype('float32')
        X_training /= 255.0
        X_testing /= 255.0
        Y_training = Y_training.reshape((-1,1))
        Y_testing = Y_testing.reshape((-1,1))
        Y_training = one_hot_encoder(Y_training)
        Y_testing = one_hot_encoder(Y_testing)

        return X_training, Y_training, X_testing, Y_testing

    elif name =='Fashion-MNIST':

        X_training, Y_training = load_mnist('/vol/biomedic2/sgp15/data/uci_datasets/fashion', kind='train')
        X_testing, Y_testing = load_mnist('/vol/biomedic2/sgp15/data/uci_datasets/fashion', kind='t10k')

        X_training = X_training.astype('float32')
        X_testing = X_testing.astype('float32')
        X_training /= 255.0
        X_testing /= 255.0
        Y_training = Y_training.reshape((-1,1))
        Y_testing = Y_testing.reshape((-1,1))
        Y_training = one_hot_encoder(Y_training)
        Y_testing = one_hot_encoder(Y_testing)

        return X_training, Y_training, X_testing, Y_testing

    elif name =='CIFAR10-Grayscale':

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        #X_train = X_train.reshape(50000, 32*32)
        #X_test = X_test.reshape(10000, 32*32)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255.0
        X_test /= 255.0
        print(y_train.shape)
        print(y_test.shape)
        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))

        print('finished processing CIFAR10 dataset')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        y_train = y_train.reshape((50000,1))
        y_test = y_test.reshape((10000,1))
        X_train = X_train.reshape(50000,32,32,3)
        X_test = X_test.reshape(10000,32,32,3)

        X_train = grayscale(X_train)
        X_test = grayscale(X_test)

        X_train = X_train.reshape((50000, 32*32))
        X_test = X_test.reshape((10000, 32*32))
        y_train = one_hot_encoder(y_train)
        y_test = one_hot_encoder(y_test)

        return X_train, y_train, X_test, y_test

    elif name =='WIFI':
         
        data = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/wifi_localization.txt', delimiter='\t')
        x_data = data[:,:-1]
        y_data = data[:,-1].reshape((-1,1))

        X_data = np.asarray(x_data, dtype=np.float32)
        X_data = preprocess_data(X_data)
        Y_data = np.asarray(y_data, dtype=np.float32)        
        Y_data = Y_data.reshape((-1, 1)) - 1.0
        Y_data = one_hot_encoder(Y_data)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing



    elif name == 'Diabetic':

        data = arff.loadarff('/vol/biomedic2/sgp15/data/uci_datasets/diabetic.arff')
        df = pandas.DataFrame(data[0])

        y_data = [float(plm) for plm in df['Class'].values.ravel()]
        x_data = df.loc[:, '0':'18'].values

        X_data = np.asarray(x_data, dtype=np.float32)
        X_data = preprocess_data(X_data)
        Y_data = np.asarray(y_data, dtype=np.float32)        
        Y_data = Y_data.reshape((-1, 1))

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing

    elif name =='Yeast':

        df = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/yeast.csv', header=None)
        df.dropna()
        data = df.values
        print(data[:10,:-1])

        X_data = np.asarray(data[:, :-1], dtype=np.float32)
        print(X_data[:10,:])

        X_data = preprocess_data(X_data)
        Y_data = data[:,-1].reshape((-1, 1))
   
        le = preprocessing.LabelEncoder()
        le.fit(Y_data.ravel())
        print(le.classes_)
        
        Y_data = le.transform(Y_data.ravel())
        Y_data = Y_data.reshape((-1,1))
        Y_data = one_hot_encoder(Y_data)


        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'Biodegradation':

        df = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/biodegradation.csv', header=None)
        df.dropna()
        data = df.values

        X_data = preprocess_data(data[:, :-1])
        Y_data = data[:,-1].reshape((-1, 1))
   
        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'Magic':

        df = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/magic.csv', header=None)
        df.dropna()
        data = df.values


        X_data = np.asarray(data[:, :-1], dtype=np.float32)
        print(X_data[:10,:])


        X_data = preprocess_data(X_data)
        Y_data = data[:,-1].reshape((-1, 1))
   
        le = preprocessing.LabelEncoder()
        le.fit(Y_data.ravel())
        print(le.classes_)
        
        Y_data = le.transform(Y_data.ravel())
        Y_data = Y_data.reshape((-1,1))

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'Vertebral':

        df = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/vertebral.csv', header=None, delimiter=' ')
        df.dropna()
        data = df.values
        print(data[:10,:])


        X_data = np.asarray(data[:, :-1], dtype=np.float32)
        print(X_data[:10,:])

        X_data = preprocess_data(X_data)
        Y_data = data[:,-1].reshape((-1, 1))
     
        le = preprocessing.LabelEncoder()
        le.fit(Y_data.ravel())
        print(le.classes_)
        
        Y_data = le.transform(Y_data.ravel())
        Y_data = Y_data.reshape((-1,1))
        Y_data = one_hot_encoder(Y_data)
        
        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing

    elif name =='Letters':

        X_data = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/letter.csv', delimiter=',', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        Y_data = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/letter.csv', delimiter=',', usecols=[0], dtype=None)
        
        X_data = X_data / 255.0

        print(X_data.shape)
        print(Y_data.shape)

        X_train, Y_train, X_test, Y_test = split_into_sets(seed = 0, prop = 0.8, X = X_data, Y = Y_data)

        le = preprocessing.LabelEncoder()
        le.fit(Y_train.ravel())

        print(le.classes_)
        Y_train = le.transform(Y_train.ravel())
        Y_test = le.transform(Y_test.ravel())


        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        Y_train = one_hot_encoder(Y_train)
        Y_test = one_hot_encoder(Y_test)
        Y_train = Y_train.astype('float32')
        Y_test = Y_test.astype('float32')

        return X_train, Y_train, X_test, Y_test

    elif name == 'Avila':
    
        X_training = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/avila-tr.txt', delimiter=',',usecols=[0,1,2,3,4,5,6,7,8,9])
        X_testing = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/avila-ts.txt', delimiter=',',usecols=[0,1,2,3,4,5,6,7,8,9])

        Y_training = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/avila-tr.txt', delimiter=',',usecols=[10],dtype=None)
        Y_testing = np.genfromtxt('/vol/biomedic2/sgp15/data/uci_datasets/avila-ts.txt', delimiter=',',usecols=[10],dtype=None)

        X_training = X_training.astype('float32')
        X_testing = X_testing.astype('float32')
        le = preprocessing.LabelEncoder()
        le.fit(Y_training.ravel())

        print(le.classes_)
        Y_training = le.transform(Y_training.ravel())
        Y_testing = le.transform(Y_testing.ravel())
        Y_training = Y_training.reshape((-1,1))
        Y_testing = Y_testing.reshape((-1,1))

        X_training = X_training.astype('float32')
        X_testing = X_testing.astype('float32')
        Y_training = one_hot_encoder(Y_training)
        Y_testing = one_hot_encoder(Y_testing)
        Y_training = Y_training.astype('float32')
        Y_testing = Y_testing.astype('float32')

        return X_training, Y_training, X_testing, Y_testing


    elif name in ['adult', 'chess-krvk', 'connect-4', 'miniboone', 'mushroom', 'nursery', 'page-blocks', 'pendigits', 'ringnorm', 'statlog-landsat' , 'statlog-shuttle', 'thyroid', 'twonorm', 'wall-following', 'waveform', 'waveform-noise']:


        X_data = pandas.read_csv('/vol/biomedic3/sgp15/selu_datasets/'+str(name)+'/'+str(name)+'_py.dat', header=None, delimiter=',').values
        Y_data = pandas.read_csv('/vol/biomedic3/sgp15/selu_datasets/'+str(name)+'/labels_py.dat', header=None, delimiter=',').values
        
        X_data = X_data.astype('float32')
        Y_data = Y_data.astype('float32')

        print(X_data)
        print(Y_data)


        X_train, Y_train, X_test, Y_test = split_into_sets(seed = 0, prop = 0.8, X = X_data, Y = Y_data)
        le = preprocessing.LabelEncoder()
        le.fit(Y_train.ravel())

        print(le.classes_)

        if len(le.classes_)==2:

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            Y_train = Y_train.astype('float32')
            Y_test = Y_test.astype('float32')

        else:

            Y_train = le.transform(Y_train.ravel())
            Y_test = le.transform(Y_test.ravel())
            Y_train = Y_train.reshape((-1,1))
            Y_test = Y_test.reshape((-1,1))

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            Y_train = one_hot_encoder(Y_train)
            Y_test = one_hot_encoder(Y_test)
            Y_train = Y_train.astype('float32')
            Y_test = Y_test.astype('float32')

        return X_train, Y_train, X_test, Y_test


    ################################
    ####### Regression part ########
    ################################

    elif name == 'Boston':

        #N, D, name = 506, 13, 'boston'
        #url = uci_base_url + 'housing/housing.data'

        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/BostonHousing.csv', header=None).values


        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))

        return X_training, Y_training, X_testing, Y_testing


    elif name == 'Concrete':

        #N, D, name = 1030, 8, 'concrete'
        #url = uci_base_url + '/Concrete_Data.xls'

        data = pandas.read_excel('/vol/biomedic2/sgp15/data/uci_datasets/Concrete_Data.xls').values
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))

        return X_training, Y_training, X_testing, Y_testing


    elif name == 'Energy':

        #N, D, name = 768, 8, 'energy'
        #url = uci_base_url + '00242/ENB2012_data.xlsx'

        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = pandas.read_excel('/vol/biomedic2/sgp15/data/uci_datasets/ENB2012_data.xlsx').values[:, :-1]
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing


    elif name == 'Kin8mn':

        #N, D, name = 8192, 8, 'kin8nm'
        #url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'
        
        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/dataset_2175_kin8nm.csv', header=None).values
        #return data[:, :-1], data[:, -1].reshape(-1, 1)
        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing

    elif name == 'Wisconsin':
      
        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/breast_wisconsin.csv', header=None)
        #return data[:, :-1], data[:, -1].reshape(-1, 1)
        #data.dropna()
        #data[~data.str.contains("?")]
        #data.apply(lambda x: pandas.to_numeric(x, errors = '?')).dropna()
        data = data.values
        X_data = np.asarray(data[:, 1:-1], dtype=np.float32)
        X_data = preprocess_data(X_data)
        Y_data = np.asarray(data[:, -1], dtype=np.float32)       
        Y_data = Y_data/2.0 - 1.0

        print(X_data.shape)
        print(Y_data.shape)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data.reshape(-1, 1))

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'Naval':	        


        #N, D, name = 11934, 14, 'naval'
        #url = uci_base_url + '00316/UCI%20CBM%20Dataset.zip'

        data = pandas.read_fwf('/vol/biomedic2/sgp15/data/uci_datasets/naval_data.txt', header=None).values
        # NB this is the first output
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)

        # dims 8 and 11 have std=0:
        X = np.delete(X, [8, 11], axis=1)
        

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)


        return X_training, Y_training, X_testing, Y_testing

    elif name =='3droad':

        #### name, N, D =  'wilson_3droad', 434874, 3

        data = pandas.read_fwf('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/3droad/3D_spatial_network.txt', header=None).values
        # NB this is the first output
        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)
 

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'houseelectric':

        df = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/houseelectric/household_power_consumption.txt', header = None, delimiter=' ')
    
        df.dropna()
        data = df.values

        X = data[:, :9]
        Y = data[:, 9].reshape(-1, 1)

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    

    elif name == 'slice':

        data = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/slice/slice_localization_data.csv', header = None, delimiter=',').values
        
        X = data[:, 1:-1]
        Y = data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    


    elif name == 'elevators':

        df1 = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/elevators/elevators.data', header = None, delimiter=', ').values
        df2 = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/elevators/elevators.test', header = None, delimiter=', ').values

        data = np.concatenate((df1,df2),axis=0)
        X = data[:, 1:-1]
        Y = data[:, -1].reshape(-1, 1)
        Y = np.log(Y)
        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    

    elif name == 'bike':

        data = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/bike/hour.csv', header = None, delimiter=',').values

        X = data[:, 2:-1]
        Y = data[:, -1].reshape(-1, 1)
        Y = np.log(Y)
        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    

    elif name == 'keggdirected':

        data = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/keggdirected/reaction_directed.txt', header = None, delimiter=',').values

        data = np.delete(data, [9,14], axis=1)
        Y_data = data[:,1]
        X_data = np.delete(data,[1], axis=1)
    
        X_data = preprocess_data(X_data)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y_data)

        return X_training, Y_training, X_testing, Y_testing    

    elif name == 'keggundirected':

        data = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/keggundirected/reaction_undirected.txt', header = None, delimiter=',').values

        data = data[data[:,20]<1,:]

        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    
    
    elif name == 'buzz':

        data = pandas.read_csv('/vol/biomedic3/sgp15/uci_wilson_datasets/uci/buzz/Twitter.data', header = None, delimiter=',').values

        X = data[:, :-1]
        Y = data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(X)

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = Y)

        return X_training, Y_training, X_testing, Y_testing    

    elif name == 'Power':

        #N, D, name = 9568, 4, 'power'
        #url = uci_base_url + '00294/CCPP.zip'

        data = pandas.read_excel('/vol/biomedic2/sgp15/data/uci_datasets/Folds5x2_pp.xlsx').values
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing


    elif name == 'Protein':

        #N, D, name = 45730, 9, 'protein'
        #url = uci_base_url + '00265/CASP.csv'

        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/CASP.csv').values
        #return data[:, 1:], data[:, 0].reshape(-1, 1)

        X_data = preprocess_data(data[:, 1:])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, 0].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing

    elif name == 'WineRed':
                        
        #N, D, name = 1599, 11, 'winered'
        #url = uci_base_url + 'wine-quality/winequality-red.csv'

        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/winequality-red.csv', delimiter=';').values
        
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing

    elif name == 'WineWhite':	        

        #N, D, name = 4898, 11, 'winewhite'
        #url = uci_base_url + 'wine-quality/winequality-white.csv'

        data = pandas.read_csv('/vol/biomedic2/sgp15/data/uci_datasets/winequality-white.csv', delimiter=';').values
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))


        return X_training, Y_training, X_testing, Y_testing

    elif name =='Yacht':

        #N, D, name = 308, 6, 'yacht'
        #url = uci_base_url + '/00243/yacht_hydrodynamics.data'

        data = pandas.read_fwf('/vol/biomedic2/sgp15/data/uci_datasets/yacht_hydrodynamics.data', header=None).values[:-1, :]
        #return data[:, :-1], data[:, -1].reshape(-1, 1)

        X_data = preprocess_data(data[:, :-1])

        X_training, Y_training, X_testing, Y_testing = split_into_sets(seed=7, prop=0.8, X = X_data, Y = data[:, -1].reshape(-1, 1))

        return X_training, Y_training, X_testing, Y_testing

    elif name == 'YearPredictionMSD':

        data = np.loadtxt('/vol/biomedic2/sgp15/data/uci_datasets/YearPredictionMSD.txt', delimiter=',')


        print(data[:10,:])
        X_data = preprocess_data(data[:, 1:])
        Y_data = data[:, 0].reshape((-1,1))

        X_training = X_data[:463715, :]
        X_testing = X_data[463715:, :]
        Y_training = Y_data[:463715, :]
        Y_testing = Y_data[463715:, :]

        return X_training, Y_training, X_testing, Y_testing
