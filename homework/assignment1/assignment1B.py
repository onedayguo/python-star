
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import cvxopt as opt
import cvxpy as cvx
import random

data_path = "E:/Python/Spyder/assinment1/PA-1-data-matlab/count_data.mat"
data = scio.loadmat(data_path)
trainx = data['trainx']
trainy = data['trainy']
lamda = 0.5
sigma_o = 5


def ArrayToList(array):
    list = []
    for i in range(len(array)):
        list.append(float(array[i]))
    return list


def ls(xx,y):
    ls=np.linalg.inv(xx @ xx.T) @ xx @ y
    return ls


def rls(xx,y,numda):
    rls=np.linalg.inv(xx @ xx.T +numda) @ xx @ y
    return rls


def lasso(xx, y, k):
    p_temp = np.vstack((np.hstack([xx @ np.transpose(xx), -1 * xx @ np.transpose(xx)]), np.hstack([-1 * xx @ np.transpose(xx), xx @ np.transpose(xx)])))
    f_temp = 0.5 * np.ones((2 * k + 2, 1)) - np.vstack([xx @ y, -1 * xx @ y])
    G_temp = -1 * np.eye(2 * k + 2)
    h_temp = np.zeros((2 * k + 2, 1))

    P = opt.matrix(p_temp)
    f = opt.matrix(f_temp)
    G = opt.matrix(G_temp)
    h = opt.matrix(h_temp)
    x_par = np.array(opt.solvers.qp(P, f, G, h)['x'])
    thetaPlus = x_par[: k + 1, :]
    thetaMinus = x_par[k + 1:, :]
    return thetaPlus - thetaMinus

def rr(xx,y,k,n):
    c = np.vstack((np.zeros((k + 1, 1)), np.ones((n, 1))))
    b = np.vstack((-1 * y, y))
    A = np.vstack((np.hstack((-1 * np.transpose(xx), -1 * np.eye(n))), np.hstack((np.transpose(xx), -1 * np.eye(n)))))
    B = ArrayToList(b)
    
    x = cvx.Variable(n + k + 1)
    obj = cvx.Minimize(c.T * x)
    constraints = [A * x <= B]
    prob = cvx.Problem(obj, constraints)
    res = prob.solve()
    theta = x.value[: k + 1]
    return np.reshape(theta,(-1,1))

#bayesian regression
def BR(xx,y,alpha):
    sigma_par = np.linalg.inv(1/alpha+1/sigma_o*xx @ xx.T)
    miu_par = 1/sigma_o*sigma_par @ xx @ y
    return miu_par

def MSE(a, b):
    return np.square(a - b).mean()

def MAE(a,b):
    return np.fabs(a-b).mean()

def predict_y(x,theta):
    predict_y=x.T @ theta
    return predict_y


#print(ls(trainx,trainy))
#print(rls(trainx,trainy))
#print(lasso(trainx,trainy,8))
#print(rr(trainx,trainy,400))
#print(BR(trainx,trainy,5))
def Plotdata(title, x, predict_y, y):
    plt.figure()
    plt.title(title)
#    plt.scatter(sample_x, sample_y,color='black',label='samples',linewidth=0.1) # sample data
    plt.plot(x.T,predict_y,'r',label='predict',linewidth=2) # test data
    plt.plot(x.T,y,'b',label='true',linewidth=2)
    #plt.scatter(x,y,color='blue',label='ploy',linewidth=0.1)
    #plt.errorbar(x.T,predict_y(polyx,BR(newx, sampy,5)),5,fmt='.k')
    plt.legend()
    

def problemA():
    testx=data['testx']
    testy=data['testy']
    print('least-squares MSE',MSE(predict_y(testx,ls(trainx,trainy)),testy))
    print('regularized LS MSE',MSE(predict_y(testx,rls(trainx,trainy, lamda)),testy))
    print('L1-regularized LS MSE',MSE(predict_y(testx,lasso(trainx,trainy,8)),testy))
    print('robust regression MSE',MSE(predict_y(testx,rr(trainx,trainy,8,400)),testy))
    print('bayesian regression MSE',MSE(predict_y(testx,BR(trainx,trainy,5)),testy)) 
    print('least-squares MAE',MAE(predict_y(testx,ls(trainx,trainy)),testy))
    print('regularized LS MAE',MAE(predict_y(testx,rls(trainx,trainy,lamda)),testy))
    print('L1-regularized LS MAE',MAE(predict_y(testx,lasso(trainx,trainy,8)),testy))
    print('robust regression MAE',MAE(predict_y(testx, rr(trainx,trainy,8,400)),testy))
    print('bayesian regression MAE',MAE(predict_y(testx, BR(trainx,trainy,5)),testy))


problemA()


def process_x(datatype, co):  # co represents order
    xx = np.zeros(datatype.shape)  # 9*400
    for i in range(datatype.shape[1]):  # 400
        for j in range(datatype.shape[0]):  # 9
            xx[j, i] = np.power(datatype[j, i], co)
    # syn_xx=np.vstack((datatype, xx))
    return xx


def generatecrossterm(data):
    xx = np.zeros(data.shape)
    for i in range(data.shape[1]):
        for j in range(9):
            if j < 8:
                xx[j, i] = data[j, i] * data[j+1, i]
            else:
                xx[j, i] = data[j, i] * data[0, i]
    return xx


def problemB():
    # 2nd order polynomial
    sample_x = np.vstack((data['trainx'],process_x(data['trainx'],2)))
    sample_y = data['trainy']
    test_x = np.vstack((data['testx'],process_x(data['testx'],2)))
    test_y = data['testy']

    print('2nd order least-squares MSE',MSE(predict_y(test_x,ls(sample_x,sample_y)),test_y))
    print('2nd order regularized LS MSE',MSE(predict_y(test_x,rls(sample_x,sample_y,0.5)),test_y))
    print('2nd order L1-regularized LS MSE',MSE(predict_y(test_x,lasso(sample_x,sample_y,17)),test_y))
    print('2nd order robust regression MSE',MSE(predict_y(test_x,rr(sample_x,sample_y,17,400)),test_y))
    print('2nd bayesian regression MSE',MSE(predict_y(test_x,BR(sample_x,sample_y,5)),test_y))
    print('-------------------------------------------------')
    print('2nd order least-squares MAE',MAE(predict_y(test_x,ls(sample_x,sample_y)),test_y))
    print('2nd order regularized LS MAE',MAE(predict_y(test_x,rls(sample_x,sample_y,0.5)),test_y))
    print('2nd order L1-regularized LS MAE',MAE(predict_y(test_x,lasso(sample_x,sample_y,17)),test_y))
    print('2nd order robust regression MAE',MAE(predict_y(test_x,rr(sample_x,sample_y,17,400)),test_y))
    print('2nd bayesian regression MAE',MAE(predict_y(test_x,BR(sample_x,sample_y,5)),test_y))
    # 3rd order polynomial
    sample_x3=np.vstack((sample_x,process_x(data['trainx'],3)))
    test_x3=np.vstack((test_x,process_x(data['testx'],3)))
    print('-------------------------------------------------')
    print('3nd order least-squares MSE',MSE(predict_y(test_x3,ls(sample_x3,sample_y)),test_y))
    print('3nd order regularized LS MSE',MSE(predict_y(test_x3,rls(sample_x3,sample_y,0.5)),test_y))
    print('3nd order L1-regularized LS MSE',MSE(predict_y(test_x3,lasso(sample_x3,sample_y,26)),test_y))
    print('3nd order robust regression MSE',MSE(predict_y(test_x3,rr(sample_x3,sample_y,26,400)),test_y))
    print('3nd bayesian regression MSE',MSE(predict_y(test_x3,BR(sample_x3,sample_y,5)),test_y))
    
    print('3nd order least-squares MAE',MAE(predict_y(test_x3,ls(sample_x3,sample_y)),test_y))
    print('3nd order regularized LS MAE',MAE(predict_y(test_x3,rls(sample_x3,sample_y,0.5)),test_y))
    print('3nd order L1-regularized LS MAE',MAE(predict_y(test_x3,lasso(sample_x3,sample_y,26)),test_y))
    print('3nd order robust regression MAE',MAE(predict_y(test_x3,rr(sample_x3,sample_y,26,400)),test_y))
    print('3nd bayesian regression MAE',MAE(predict_y(test_x3,BR(sample_x3,sample_y,5)),test_y))

def problemBcross():
    sample_x=np.vstack((data['trainx'],generatecrossterm(data['trainx'])))
    sample_y=data['trainy']
    test_x=np.vstack((data['testx'],generatecrossterm(data['testx'])))
    test_y=data['testy']
    print('cross order least-squares MSE',MSE(predict_y(test_x,ls(sample_x,sample_y)),test_y))
    print('crossorder regularized LS MSE',MSE(predict_y(test_x,rls(sample_x,sample_y,0.5)),test_y))
    print('cross order L1-regularized LS MSE',MSE(predict_y(test_x,lasso(sample_x,sample_y,17)),test_y))
    print('cross order robust regression MSE',MSE(predict_y(test_x,rr(sample_x,sample_y,17,400)),test_y))
    print('cross bayesian regression MSE',MSE(predict_y(test_x,BR(sample_x,sample_y,5)),test_y))
    print('-------------------------------------------------')
    print('cross order least-squares MAE',MAE(predict_y(test_x,ls(sample_x,sample_y)),test_y))
    print('cross order regularized LS MAE',MAE(predict_y(test_x,rls(sample_x,sample_y,0.5)),test_y))
    print('cross order L1-regularized LS MAE',MAE(predict_y(test_x,lasso(sample_x,sample_y,17)),test_y))
    print('cross order robust regression MAE',MAE(predict_y(test_x,rr(sample_x,sample_y,17,400)),test_y))
    print('cross bayesian regression MAE',MAE(predict_y(test_x,BR(sample_x,sample_y,5)),test_y))

# problemBcross()


# problemA()
def problemAPlot():
    testx = data['testx']
    testy = data['testy']
    lspredicty = np.round(predict_y(testx,ls(trainx, trainy)))
    rlspredicty = np.round(predict_y(testx,rls(trainx, trainy,0.5)))
    lassopredicty = np.round(predict_y(testx,lasso(trainx, trainy,8)))
    rrpredicty=np.round(predict_y(testx,rr(trainx, trainy,8,400)))
    brpredicty = np.round(predict_y(testx,BR(trainx, trainy,5)))
    x=np.linspace(1,600,600)
    np.reshape(x,(1,-1))
    
    Plotdata('Ls', x,lspredicty,testy)
    Plotdata('Rls',  x, rlspredicty,testy)
    Plotdata('Lasso', x, lassopredicty,testy)
    Plotdata('RR',  x, rrpredicty,testy)
    Plotdata('BR', x, brpredicty,testy)


def crosstermplot():
    sample_x = np.vstack((data['trainx'],generatecrossterm(data['trainx'])))
    sample_y=data['trainy']
    test_x=np.vstack((data['testx'],generatecrossterm(data['testx'])))
    test_y=data['testy']
    lspredicty=np.round(predict_y(test_x,ls(sample_x, sample_y))) 
    rlspredicty=np.round(predict_y(test_x,rls(sample_x, sample_y,0.5)))
    lassopredicty=np.round(predict_y(test_x,lasso(sample_x, sample_y,17))) 
    rrpredicty=np.round(predict_y(test_x,rr(sample_x, sample_y, 17, 400)))
    brpredicty=np.round(predict_y(test_x,BR(sample_x, sample_y, 5)))
    x=np.linspace(1,600, 600)
    np.reshape(x,(1,-1))
    
    Plotdata('Ls', x,lspredicty, test_y)
    Plotdata('Rls',  x, rlspredicty, test_y)
    Plotdata('Lasso', x, lassopredicty, test_y)
    Plotdata('RR',  x, rrpredicty, test_y)
    Plotdata('BR', x, brpredicty, test_y)

def twondplot():
    sample_x=np.vstack((data['trainx'],process_x(data['trainx'],3)))
    sample_y=data['trainy']
    test_x=np.vstack((data['testx'],process_x(data['testx'],3)))
    test_y=data['testy']
    lspredicty=np.round(predict_y(test_x,ls(sample_x, sample_y))) 
    rlspredicty=np.round(predict_y(test_x,rls(sample_x, sample_y,0.5)))
    lassopredicty=np.round(predict_y(test_x,lasso(sample_x, sample_y,17))) 
    rrpredicty=np.round(predict_y(test_x,rr(sample_x, sample_y,17,400)))
    brpredicty=np.round(predict_y(test_x,BR(sample_x, sample_y,5)))
    x=np.linspace(1, 600, 600)
    np.reshape(x,(1,-1))
    
    Plotdata('Ls', x,lspredicty,test_y)
    Plotdata('Rls',  x, rlspredicty,test_y)
    Plotdata('Lasso', x, lassopredicty,test_y)
    Plotdata('RR',  x, rrpredicty,test_y)
    Plotdata('BR', x, brpredicty,test_y)


# twondplot()

#problemB()
    



