import math

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
import numpy as np
import matplotlib.pyplot as plt
import time

# This class represents the weights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    # dict {token:weight, token:weight.....}
    # {token: weight, token: weight}
    self.w_tokens = {}
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.

    # {token:last update time, token: last update time}
    self.access_time = {}
    # define an empty flat weight list initialized with 0
    # self.weight_list = [0] * (len(self.total_token_list)+4)
    
  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string
  
  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)
  
  # @return {Int} the l2 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552

  # def __init__(self):
  #   self.weights = Weights()
  #   self.token_len = len(self.weights.total_token_list)

  '''
    wt = wT*x + w0
  '''
  def sigmoid(self,wt):
    z = 1.0/(1.0+np.exp((-1)*wt))
    return z

    '''
    neg log likelihood for individual data point
    '''
  def cost(self,y,sigmoid):
    cost1= math.log(sigmoid) * y 
    cost2 = math.log(1-sigmoid) * (1-y)
    cost = -cost1-cost2
    return cost


  def loss(self,predict,y):
    return math.pow((predict-y),2)


    
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights,x,instance):
   
    # construct xi
    # fea_list = list()
    fea_list = x
    token_list = instance.tokens
    token_set = set(token_list)

    weight_vec = list()
    weight_vec.append(weights.w_age)
    weight_vec.append(weights.w_gender)
    weight_vec.append(weights.w_depth)
    weight_vec.append(weights.w_position)
    # weight_set = set(weights.w_tokens.keys())

    for token in token_set:
      weight_vec.append(weights.w_tokens[token])
       
    dot_prod = np.dot(fea_list,weight_vec)+weights.w0 
    return dot_prod,weight_vec

    '''
    for test data
    '''
  def compute_weight_feature_product_test(self, weights,x,instance):
   
    # construct xi
    t4 = time.time()
    fea_list = x
    token_list = instance.tokens
    token_set = set(token_list)
    t9 = time.time()
   

    weight_vec = list()
    weight_vec.append(weights.w_age)
    weight_vec.append(weights.w_gender)
    weight_vec.append(weights.w_depth)
    weight_vec.append(weights.w_position)
    t10 = time.time()
    # print("----------time for appending weights",t10-t9)
    # weight_set = set(weights.w_tokens.keys())
    # 
    t5 = time.time()
    
    count = 0
    for token in token_set:
      # starttime = time.time()
      # dealing with unseen tokens in test set
      # if token in weight_set:
      t7 = time.time()
      if weights.w_tokens.has_key(token):
        weight_vec.append(weights.w_tokens.get(token))
        count = count +1
      t8 = time.time()
    t9 = time.time()
    fea_list.extend([1]*count)

    dot_prod = np.dot(fea_list,weight_vec)+weights.w0 
    return dot_prod
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, token, weights, now, step, lambduh):
    if weights.access_time.has_key(token):
      last_time = weights.access_time.get(token)
      downweight = math.pow((1- step*lambduh),now-last_time)
    else:
      downweight = 0
      # downweight_list.append(downweight)
      # weights.w_tokens[token] = weights.w_tokens.get(token) * downweight
    return downweight

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    weights = Weights()
    # For each data point:
      # Your code: perform delayed regularization

      # Your code: predict the label, record the loss
  
      # Your code: compute w0 + <w, x>, and gradient
      
      # Your code: update weights along the negative gradient
    step = step
    total_loss = 0.0
    avg_loss_list =list()
    count = 0
    print "training: Loading data from " + dataset.path + "..."
    t130 = time.time()
    while dataset.hasNext():    
      instance = dataset.nextInstance()
      # y labels
      y = instance.clicked
      # x : features
      x = list()
      x.append(instance.age)
      x.append(instance.gender)
      x.append(instance.depth)
      x.append(instance.position)
      token_list = instance.tokens
      token_set = set(token_list)
      x.extend([1]*len(token_set))
      
      # construct the weights token
      for token in token_set:
        weights.w_tokens[token] = weights.w_tokens.get(token,0)+0.0
        # record most recent appearance moment of a feature
        # weights.access_time[token] = weights.access_time.get(token,count)
        #if weights.access_time.has_key(token):
        #weights.access_time[token] =count
        #else:
        #  weights.access_time[token] =count
      
      t131 = time.time()
      wt,weight_vec = self.compute_weight_feature_product(weights,x, instance)
      sigmoid = self.sigmoid(wt)
      # compute gradients and update, gradients are a list of gradient :
      # gradients = (y - sigmoid) * x
      # gradients = [xi*(y-sigmoid) for xi in x]
      gradients = np.multiply((y-sigmoid),x)
      # weights update
      # for w0
      # regualarization (1-lamda * step)
      t14 = time.time()
      downweight = (1- lambduh * step)
      weights.w0 = weights.w0 + np.multiply((y-sigmoid),step)
      weights.w_age= np.multiply(weights.w_age,downweight) + np.multiply(gradients[0],step)
      weights.w_gender = np.multiply(weights.w_gender,downweight) + np.multiply(gradients[1],step)
      weights.w_depth = np.multiply(weights.w_depth,downweight)+ np.multiply(gradients[2],step)
      weights.w_position = np.multiply(weights.w_position,downweight) + np.multiply(gradients[3],step)
      t16 = time.time()
      #print("-------update other weights----",t16-t14)
      # for token in token_set:
      #   print("weight",weights.w_tokens[token])
        
      temp_ct = 1
      for token in token_set:
        # delay regularization
        t12 = time.time()
        downweight_token = self.perform_delayed_regularization(token,weights,count,step,lambduh)
        t13 = time.time()
        #print("token update -----delay regulraiation---",t13-t12)
        weights.w_tokens[token] = np.multiply(weights.w_tokens[token],downweight_token)+ np.multiply(gradients[3+temp_ct],step)
        t14 = time.time()
        #print("token update -----token weight update ---",t14-t13)
        temp_ct += 1
        weights.access_time[token] =count
        #print("token update ---- updated weight",weights.w_tokens[token])
      t17= time.time()
      #print("--------total token update time----",t17-t16)
     

      # predict plot ave loss
      if sigmoid > 0.5:
        pred = 1.0
      else:
        pred = 0.0
      total_loss += math.pow((pred-y),2)
      # total_cost += self.cost(y,sigmoid)
      # l2_norm = weights.l2_norm()
      count += 1
      if count % 100 is 0:
        # print "Loaded " + str(count) + " lines"
        # print "avg training loss"
        # avg loss
        avg_loss_list.append(float(total_loss) / float(count))
        
    t132= time.time()
    print(" one epoch time: ", t132-t130)
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    print "plotting"
    self.plotloss(avg_loss_list,step)   
    dataset.reset()
      
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # return a list of predicted values
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    count = 0
    print "testing: Loading data from " + dataset.path + "..."
    predicted_y = list()
    while dataset.hasNext():
      t0 = time.time()
      instance = dataset.nextInstance()
      # y labels
      # y = instance.clicked
      # x : features
  
      x = list()
      x.append(instance.age)
      x.append(instance.gender)
      x.append(instance.depth)
      x.append(instance.position)
      token_list = instance.tokens
      token_set = set(token_list)
      # x.extend([1]*len(token_set))
      
      # construct the weights token
      # for token in instance.tokens:
      #   weights.w_tokens[token] = weights.w_tokens.get(token,0)+0.0
      t1 = time.time()
      # print("------ construct x",t1-t0)
      wt = self.compute_weight_feature_product_test(weights,x, instance)
      t2 = time.time()
      # print("------compute product--",t2-t1)
      sigmoid = self.sigmoid(wt)
      count += 1
      predicted_y.append(sigmoid)
      # if count % 100 is 0:
      #   # print "Loaded " + str(count) + " lines"
      #   # print "avg training loss"
      #   # avg loss
      #   avg_loss_list.append(float(total_loss) / float(count))
        # ave_cost_list.append(total_cost)
        
        # print(float(total_loss) / float(count))

    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
   
    dataset.reset()
    return predicted_y

  def plotloss(self,loss_list,step):
    fig = plt.figure()
    plt.plot(loss_list)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.02,0.05))
    filename = "ave_loss_"+str(step)+".png"
    fig.savefig(filename)
    plt.close()

  def plot_regularization(self,lambduh_list,l2_list,filename):
    fig = plt.figure()
    plt.plot(lambduh_list,l2_list)
    filename = filename
    fig.savefig(filename)
    plt.close()

  
  
if __name__ == '__main__':
  # TODO: Fill in your code here
  print "Training Logistic Regression..."
  lr = LogisticRegression()
  evaluation = EvalUtil()

  size = 10000
  # step = [0.001,0.01,0.05]
  step=0.05
  training = DataSet("train.txt", True, lr.TRAININGSIZE)
  test = DataSet("test.txt", False, lr.TESTINGSIZE)
  path_to_sol = 'test_label.txt'

  lambduhs = np.arange(0,0.014,0.002)
  # lambduhs = [0.002,0.012]
  print lambduhs
  n2norm_list =list()
  rmse_list = list()
  
  for lambduh in lambduhs:
    print("begin training with lamnda:",lambduh)
    t11 = time.time()
  
    weights = lr.train(training,lambduh,step,0)
    l2_norm = weights.l2_norm()
    n2norm_list.append(l2_norm)
    print("end training and begin prediction, time used:",time.time()-t11)
   
    pred_list = lr.predict(weights,test)
    print("prediction completed!")
    t3 = time.time()
    rmse = evaluation.eval(path_to_sol, pred_list)
    rmse_list.append(rmse)
    # print("---------time for rmse---",time.time()-t3)
    print("rmse",rmse)
    print("intercept, age, gender, depth, position",weights.w0,
    weights.w_age,weights.w_gender, weights.w_depth,weights.w_position)
    print "l2 norm %f" %l2_norm
  # plotting
  lr.plot_regularization(lambduhs,n2norm_list,"l2norm_lamda.png")
  lr.plot_regularization(lambduhs,rmse_list,"rmse_lamda.png")


#  RMSE of baseline
  pred_list_base = [0.0337]*lr.TESTINGSIZE
  rmse_baseline = evaluation.eval(path_to_sol, pred_list_base)
  print("rmse",rmse_baseline)
