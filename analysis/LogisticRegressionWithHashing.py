import math

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from util.HashUtil import HashUtil
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy
from scipy.sparse import csc_matrix

class Weights:
  def __init__(self, featuredim):
    self.featuredim = featuredim
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # hashed feature weights
    # {range_hash_val : weights} in other words {i: weight}
    self.w_hashed_features = [0.0 for _ in range(featuredim)]
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}

  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Hashed Feature: "
    string += " ".join([str(val) for val in self.w_hashed_features])
    string += "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_hashed_features:
      l2 += w * w
    return math.sqrt(l2)


class LogisticRegressionWithHashing:
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552
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
  def compute_weight_feature_product(self, weights,x,compr_hash_fea, instance):
    # TODO: Fill in your code here
    fea_list = x
    weight_vec = list()
    weight_vec.append(weights.w_age)
    weight_vec.append(weights.w_gender)
    weight_vec.append(weights.w_depth)
    weight_vec.append(weights.w_position)

    for i in compr_hash_fea.keys():
       weight_vec.append(weights.w_hashed_features[i])
       fea_list.append(compr_hash_fea[i])
 
    dot_prod =  np.dot(weight_vec,fea_list)
    return dot_prod,fea_list
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param featureids {[Int]} list of feature ids
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, featureid, weights, now, step, lambduh):
    if weights.access_time.has_key(featureid):
      last_time = weights.access_time.get(featureid)
      downweight = math.pow((1- step*lambduh),now-last_time)
    else:
      downweight = 0
    return downweight

  
  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, dim, lambduh, step, avg_loss, personalized):
    weights = Weights(dim)
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

      hashed_fea_dict = {}
      # i_list = list()
      # hashed_features = [0.0 for _ in range(dim)]
      # construct the hashed tokens
      for token in token_set:
        # range_val = i = hashed feature key
        # token = j, i = hash(token), Xj = 1 or 0
        # fi (x)i = sum j:h(j)=1 sign(j) Xj
        i = HashUtil.hash_to_range(token,dim)
        sign_val =HashUtil.hash_to_sign(token)
        Xj = 1
        # hashed_features[i] = hashed_features[i]+sign_val
        hashed_fea_dict[i] = hashed_fea_dict.get(i,sign_val) +sign_val

      # obtain non-zero hashed features: compr_hash_fea
      compr_hash_fea = {k:v for k,v in hashed_fea_dict.items() if v!=0}
      wt,fea_list = self.compute_weight_feature_product(weights,x, compr_hash_fea,instance)
      sigmoid = self.sigmoid(wt)
      x= fea_list

      # compute gradients and update, gradients are a list of gradient :
      # gradients = (y - sigmoid) * x
      # gradients = [xi*(y-sigmoid) for xi in x]
      # x.extend(hashed_features)
      gradients = np.multiply((y-sigmoid),x)
      # weights update
      # for w0
      # regualarization (1-lamda * step)
      t14 = time.time()
      #print("compute weight feature product time -----", t14-t131)

      downweight = (1- lambduh * step)
      weights.w0 = weights.w0 + np.multiply((y-sigmoid),step)
      weights.w_age= np.multiply(weights.w_age,downweight) + np.multiply(gradients[0],step)
      weights.w_gender = np.multiply(weights.w_gender,downweight) + np.multiply(gradients[1],step)
      weights.w_depth = np.multiply(weights.w_depth,downweight)+ np.multiply(gradients[2],step)
      weights.w_position = np.multiply(weights.w_position,downweight) + np.multiply(gradients[3],step)
      t16 = time.time()
        
      temp_ct = 1
      for featureid in compr_hash_fea.keys():
        # delay regularization
        t12 = time.time()

        downweight_token = self.perform_delayed_regularization(featureid,weights,count,step,lambduh)
        # print("downweight_token",downweight_token)
        t13 = time.time()
        #print("token update -----delay regulraiation---",t13-t12)
        weights.w_hashed_features[featureid] = np.multiply(weights.w_hashed_features[featureid],downweight_token) + np.multiply(gradients[3+temp_ct],step)
        t14 = time.time()
        #print("token update -----token weight update ---",t14-t13)
        temp_ct += 1
        weights.access_time[featureid] =count
        #print("token update ---- updated weight",weights.w_tokens[token])

      t17= time.time()
      #print("--------total token update time----",t17-t16)
      count += 1
    t132= time.time()
    print(" one epoch time: ", t132-t130)

    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    # print "plotting"
    # self.plotloss(avg_loss_list,step)   
    dataset.reset()
    return weights
      

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # @param personalized {Boolean}
  # ==========================
  def predict(self, weights, dataset, personalized):
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
      dim = weights.featuredim
      # construct the hashed tokens
      hashed_fea_dict = {}
      # i_list = list()
      # hashed_features = [0.0 for _ in range(dim)]
      # construct the hashed tokens
      for token in token_set:
        # range_val = i = hashed feature key
        # token = j, i = hash(token), Xj = 1 or 0
        # fi (x)i = sum j:h(j)=1 sign(j) Xj
        i = HashUtil.hash_to_range(token,dim)
        sign_val =HashUtil.hash_to_sign(token)
        Xj = 1
        hashed_fea_dict[i] = hashed_fea_dict.get(i,sign_val) +sign_val
      
       # obtain non-zero hashed features: compr_hash_fea
      compr_hash_fea = {k:v for k,v in hashed_fea_dict.items() if v!=0}
      # construct the weights token
      # for token in instance.tokens:
      #   weights.w_tokens[token] = weights.w_tokens.get(token,0)+0.0
      t1 = time.time()
      # print("------ construct x",t1-t0)
      wt,fea_list = self.compute_weight_feature_product(weights,x, compr_hash_fea,instance)
      t2 = time.time()
      pred = self.sigmoid(wt)   
      count += 1
      predicted_y.append(pred)
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    dataset.reset()
    return predicted_y

  

  def plot_regularization(self,lambduh_list,l2_list,filename):
    fig = plt.figure()
    plt.plot(lambduh_list,l2_list)
    filename = filename
    fig.savefig(filename)
    plt.close()


  
if __name__ == '__main__':
  print "Training Logistic Regression with Hashed Features..."
  fea_dims = [101,12277,1573549]
  # fea_dims = [101,12277]
  lr = LogisticRegressionWithHashing()
  evaluation = EvalUtil()

  size = 10000
  # step = [0.001,0.01,0.05]
  step=0.01
  training = DataSet("train.txt", True, lr.TRAININGSIZE)
  test = DataSet("test.txt", False, lr.TESTINGSIZE)
  path_to_sol = 'test_label.txt'

  # lambduhs = np.arange(0,0.014,0.002)
  lambduh = 0.001
  # n2norm_list =list()
  rmse_list = list()
  for fea_dim in fea_dims:
    print("begin training with fea dim:",fea_dim)
    t11 = time.time()
    
  
    weights = lr.train(training, fea_dim, lambduh, step, 0, 0)
    # print("weights",weights.__str__())
    # l2_norm = weights.l2_norm()
    # n2norm_list.append(l2_norm)
    print("end training and begin prediction, time used:",time.time()-t11)
   
    pred_list = lr.predict(weights,test,0)
    print("prediction completed!")
    # print("pred_list",pred_list)
    t3 = time.time()
    rmse = evaluation.eval(path_to_sol, pred_list)
    # rmse_list.append(rmse)
    # print("---------time for rmse---",time.time()-t3)
    print("rmse",rmse)
    # print("intercept, age, gender, depth, position",weights.w0,
    # weights.w_age,weights.w_gender, weights.w_depth,weights.w_position)
    # print "l2 norm %f" %l2_norm
  # plotting
  # lr.plot_regularization(lambduhs,n2norm_list,"l2norm_lamda.png")
  # lr.plot_regularization(fea_dims,rmse_list,"rmse_feadim.png")
