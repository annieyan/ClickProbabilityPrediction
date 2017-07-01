from analysis.DataSet import DataSet
from analysis.BasicAnalysis import BasicAnalysis
from analysis.LogisticRegression import Weights,LogisticRegression

class DummyLoader:
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552

  # def __init__(self):
  #   self.token_size = 0
  #   self.user_size = 0
  
     

  # ==========================
  # Scan the data and print out to the stdout
  # @param dataset {DataSet}
  # ==========================
  def scan_and_print(self, dataset):
    count = 0
    print "Loading data from " + dataset.path + "..."
    # get a set of tokens
    token_set = set()
    token_map = dict()
    age_map = dict()
    while dataset.hasNext():
      instance = dataset.nextInstance()
      # Here we printout the instance. But your code for processing each
      # instance will come here. For example: tracking the max clicks,
      # update token frequency table, or compute gradient for logistic
      # regression...
      for token in instance.tokens:
        token_map[token] = token_map.get(token,0)+1

      # age = instance.age
      # age_map[age]=age_map.get(age,0)+1

      # Logistic regression

        
      # print str(instance.tokens)
      count += 1
      if count % 1000000 is 0:
        print "Loaded " + str(count) + " lines"
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    token_set = set(token_map.keys())
    token_set_size = len(token_set)
    print "token set size %d" %token_set_size
    # for key,val in age_map.iteritems():
    #   print "age group: %d" % key
    #   print "user count %d"% val
    # Important: remember to reset the dataset everytime
    # you are done with processing.
    dataset.reset()
    return token_set

    '''
    return summary of trainng or test data
    @param dataset
    '''
  @classmethod
  def basic_stat(self, dataset):
    print "producing stat...."
    count = 0
    print "Loading data from " + dataset.path + "..."
    CRT_sum = 0.0
    CRT_avg = 0.0
    while dataset.hasNext():
      instance = dataset.nextInstance()
        # Here we printout the instance. But your code for processing each
        # instance will come here. For example: tracking the max clicks,
        # update token frequency table, or compute gradient for logistic
        # regression...
        #print str(instance)
      CRT_sum = CRT_sum + instance.clicked


      count += 1
      if count % 1000000 is 0:
        print "Loaded " + str(count) + " lines"
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    CRT_avg = CRT_sum / float(self.TRAININGSIZE)
      
    print "Done. Total processed instances: %d" % count
    print "average CRT_avg: %.4f" % CRT_avg
      
      # Important: remember to reset the dataset everytime
      # you are done with processing.
    dataset.reset()

      




if __name__ == '__main__':
  loader = DummyLoader()
  
  size = 10
  ba = BasicAnalysis()
  
  # prints a dataset from the training data with size = 10;
  training = DataSet("train.txt", True, loader.TRAININGSIZE)
  # train_token_set = loader.scan_and_print(training)
  # loader.basic_stat(training)
  train_token_list = ba.uniq_tokens(training)
  # print train_token_list
  weight = Weights()
  if 110428 in train_token_list:
    print("has key ")
  
  # train_age_map = ba.uniq_users_per_age_group(training)
  # prints a dataset from the test data with size = 10;
  testing = DataSet("test.txt", False, loader.TESTINGSIZE)
  # test_token_set = loader.scan_and_print(testing)
  test_token_list = ba.uniq_tokens(testing)
  
  # test_age_map = ba.uniq_users_per_age_group(testing)
  train_token_list.extend(test_token_list)
  token_set = set(train_token_list)
  token_size = len(token_set)
  print "total unique token size in training and test %d" %token_size

