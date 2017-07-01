from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from collections import defaultdict

class BasicAnalysis:
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    count = 0
    print "Loading data from " + dataset.path + "..."
    token_set = set()
    token_map = dict()
    while dataset.hasNext():
      instance = dataset.nextInstance()
      # Here we printout the instance. But your code for processing each
      # instance will come here. For example: tracking the max clicks,
      # update token frequency table, or compute gradient for logistic
      # regression...
      # token_set = set(instance.tokens).union(token_set)
      for token in instance.tokens:
        token_map[token] = token_map.get(token,0)+1
        
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
    return list(token_map.keys())

  
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    # TODO

    return set()

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    # TODO
    count = 0
    print "Loading data from " + dataset.path + "..."
    age_map = defaultdict(set)
    while dataset.hasNext():
      instance = dataset.nextInstance()
      
      age = instance.age
      userid = instance.userid
      if (age in age_map.keys()):
        temp_list2 = set(age_map.get(age))
        temp_list2.add(userid)
        age_map[age] = temp_list2
      else:
        temp_list = set()
        temp_list.add(userid)
        age_map[age]=temp_list

        
      # print str(instance.tokens)
      count += 1
      if count % 1000000 is 0:
        print "Loaded " + str(count) + " lines"
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    
    for key,val in age_map.iteritems():
      print "age group: %d" % key
      print "user count %d"% len(set(val))
    # Important: remember to reset the dataset everytime
    # you are done with processing.
    dataset.reset()
    return age_map
    

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    print "producing average stat...."
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
    return CRT_avg

if __name__ == '__main__':
  print "Basic Analysis..."