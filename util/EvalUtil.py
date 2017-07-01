import math

class EvalUtil:
  # def __init__(self,path_to_sol):
  #   self.path_to_sol = path_to_sol
  
  # =====================
  # Evaluates the model by computing the weighted rooted mean square error of
  # between the prediction and the true labels.
  # 
  # @param path_to_sol
  # @param ctr_predictions
  # @return the weighted rooted mean square error of between the prediction
  #         and the true labels.
  # ======================
  @classmethod
  def eval(cls, path_to_sol, ctr_predictions):
    size = len(ctr_predictions)
    wmse = 0.0
    with open(path_to_sol, 'r') as f:
      # count = 0
      for i, line in enumerate(f):
        if i == size:
          break 
        # print("i",i)
        # print("ctr_predictions[i]",ctr_predictions[i])
        ctr = float(line)
        wmse += pow((ctr - ctr_predictions[i]), 2)
        
        # count+=1
      # print("len of solution %d" %count)

    return math.sqrt(wmse / size)
  
  # =====================
  # Evaluates the model by computing the weighted rooted mean square error of
  # between the prediction and the true labels.
  # 
  # @param path_to_sol
  # @param path_to_predictions
  # @return the weighted rooted mean square error of between the prediction
  #         and the true labels.
  # =====================
  @classmethod
  def eval_paths(cls, path_to_sol, path_to_predictions):
    ctr_predictions = []
    with open(path_to_predictions, 'r') as f:
      for line in f:
        ctr_predictions.append(float(line))
    return cls.eval(path_to_sol, ctr_predictions)

  # =====================
  # Evaluates the model by computing the weighted rooted mean square error of
  # between the prediction and the true labels.
  # @param path_to_sol {String} path to solution
  # @param average_ctr {Float} average CTR
  # @return {Float} baseline RMSE
  # =====================
  @classmethod
  def eval_baseline(cls, path_to_sol, average_ctr):
    rmse = 0.0
    count = 0
    with open(path_to_sol, 'r') as f:
      for line in f:
        ctr = float(line)
        rmse += math.pow(ctr - average_ctr, 2)
        count += 1
    return math.sqrt(rmse / count)
  
  # =====================
  # Evaluates the model by computing the weighted rooted mean square error of
  # between the prediction and the true labels, using the including list to decide whether the each
  # test data point should be included in evaluation.
  # 
  # This is useful for evaluating the RMSE on a subset of test data (with common users ...).
  # @param path_to_sol
  # @param ctr_predictions
  # @param including_list
  # @return
  # =====================
  @classmethod
  def eval_with_including_list(cls, path_to_sol, ctr_predictions, including_list):
    wmse = 0.0
    total = 0
    with open(path_to_sol, 'r') as f:
      for i, line in enumerate(f):
        if not including_list[i]:
          continue
        ctr = float(line)
        wmse += pow((ctr - ctr_predictions[i]), 2)
        total += 1
    return math.sqrt(wmse / total)