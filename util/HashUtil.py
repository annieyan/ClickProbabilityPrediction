class HashUtil:
  @classmethod
  def hash_to_range(cls, s, upper):
    hashval = hash(str(s)) % upper
    if hashval < 0:
      hashval = upper + hashval
    return hashval
    
  @classmethod
  def hash_to_sign(cls, s):
    if hash(str(s)) % 2 == 0:
      return -1
    else:
      return 1
