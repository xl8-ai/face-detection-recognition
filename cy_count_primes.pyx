import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

cpdef int count_primes(limit:int):
  """ Vanilla python that returns the number of primes between 0 and [limit] """
  arr1 = np.array([1, 2, 3, 4, 5])
  print(arr1[2:4].shape)
  count:int = 0
  for candidate_int in range(limit):
    if (candidate_int > 1):
      for factor in range(2, candidate_int):
        if candidate_int % factor == 0:
          break
      else:
        count += 1
  return count
