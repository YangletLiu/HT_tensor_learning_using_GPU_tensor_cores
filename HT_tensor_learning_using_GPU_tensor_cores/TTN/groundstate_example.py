# Copyright 2019 Ashley Milsted
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tree Tensor Network for the groundstate of the Transverse Ising chain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tree_tensor_network
import time

if __name__ == "__main__":
  backend = "jax"  # "numpy" and "tensorflow" are also supported!

  if backend == "tensorflow":
    import tensorflow as tf
    dtype = tf.float32
  elif backend == "jax":
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    dtype = np.float32
  elif backend == "numpy":
    import numpy as np
    dtype = np.float32

  tree_tensor_network.set_backend(backend)

  num_layers = 4
  max_bond_dim = 16
  build_graphs = True

  num_sweeps = 10

  # i = 1-6;
  #
  Ds = [min(2**i, max_bond_dim) for i in range(1, num_layers + 1)]
  

  print("----------------------------------------------------")
  print("Variational ground state optimization.")
  print("----------------------------------------------------")
  print("System size:", 2**num_layers)
  print("Bond dimensions:", Ds)

  H = tree_tensor_network.get_ham_ising(dtype)
  isos_012 = tree_tensor_network.random_tree_tn_uniform(Ds, dtype, top_rank=1)
  print("--------------------------------------------------------")

  t0 = time.time()
  isos_012 = tree_tensor_network.opt_tree_energy(
      isos_012,
      H,
      num_sweeps,
      1,
      verbose=1,
      graphed=build_graphs,
      ham_shift=0.2)
  cost = time.time() - t0
  print("cost time : ",cost)
