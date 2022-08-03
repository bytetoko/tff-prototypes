"""This is an attempt to do the simplest possible federated analytics.

It is based on: https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1

Per the tutorial. It relies on the lower level federate core (FC) interfaces.
"""
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation(tff.SequenceType(tf.float32))
def get_local_temperature_average(local_temperatures):
  """This function uses TF to define the computations to be run on each worker.
  
  The argument of tf_computation defines the type of the input. Using tff.SequenceType
  means each worker will be instantiated with a sequence of float 32s. On
  the worker the instantiated type will be of tf.Dataset.

  """
  # The concrete type of local_temperatures on the workers will be a TF.Dataset.
  # We use the tf.Dataset reduce function to compute the local average for the
  # values of that worker.
  sum_and_count = (
      local_temperatures.reduce((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1)))
  return sum_and_count[0] / tf.cast(sum_and_count[1], tf.float32)

@tff.federated_computation(
  tff.type_at_clients(tff.SequenceType(tf.float32)))
def get_global_temperature_average(sensor_readings):
  """"This defines the federated computation."""
  # get_local_temperature_average operates on a single workers values; so to
  # apply it to all values in a federated dataset we use federated map
  return tff.federated_mean(
      tff.federated_map(get_local_temperature_average, sensor_readings))

def run_local(data=None):
  """This runs the federated computation locally.
  
  data is a list of lists. The outer list iterates over the list of devices
  represented by the group tff.CLIENTs. Inner list is the data for each device
  e.g.
  data[i] is a list containing the items for the i'th worker.
  """
  # Load default data if none is provided.
  if data is None:
    data = [[68.0, 70.0], [71.0], [68.0, 72.0, 70.0]] 

  # Compute the expected value to confince ourselves
  # Note that the way the federated average is implemented each sample isn't properly weighted.
  # Instead the algorithm works by computing the average local to each worker and then averaging
  # the averages (so giving equal weight to the average from each worker)
  expected = np.mean([np.mean(x) for x in data])
  result = get_global_temperature_average(data)
  # N.B. 
  print(f"Actual={result} Properly Weighted={expected}")

def main():
  run_local()

if __name__ == "__main__":
  main()