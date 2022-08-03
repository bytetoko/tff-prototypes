"""This is an attempt to understand working with ClientData and loading of remote datasets.

Based on.
https://www.tensorflow.org/federated/tutorials/working_with_client_data

The script attempts to compute a federated average of the data local to each TFF worker.

The data on each local worker is just a Tensor containing the length of the URI for the data on that Tensor.
"""
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation.datasets import client_data

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

class DataBackendForURI(tff.framework.DataBackend):
    async def materialize(self, data, type_spec):
      """This is the function used to materalize the dataset inside the worker from the URI. 
      """
      # We simulate loading from the URI by creating a tensor whose value is the length of the URI.
      # In practice, the URI would be something like a GCS URI. 
      # TODO(jeremy): If the coordinator knows the URI of the data on each client (e.g. the GCS path); then it
      # can pass it to each worker. In practice, the coordinator won't know the exact location; it might know
      # some key (e.g. the application name) that would be the same for each client but each client would then
      # map that key to a different location (e.g. GCS path). For example by looking at an environment variable. 
      # How do you that given this function has to be serializable? What would be the commands to create delayed
      # evaluation of an environment variable? One solution might be to use tf.data.Dataset to read a file whose
      # path is the same on each worker. The contents of the file would then be the actual GCS path.
      client_dataset = tf.data.Dataset.from_tensors(float(len(data.uri)))
      return client_dataset

def run(data_uris=None):
  """This runs the federated computation.

  Args:
    data_uris: List of URIs of the data; one for each client.
  """
  if data_uris is None:
    data_uris = ["5char", "10charchar"]

  # Plug the DataBackend into the ExecutionContext  
  def ex_fn(
    device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device),
        data_backend=DataBackendForURI())
  
  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
  ctx = tff.framework.ExecutionContext(executor_fn=factory)
  tff.framework.set_default_context(ctx)

  # Materialize an example client dataset. This is just a way to get
  # element spec. 
  example_dataset = tf.data.Dataset.from_tensors(float(1))
  element_spec = example_dataset.element_spec

  # Create a handle to the data to be passed to the Federated computations
  dataset_type = tff.types.SequenceType(element_spec)
    
  data_handle = tff.framework.CreateDataDescriptor(arg_uris=data_uris, arg_type=dataset_type)

  expected = np.mean([len(u) for u in data_uris])

  result = get_global_temperature_average(data_handle)
  print(f"Actual={result}, Expected={expected}")

def main():
  run()

if __name__ == "__main__":
  main()