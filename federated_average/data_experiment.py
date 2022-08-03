"""This is an attempt to understand working with ClientData and loading of remote datasets.

Based on.
https://www.tensorflow.org/federated/tutorials/working_with_client_data

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


def dataset_builder(client_id: str) -> tf.data.Dataset:
  """This is the function that is invoked on each worker to construct the dataset from the client_id.  

  It must be serializable
  """
  # This is 
  return tf.data.Dataset.from_tensors(float(client_id))


def create_backend(client_data):
  # bind TestDataBackend to client_data. client_data is an instance
  # of ConcreteClientData which is serializable. So it can be shipped
  # to the TFF workers. When run on the TF worker it will construct
  # the dataset for the given client.
  class TestDataBackend(tff.framework.DataBackend):
    async def materialize(self, data, type_spec):
      client_id = int(data.uri[-1])
      client_dataset = client_data.create_tf_dataset_for_client(
          client_data.client_ids[client_id])
      return client_dataset
  return TestDataBackend

def run():
  """This runs the federated computation.  
  """
  data = client_data.ConcreteClientData(["1",  "12"], dataset_builder)
  DataBackendClass = create_backend(data)

  # Plug the DataBackend into the ExecutionContext
  def ex_fn(
    device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device),
        data_backend=DataBackendClass())
  
  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
  ctx = tff.framework.ExecutionContext(executor_fn=factory)
  tff.framework.set_default_context(ctx)

  # Materialize one of the client datasets. This is just a way to get
  # element spec. Presumably in actualy scenario where we can't materialize the
  # data from one client we would just specify the element_spec directly
  example_dataset = data.create_tf_dataset_for_client("1")
  element_spec = example_dataset.element_spec

  # Create a handle to the data to be passed to the Federated computations
  dataset_type = tff.types.SequenceType(element_spec)

  data_uris = [f'uri://{i}' for i in range(2)]
  data_handle = tff.framework.CreateDataDescriptor(arg_uris=data_uris, arg_type=dataset_type)

  result = get_global_temperature_average(data_handle)
  # N.B. 
  print(f"Actual={result}")

def main():
  run()

if __name__ == "__main__":
  main()