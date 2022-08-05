# Remote Federated Average

This directory contains the code for doing a federated average using a set of remote workers.

Everything is intended to be run in a GKE cluster in particular

* The TFF workers
* The coordinator

The coordinator needs to run inside the cluster so that it can communicate with the
workers without exposing the workers to the internet.

# Code

* namespace - Contains resources to setup the the `tff` namespace which is where the workers run

* workers - Kustomize package for deploying the tff-workers

# Running it

1. Deploy the workers

   ```
   kustomize build workers | kubectl apply -f -
   ```