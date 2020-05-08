# Documentation

This project replicates implementation of the [Ottertune](https://db.cs.cmu.edu/papers/2017/p1009-van-aken.pdf0) by CMU.

## How to run:

* Download the dataset from `/courses/cs600/cs645/cs645/project3/project3_dataset.tar.gz` from EDLAB.
* The point of entry of code is `Code/Workload_Characterization/main.py`.
The output of the code is 'Pruned Metrics' after factor analysis and K-means.
The clustering algorithm can be switched to Kmeans/EM using the parameter `c_algo` ('kmeans' for K-means clustering and 'em' for EM clustering)
* Run `PreProcessing/WorkloadDivision&Pruning.ipynb` to divide the workload by workload Id and only keep pruned metrics.
For WorkloadB, run `PreProcessing/WorkloadWiseDivision-WorkloadB.ipynb` as we need to create validation set from 6th row of WorkloadB along with division by workload and pruning.
* Run `WorkloadMapping.py` to find nearest neighbours, map both workload to offline workload and latency prediction on workloadB.
The model type can be switched to GPR/Neural/Random forest using the parameter `model_type`.
 If `plot_val=True` in latency prediction, it will also plot scatter plot for ground truth and predicted value.
