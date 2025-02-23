#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Run the GAIN model
@Author      :   siyi.sun
@Time        :   2025/02/21 01:02:40
"""
from GAIN import GAIN
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# conda init
# source ~/.bashrc
# conda activate tf115_env


class experiments:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_path = Path("/home/siyi.sun/CMIE_Project/data_stored")
        # Output path for time recording
        self.output_file = Path("/home/siyi.sun/CMIE_Project/imputation_times_gain.csv")
        self.cohort = "C19"
        self.miss_methods = ["MCAR", "MAR", "MNAR"]
        self.miss_ratios = [10]

        ## In detail, if you want to measure the epoch size from performance of continuous features, mark "continuous_first"
        ## else mark "categorical_first"
        self.index_pick = "continuous_first"

        self.num_sampletest = 1  # num of test defined in ../Rcode/sample_miss.R
        self.num_experiments = 5  # num of train defined in ../Rcode/sample_miss.R
        self.batch_size = 64  # batch-size
        self.Epoch_sampletest = 100  # num of epoch during each sample test case

    def run_model(self):
        print("Start running the model on sample...")
        epoch_dict = self.run_sampletest()
        print("Start running the model on main datasets...")
        self.run_experiment(epoch_dict=epoch_dict)

    def run_sampletest(self):
        epoch_case = {}
        for miss_method in tqdm(self.miss_methods):
            epoch_case[miss_method] = {}
            for miss_ratio in tqdm(self.miss_ratios):
                epoch_stop = []
                for index_file in tqdm(range(self.num_sampletest)):
                    print("Model initialization...")
                    model = GAIN(
                        self.base_path,
                        self.cohort,
                        miss_method,
                        miss_ratio,
                        index_file,
                        self.batch_size,
                        self.Epoch_sampletest,
                        sampletest=True,
                        index_pick=self.index_pick,
                    )
                    print("Start training...")
                    index_stop = model.train_process_sample()
                    epoch_stop.append(index_stop)
                    print(
                        f"{miss_method},{miss_ratio}, Sample test {index_file} finished."
                    )

                epoch_case[miss_method][miss_ratio] = int(
                    np.around(np.mean(epoch_stop))
                )
                del epoch_stop
                gc.collect()
        return epoch_case

    def run_experiment(self, epoch_dict):
        time_records = []
        for miss_method in tqdm(self.miss_methods):
            for miss_ratio in tqdm(self.miss_ratios):
                imputation_times = []
                for index_file in tqdm(range(self.num_experiments)):
                    print("Model initialization...")
                    model = GAIN(
                        self.base_path,
                        self.cohort,
                        miss_method,
                        miss_ratio,
                        index_file,
                        self.batch_size,
                        epoch_dict[miss_method][miss_ratio],
                        sampletest=False,
                        index_pick=self.index_pick,
                    )
                    print("Start training...")
                    # Measure imputation time
                    start_time = time.time()
                    model.train_process()
                    end_time = time.time()
                    imputation_time = end_time - start_time
                    imputation_times.append(imputation_time)

                # Compute average time
                avg_time = sum(imputation_times) / self.num_experiments

                # Store result
                time_records.append([self.cohort, miss_method, miss_ratio, avg_time])

                # Save results to CSV
                time_df = pd.DataFrame(
                    time_records,
                    columns=["Dataset", "Mechanism", "MissingRatio", "AvgTime"],
                )
                time_df.to_csv(self.output_file, index=False)
                print(f"{miss_method}, {miss_ratio} Imputation times recorded.")
                gc.collect()


# Run the model
if __name__ == "__main__":
    print("Start initializing the experiment...")
    model_experiment = experiments(model_name="GAIN")
    print("Start running the experiment...")
    model_experiment.run_model()
