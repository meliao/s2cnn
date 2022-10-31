date=2022-10-31
results=${date}_S2CNN_latency_results.pickle
log=logs/${date}_inference_timing.txt


rm $log

python run_latency_testing.py \
--num_epochs_mlp 2 \
--num_epochs_s2cnn 2 \
--log_file $log \
--data_path data/data.joblib \
--n_latency_runs 5 \
--latency_results_fp $results