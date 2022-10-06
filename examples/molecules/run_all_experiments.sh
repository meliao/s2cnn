#!/bin/zsh
runid=0;
for run in 0 1 2 3 4;
do
    for strat in 0 1 2 3 4;
    do
        echo "starting run $run for strat $strat"
        log_file=logs/default_settings_strat_${strat}_run_${run}.txt
        python3 run_experiment.py --log_file $log_file --data_path data/data.joblib
    done;
done;

