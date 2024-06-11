
python -m cbig.osama2024.gen_cv_fold --spreadsheet data/TADPOLE_D1_D2.csv --features data/features --folds 10 --output_dir output 

python -m cbig.osama2024.gen_cv_pickle --spreadsheet data/TADPOLE_D1_D2.csv --features data/features --mask output/fold_0_mask.csv --strategy model --batch_size 128 --out output/val.pkl

python -m cbig.osama2024.train --verbose --data output/val.pkl --i_drop 0.1 --h_drop 0.1 --h_size 512 --epochs 50 --lr 5e-4 --model MinRNN --weight_decay 5e-7 --out output/model.pt

python -m cbig.osama2024.predict --checkpoint output/model.pt --data output/val.pkl -o output/prediction_val.csv


# for test data
python -m cbig.osama2024.predict --checkpoint output/model.pt --data output/test.pkl -o output/prediction_test.csv
python -m cbig.osama2024.evaluation --reference output/fold_0_test.csv --prediction output/prediction_test.csv




python -m cbig.osama2024.baseline_constant --spreadsheet data/TADPOLE_D1_D2.csv --mask output/fold_0_mask.csv --out output/constant_prediction.csv

python -m cbig.osama2024.evaluation --reference output/fold_0_test.csv --prediction output/constant_prediction.csv



python -m cbig.osama2024.baseline_svm  --spreadsheet data/TADPOLE_D1_D2.csv --mask output/fold_0_mask.csv --features data/features --agamma .01 --vgamma .01 --out output/svm_prediction.csv

python -m cbig.osama2024.evaluation --reference output/fold_0_test.csv --prediction output/svm_prediction.csv





