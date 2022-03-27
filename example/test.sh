python ../feature_gen.py --target example/chains611_tmp81 --input_dir example/inputs --output_dir example/feature/
echo "feature save"
python ../predictor.py -m models/pretrain -t example/chains611_tmp81 -f example/feature -o example/results -g 0
echo "prediction end"

