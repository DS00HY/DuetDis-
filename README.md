# DuetDis 
DuetDis is a method based on duet feature sets and deep residual network with squeeze-and-excitation (SE), for protein inter-residue distance prediction. Duet embraces the ability  to learn and fuse features directly or indirectly extracted from the whole-genome/metagenomic databases, and therefore  minimize the information loss through ensembling models trained on different feature sets.

**Requirements**
*	[Python (>= 3.6)](https://www.python.org/downloads)
*	[Pytorch (1.10)](https://pytorch.org/)
*	Protein Dataset

**1. Command for generating features**

```
python feature_gen.py -t [target_chainlist] -f [input_dir] -o [output_dir]
```
After that, you can see xxx_features_m1.npz and xxx_features_m3.npz in your output_dir. The m1.npz is the features for our M1 and M2 model, and m3.npz is the features for our M3, M4 and M5 model.

You should prepare your protein data for input correctly, which like example/inputs. Otherwise, features may not be generated.

**2. Command for prediction** 
```
python predictor.py -m [pretrain_models_dir] -t [target_chainlist] -f [feature_dir] -o [output_dir] -g [gpu_id] --mid [model_to_use]
```
After that, You can see you result like xxx.map_std in your output_dir. 
Our pretrain models is put in models/pretrain, you can use these models to predict your protein. You can use --mid to choose which model to use, our default param is "12345".

**3. An example:**
```
python feature_gen.py --target example/chains611_tmp81 --input_dir example/inputs --output_dir example/feature/

python predictor.py -m models/pretrain -t example/chains611_tmp81 -f example/feature -o example/results -g 0
```

**References**

1. Huiling Zhang, Ying Huang.et al. Inter-Residue Distance Prediction from Duet Deep Learning models()
2. [tr-rosetta-pytorch](https://github.com/lucidrains/tr-rosetta-pytorch)