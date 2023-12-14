# Acceleration-Guided Diffusion Model for Multivariate Time Series Imputation


## File Structure
* Code: source code of our implementation
* Data: some source files of datasets used in experiments


## Preprocessing each dataset
0. Enter the "Code" folder

1. To get the Beijing18 dataset:
```
python loadBeijing18Dataset.py
```

2. To get UrbanTraffic dataset:
```
python loadUrbanTrafficDataset.py
```
3. To get the PhysioNet12 dataset:
```
python loadPhysioNet12Dataset.py
```

## Demo Script Running
```
python A_diffusion_train.py --dataset beijing18 --missing_rate 0.1 --enc_in 99 --c_out 99
```

```
python A_diffusion_train.py --dataset urbantraffic --missing_rate 0.1 --enc_in 214 --c_out 214
```

```
python A_diffusion_train.py --dataset physionet12  --enc_in 37 --c_out 37
```

## Dataset Sources
* Beijing18: http://www.kdd.org/kdd2018/
* UrbanTraffic: https://zenodo.org/record/1205229
* PhysioNet12: https://physionet.org/content/challenge-2012/1.0.0/
