# Guide
This code is reference the website: https://github.com/Mckinsey666/ACGAN-Conditional-Anime-Generation

## Usage
### Training 
If you want to train the normal one, you can use the command like this:
```
>>> cd HW4
>>> python train_split.py --root_dir <direction of 'selected_cartoonset100k'>
```
If you want to train the GAN by unsupervised conditional case, you can use the command like this:
```
>>> cd Bonus
>>> bash bonus.sh <training_data_dir>
```
And, the graph of training result, process will be save automatically.
### Testing
If you want to test the normal one, you can use the command like this:
```
>>> bash cgan.sh <testing_labels.txt> <output_dir>
```
### For example
If you want to generate the FID image, you can type the command like:
```
>>> bash cgan.sh ../sample_test/sample_fid_testing_labels.txt ./results/fid
```
If you want to generate the human evaluation image, you can type the command like:
```
>>> bash cgan.sh ../sample_test/sample_human_testing_labels.txt ./results/human
```

## Library page used
1. pytorch
2. tensorflow
3. PIL
4. tqdm