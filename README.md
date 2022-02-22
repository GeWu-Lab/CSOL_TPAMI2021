# **Class-aware Sounding Objects Localization**


Audiovisual scenes are pervasive in our daily life. It is commonplace for humans to discriminatively localize different sounding objects but quite challenging for machines to achieve class-aware sounding objects localization without category annotations, i.e., localizing the sounding object and recognizing its category. To address this problem, we propose a two-stage step-by-step learning framework to localize and recognize sounding objects in complex audiovisual scenarios using only the correspondence between audio and vision. First, we propose to determine the sounding area via coarse-grained audiovisual correspondence in the single source cases. Then visual features in the sounding area are leveraged as candidate object representations to establish a category-representation object dictionary for expressive visual character extraction. We generate class-aware object localization maps in cocktail-party scenarios and use audiovisual correspondence to suppress silent areas by referring to this dictionary. Finally, we employ category-level audiovisual consistency as the supervision to achieve fine-grained audio and sounding object distribution alignment. Experiments on both realistic and synthesized videos show that our model is superior in localizing and recognizing objects as well as filtering out silent ones. We also transfer the learned audiovisual network into the unsupervised object detection task, obtaining reasonable performance.

TPAMI 2021: https://ieeexplore.ieee.org/abstract/document/9662191

arxiv version: https://arxiv.org/abs/2112.11749


## **Dataset**

- MUSIC-Synthetic dataset: [Download](https://zenodo.org/record/4079386#.X4PFodozbb2)
- VGGSound-Synthetic: [Download](#)
- DailyLife: [Download](#)
- Realistic MUSIC: [Download](#)

## **Code**

The [**Code**](https://github.com/GeWu-Lab/CSOL_TPAMI2021) is implemented on PyTorch with python3. 





### Requirements

- PyTorch 1.1
- torchvision
- scikit-learn
- librosa
- Pillow
- opencv

### Running Procedure

For experiments on Music, AudioSet-instrument and VGGSound, the training and evaluation procedures are similar, respectively under the folder `music-exp` and `audioset-instrument`. Here, we take the experiments on Music dataset as an example.

#### Data Preparation

- Download dataset, e.g., MUSIC, and split into training/validation/testing set. Specifically, for the training@stage_one, please use the [solo_training_1.txt](https://github.com/DTaoo/Discriminative-Sounding-Objects-Localization/blob/master/music-exp/data/data_indicator/music/solo/solo_training_1.txt). For the training@stage_two, we use the the music clip in [solo_training_2.txt](https://github.com/DTaoo/Discriminative-Sounding-Objects-Localization/blob/master/music-exp/data/data_indicator/music/solo/solo_training_2.txt) to synthesize the [cocktail-party scenarios](https://zenodo.org/record/4079386#.X4PFodozbb2).

- Extract frames at 4 fps by running 

  ```
  python3 data/cut_video.py
  ```

- Extract 1-second audio clips and turn into Log-Mel-Spectrogram by running

  ```
  python3 data/cut_audio.py
  ```

The sounding object bounding box annotations on solo and duet are stored in `music-exp/solotest.json` and `music-exp/duettest.json`, and the data and annotations of synthetic set are available at https://zenodo.org/record/4079386#.X4PFodozbb2 . And the Audioset-instrument balanced subset bounding box annotations are in `audioset-instrument/audioset_box.json`

#### Training

##### Stage one

```
training_stage_one.py [-h]
optional arguments:
[--batch_size] training batchsize
[--learning_rate] learning rate
[--epoch] total training epoch
[--evaluate] only do testing or also training
[--use_pretrain] whether to initialize from ckpt
[--ckpt_file] the ckpt file path to be resumed
[--use_class_task] whether to use localization-classification alternative training
[--class_iter] training iterations for classification of each epoch
[--mask] mask threshold to determine whether is object or background
[--cluster] number of clusters for discrimination
```

```
python3 training_stage_one.py
```

After training of stage one, we will get the cluster pseudo labels and object dictionary of different classes in the folder `./obj_features`, which is then used in the second stage training as category-aware object representation reference.

##### Stage two

```
training_stage_two.py [-h]
optional arguments:
[--batch_size] training batchsize
[--learning_rate] learning rate
[--epoch] total training epoch
[--evaluate] only do testing or also training
[--use_pretrain] whether to initialize from ckpt
[--ckpt_file] the ckpt file path to be resumed
```

```
python3 training_stage_two.py
```

#### Evaluation

##### Stage one

We first generate localization results and save then as a pkl file, then calculate metrics, IoU and AUC and also generate visualizations, by running

```
python3 training_stage_one.py --mode test --use_pretrain 1 --ckpt_file your_ckpt_file_path
python3 tools.py
```

##### Stage two

For evaluation of stage two, i.e., class-aware sounding object localization in multi-source scenes, we first match the cluster pseudo labels generated in stage one with gt labels to accordingly assign one object category to each center representation in the object dictionary by running

```
python3 match_cluster.py
```

It is necessary to manually ensure there is one-to-one matching between object category and each center representation.

Then we generate the localization results and calculate metrics, CIoU AUC and NSA, by running

```
python3 test_stage_two.py
python3 eval.py
```

