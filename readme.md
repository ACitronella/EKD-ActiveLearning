# EKD-ActiveLearning

The code is for demonstrate active learning algorithms from the paper "A Long-Term Active Learning Framework for Eyelid Keypoint Detection in High-Frame-Rate Blinking Videos".

## Installing Dependencies

Tested on python 3.7.13

```bash
conda create env --name env1 python=3.7.13
conda activate env1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy pandas matplotlib tqdm scipy scikit-learn
```

## Downloading data

Download link: [Google Drive](https://drive.google.com/file/d/1maw8QuWMZ2_FovZwP6jmHy42HizEJwd2/view?usp=sharing)

A zip file contains 
1. `emb`, embeddings of each experiment settings.
2. `isblinkingonly`, array of boolean for each frame of video indicates that whether the patient is blinking in that frame.

Place both folders in home directory `./` as follow

```
|- ./
    |- al
    |- emb
    |- isblinkingonly
    |- ... (other python files)
```

## Running 

### RQ1

Performs an active learning `$SAMPLING_FN` in RQ1 scheme. Running this will create `activeset/leaveoneout/$SAMPLING_FN`.
```bash
python leaveoneout.py --sampling_fn $SAMPLING_FN
```

### RQ2

Performs an active learning `$SAMPLING_FN` in RQ2 scheme. Running this will create `activeset/progressive/$SAMPLING_FN`.

```bash
python progressive.py --sampling_fn $SAMPLING_FN
```

### Plotting selected data from RQ1

Running this will use selected index from `activeset/leaveoneout/$SAMPLING_FN` to create `selected_idx/leaveoneout/$SAMPLING_FN`.

```bash
python plot_selected.py --sampling_fn $SAMPLING_FN
```

All available sampling_fn are "random", "embedding_difference_as_probability_density", "probcover".
