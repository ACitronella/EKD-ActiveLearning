import os
import numpy as np
import pandas as pd
from al.ActiveLearning import ActiveLearning
from utils import Data
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sampling_fn", required=True, type=str, choices=["random", "embedding_difference_as_probability_density", "probcover"])
    args = parser.parse_args()

    SAMPLING_FN = args.sampling_fn
    BUDGET_SIZE = 5
    MAX_ITER = 5
    EMBEDDING_PATH = "emb/blink_progressive_fold{fold_idx}/pretext/features_seed132_yml.npy"
    DATASET_INFO_PATH = "isblinkingonly/dataset_info.csv"
    random_seed = 132
    delta = 0.25 # probcover's hyperparamerter
    FOLDS = 5
    print("RQ2: Performing active learning for progressive training")

    dataset_info = pd.read_csv(DATASET_INFO_PATH)
    all_patient_code = dataset_info["patient_code"].unique()

    dataset_info.loc[dataset_info["patient_code"] == "A3104", "fold_idx"] = 0
    dataset_info.loc[dataset_info["patient_code"] == "C2104", "fold_idx"] = 0

    dataset_info.loc[dataset_info["patient_code"] == "A2111", "fold_idx"] = 1
    dataset_info.loc[dataset_info["patient_code"] == "C2205", "fold_idx"] = 1

    dataset_info.loc[dataset_info["patient_code"] == "A3102", "fold_idx"] = 2
    dataset_info.loc[dataset_info["patient_code"] == "C2202", "fold_idx"] = 2

    dataset_info.loc[dataset_info["patient_code"] == "A1115", "fold_idx"] = 3
    dataset_info.loc[dataset_info["patient_code"] == "C2201", "fold_idx"] = 3

    dataset_info.loc[dataset_info["patient_code"] == "A2211", "fold_idx"] = 4
    dataset_info.loc[dataset_info["patient_code"] == "C2204", "fold_idx"] = 4
    dataset_info["fold_idx"] = dataset_info["fold_idx"].astype(int)

    data_obj = Data(dataset_info)

    for fold_idx in range(FOLDS):
        exp_patient_dir = f"activeset/progressive/{SAMPLING_FN}/fold{fold_idx}"
        
        train_patient_code = dataset_info[dataset_info["fold_idx"] == (fold_idx + 1) % FOLDS]["patient_code"].unique()
        unlabel_patient_code = dataset_info[(dataset_info["fold_idx"] != (fold_idx + 2) % FOLDS) & (dataset_info["fold_idx"] != (fold_idx + 1) % FOLDS) & (dataset_info["fold_idx"] != (fold_idx) % FOLDS)]["patient_code"].unique()
        val_patient_code = dataset_info[(dataset_info["fold_idx"] == (fold_idx + 2) % FOLDS)]["patient_code"].unique()
         
        lSet, uSet, vSet = data_obj.makeLUNSetsByPatientsNotSave(train_patient_code, unlabel_patient_code, data_obj)

        emb_path = EMBEDDING_PATH.format(fold_idx=fold_idx)
        
        for cur_episode in range(0, MAX_ITER):
            print("======== EPISODE {} BEGINS ========\n".format(cur_episode))

            # Creating output directory for the episode
            episode_dir = os.path.join(exp_patient_dir, f'episode_{cur_episode}')
            os.makedirs(episode_dir, exist_ok=True)

            # Active Sample 
            print("======== ACTIVE SAMPLING ========\n")
            al_obj = ActiveLearning(data_obj, cur_episode+1, dataset_info, random_seed)
            activeSet, new_uSet = al_obj.sample_from_uSet(SAMPLING_FN, BUDGET_SIZE, emb_path, lSet, uSet, dataset_info=dataset_info, delta=delta)

            # Save current lSet, new_uSet and activeSet in the episode directory
            data_obj.saveSets(lSet, uSet, activeSet, episode_dir)

            # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
            lSet = np.append(lSet, activeSet).astype(int)
            uSet = new_uSet
            msg = "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet))
            print(msg)
            print("================================\n\n")
            
            if len(uSet) == 0:
                print("Get out of training loop since out of unlabeled set")
                break
