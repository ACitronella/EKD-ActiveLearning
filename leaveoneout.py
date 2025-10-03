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
    EMBEDDING_PATH = "emb/blink_all_ex_{patient_code}/pretext/features_seed132_yml.npy"
    DATASET_INFO_PATH = "isblinkingonly/dataset_info.csv"
    random_seed = 132
    delta = 0.25 # probcover's hyperparamerter
    print("RQ1: Performing active learning on left over patient")

    dataset_info = pd.read_csv(DATASET_INFO_PATH)
    all_patient_code = dataset_info["patient_code"].unique()
    data_obj = Data(dataset_info)
    
    for patient_code in all_patient_code:
        print(f"Leaving {patient_code} out")
        exp_patient_dir = f"activeset/leaveoneout/{SAMPLING_FN}/{patient_code}"
        train_patient_code = np.setdiff1d(all_patient_code, patient_code)
        _, uSet = Data.makeLUSetsByPatientsNotSave(train_patient_code, data_obj)
        lSet = np.asarray([])
        emb_path = EMBEDDING_PATH.format(patient_code=patient_code)

        for cur_episode in range(0, MAX_ITER):
            print("======== EPISODE {} BEGINS ========\n".format(cur_episode))

            # Creating output directory for the episode
            episode_dir = os.path.join(exp_patient_dir, f'episode_{cur_episode}')
            os.makedirs(episode_dir, exist_ok=True)

            # Active Sample 
            print("======== ACTIVE SAMPLING ========\n")
            al_obj = ActiveLearning(data_obj, cur_episode+1, dataset_info, random_seed=random_seed)
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
