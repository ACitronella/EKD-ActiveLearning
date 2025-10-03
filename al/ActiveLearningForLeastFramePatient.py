# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import pandas as pd
from .Sampling import Sampling
import numpy as np


class ActiveLearningForLeastFramePatient:
    """
    Implements standard active learning methods.

    This class will pick the Active Set from the patient that has least frames in the dataset.
    """

    def __init__(self, dataObj, cur_episode, dataset_info, random_seed):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj, random_seed=random_seed)
        self.cur_episode = cur_episode
        self.dataset_info = dataset_info
        
    def sample_from_uSet(self, SAMPLING_FN, BUDGET_SIZE, emb_path, lSet, uSet, **kwargs):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), BUDGET_SIZE)
        relevent_set = np.concatenate([lSet, uSet])
        lset_patient_count = pd.Series([self.dataObj.get_patient_code_and_frame_from_idx(idx)["patient_code"] for idx in lSet]).value_counts().to_dict()
        relevent_patient = pd.Series([self.dataObj.get_patient_code_and_frame_from_idx(idx)["patient_code"] for idx in relevent_set]).unique()
        uset_patient_list = np.array([self.dataObj.get_patient_code_and_frame_from_idx(idx)["patient_code"] for idx in uSet])
        listof_patient = [(p_code, lset_patient_count.get(p_code, 0)) for p_code in relevent_patient]
        listof_patient = sorted(listof_patient, key=lambda x:x[1])
        p_code_tobe_selected = listof_patient[0][0]
        
        # make uset has only a patient 
        uSet = uSet[np.argwhere(uset_patient_list == p_code_tobe_selected).flatten()]
        uSet = np.setdiff1d(uSet, lSet)
        
        # sanity check
        patient_uset = pd.Series([self.dataObj.get_patient_code_and_frame_from_idx(idx)["patient_code"] for idx in uSet])
        assert (patient_uset == patient_uset[0]).all()

        if SAMPLING_FN == "random":
            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=BUDGET_SIZE)
        
        elif SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            DELTA = kwargs["delta"]
            probcov = ProbCover(lSet, uSet, budgetSize=BUDGET_SIZE,
                            delta=DELTA, embedding_path=emb_path)
            activeSet, uSet = probcov.select_samples()
        
        elif SAMPLING_FN == "embedding_difference_as_probability_density":
            # assert
            from .embedding_difference_as_probability_density import EmbeddingDifferenceAsProbabilityDensity
            dataset_info = kwargs["dataset_info"]
            al = EmbeddingDifferenceAsProbabilityDensity(lSet, uSet, BUDGET_SIZE,
                                                         emb_path, dataset_info, kernel_size=11)
            activeSet, uSet = al.select_samples()
        else:
            print(f"{SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        # return all patient to uset
        uSet = np.setdiff1d(relevent_set, np.concatenate([activeSet, lSet]))
        return activeSet, uSet
        
