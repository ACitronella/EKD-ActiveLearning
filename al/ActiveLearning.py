# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling
import numpy as np

class ActiveLearning:
    """
    Implements standard active learning methods.
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

        return activeSet, uSet
        
