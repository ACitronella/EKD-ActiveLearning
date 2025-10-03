import numpy as np

class Data():
    def __init__(self, dataset_info) -> None:
        self.dataset_info = dataset_info
        self.indices_table = np.append([0], dataset_info["frames"].cumsum())

    def __len__(self):
        return self.dataset_info["frames"].sum()
    
    def saveSets(self, lSet, uSet, activeSet, save_dir):

        lSet = np.array(lSet, dtype="int64")
        uSet = np.array(uSet, dtype="int64")
        activeSet = np.array(activeSet, dtype="int64")

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/activeSet.npy', activeSet)

    @staticmethod
    def makeLUSetsByPatientsNotSave(labeled_patient_list, dataset): # rename pls
        # assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)
        # assert any([ds_name in self.dataset for ds_name in ["blink"]]), "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, ["blink2"])
        lSet = []
        uSet = []
        
        n_dataPoints = len(dataset)
        all_idx = [i for i in range(n_dataPoints)]

        dataset_info = dataset.dataset_info
        indices_table = dataset.indices_table
        patient_code = dataset_info["patient_code"].unique()

        unlabel_patient_code = np.setdiff1d(patient_code, labeled_patient_list)
        for p_code in unlabel_patient_code:
            this_patient_info = dataset_info[dataset_info["patient_code"] == p_code]
            for idx in this_patient_info.index:
                uSet.append(np.arange(indices_table[idx], indices_table[idx+1], dtype="int32"))
        uSet = np.concatenate(uSet)
        lSet = np.setdiff1d(all_idx, uSet)

        # sanity check
        train_patient = set([dataset.get_patient_code_and_frame_from_idx(idx)['patient_code'] for idx in lSet])
        unlabel_patient = set([dataset.get_patient_code_and_frame_from_idx(idx)['patient_code'] for idx in uSet])
        assert len(train_patient.intersection(unlabel_patient)) == 0, "In the seperation phase, label and unlabel must be disjoint."

        return lSet, uSet
    
    @staticmethod
    def makeLUNSetsByPatientsNotSave(labeled_patient_code, unlabeled_patient_code, dataset): # rename pls
        # assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)
        # assert any([ds_name in self.dataset for ds_name in ["blink"]]), "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, ["blink2"])
        lSet = [];uSet = [];nSet = []

        dataset_info = dataset.dataset_info
        indices_table = dataset.indices_table
        patient_code = dataset_info["patient_code"].unique()

        notuse_patient_code = np.setdiff1d(patient_code, np.concatenate([labeled_patient_code, unlabeled_patient_code]))
        for p_code in unlabeled_patient_code:
            this_patient_info = dataset_info[dataset_info["patient_code"] == p_code]
            for idx in this_patient_info.index:
                uSet.append(np.arange(indices_table[idx], indices_table[idx+1], dtype="int32"))
        uSet = np.concatenate(uSet)
        for p_code in notuse_patient_code:
            this_patient_info = dataset_info[dataset_info["patient_code"] == p_code]
            for idx in this_patient_info.index:
                nSet.append(np.arange(indices_table[idx], indices_table[idx+1], dtype="int32"))
        nSet = np.concatenate(nSet)
        for p_code in labeled_patient_code:
            this_patient_info = dataset_info[dataset_info["patient_code"] == p_code]
            for idx in this_patient_info.index:
                lSet.append(np.arange(indices_table[idx], indices_table[idx+1], dtype="int32"))
        lSet = np.concatenate(lSet)

        # sanity check
        train_patient = set([dataset.get_patient_code_and_frame_from_idx(idx)['patient_code'] for idx in lSet])
        unlabel_patient = set([dataset.get_patient_code_and_frame_from_idx(idx)['patient_code'] for idx in uSet])
        notuse_patient = set([dataset.get_patient_code_and_frame_from_idx(idx)['patient_code'] for idx in nSet])
        assert len(train_patient.intersection(unlabel_patient)
                   .union(train_patient.intersection(notuse_patient))
                   .union(unlabel_patient.intersection(notuse_patient))) == 0, "In the seperation phase, label and unlabel and validation must be disjoint."
        return lSet, uSet, nSet

    
    def get_patient_code_and_frame_from_idx(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]
        row = self.dataset_info.loc[file_idx]
        row["frame_idx"] = frame_idx
        row["dataset_idx"] = idx
        return row.to_dict()
