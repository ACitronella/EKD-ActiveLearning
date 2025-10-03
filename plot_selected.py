import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


if __name__ == "__main__":
    # folder_name = "random"
    # folder_name = "embedding_difference_as_probability_density"
    folder_name = "probcover"

    DATASET_INFO_PATH = "isblinkingonly/dataset_info.csv"
    dataset_prefix_path = "isblinkingonly"
    save_path = os.path.join('selected_idx/leaveoneout', folder_name)

    print(f"Plotting selected frame from {folder_name}")

    os.makedirs(save_path, exist_ok=True)
    metrics_dir = os.path.join("activeset", folder_name)

    dataset_info = pd.read_csv(DATASET_INFO_PATH)
    dataset_info["keypoint_file"] = dataset_info["keypoint_file"].apply(lambda x: os.path.join(dataset_prefix_path, x))
    all_patient_code = dataset_info["patient_code"].unique()
    model_dir = os.path.join(metrics_dir, f"{all_patient_code[0]}") # assume every fold must have the same number of trainps
    
    tb = np.append([0], dataset_info["frames"].cumsum())

    for pa_idx, p_code in enumerate(all_patient_code):
        model_dir = os.path.join(metrics_dir, p_code)
        
        rows = dataset_info.loc[dataset_info['patient_code'] == p_code]
        kp_files = rows["keypoint_file"].values
        if len(kp_files) == 4:
            kp_files = [(kp_files[0], kp_files[2]), (kp_files[1], kp_files[3])]
        fig, axes = plt.subplots(2, 1, figsize=(7, 0.1+len(kp_files)), dpi=150)
        kp_files2axes = {}
        is_blinking = None
        for ax, kp_path in zip(axes, kp_files):
            kp_files2axes[kp_path] = ax
            if isinstance(kp_path, str):
                vid_kp = np.load(kp_path)
            elif isinstance(kp_path, tuple):
                vid_kp1 = np.load(kp_path[0])
                vid_kp2 = np.load(kp_path[1])
                vid_kp = {'is_blinking': np.append(vid_kp1["is_blinking"], vid_kp2["is_blinking"])}
            else:
                assert False
            if is_blinking is None:
                is_blinking = vid_kp["is_blinking"].astype('int32')
            else:
                is_blinking = is_blinking | vid_kp["is_blinking"].astype('int32')
            c = st = False
            s = []; e = []
            for idx, b in enumerate(is_blinking):
                if b != c:
                    if st: e.append(idx)
                    else:  s.append(idx)
                    st = not st; 
                    c = b
            if is_blinking[-1]: 
                e.append(len(is_blinking)-1)
        for ax in axes:
            ax.set_xlim(0, len(is_blinking))
            ax.tick_params(axis='x', labelsize=14)
            ax.set_ylim(0, 1); ax.set_yticks([])
            for s_idx, e_idx in zip(s, e):
                ax.fill_between([s_idx, e_idx], 0, 1, alpha=0.4, color="C9")
        
        for p_idx in range(len(os.listdir(model_dir))):
            activeset = np.load(os.path.join(model_dir, f"episode_{p_idx}", "activeSet.npy"))
            for a_idx in activeset:
                file_idx = np.argmax(tb > a_idx) - 1 # get first file that >= idx
                if file_idx == 2 or file_idx == 3:
                    frame_idx = a_idx - tb[file_idx] + tb[1]
                    kp_path = (dataset_info.loc[file_idx-2, "keypoint_file"], dataset_info.loc[file_idx, "keypoint_file"])
                elif file_idx == 0 or file_idx == 1:
                    frame_idx = a_idx - tb[file_idx]
                    kp_path = (dataset_info.loc[file_idx, "keypoint_file"], dataset_info.loc[file_idx+2, "keypoint_file"])
                else:
                    frame_idx = a_idx - tb[file_idx]
                    kp_path = dataset_info.loc[file_idx, "keypoint_file"]

                ax = kp_files2axes[kp_path]
                ax.plot([frame_idx, frame_idx], [0, 1], c=f"C{p_idx}", linewidth=3)
                
                ax.set_title(f"{dataset_info.loc[file_idx, 'patient_code']} {dataset_info.loc[file_idx, 'keypoint_file'][-5].upper()}", fontsize='x-large')
        plt.tight_layout(h_pad=0.5)
        plt.savefig(os.path.join(save_path, f"{p_code} selection idx.png"))
        plt.show()    
        plt.close()
