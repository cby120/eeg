import os
import re
import mne
import numpy as np


# get label based on task id and event Tid
def get_label(task_id, event_Tid):
    if event_Tid in (0, 1, 2):
        return 0
    elif task_id == (3, 4, 7, 8, 11, 12):
        if event_Tid == 1:
            return 1 # left fist
        else:
            return 2 # right fist
    else:
        if event_Tid == 1:
            return 3 # both fist
        else:
            return 4 # both feet


# slice task into [T, D] samples
def slice_task(path, subject_id, task_id):
    raw_data = mne.io.read_raw_edf(path)
    events, _ = mne.events_from_annotations(raw_data)
    dataframe = raw_data.to_data_frame()
    for i in range(len(events)):
        if i == len(events) - 1:
            data_perT = dataframe.iloc[events[i][0]:, 1:].values
        else:
            data_perT = dataframe.iloc[events[i][0]:events[i+1][0], 1:].values
        label_perT = get_label(task_id, events[i][2])
        data_array = np.array(data_perT)
        # save one sample as an npz file
        np.savez_compressed(f"../data/{subject_id:03d}/{task_id:02d}/{label_perT}_{i:02d}.npz", data=data_array)


if __name__ == "__main__":
    # match subject name
    subject_pattern = re.compile(r'^S\d{3}$')
    # match task name
    task_pattern = re.compile(r'^S\d{3}R\d{2}\.edf$')
    # match task id
    task_id_pattern = re.compile(r'R\d{2}')
    event_list = []
    if not os.path.exists('../data'):
        os.makedirs('../data')
    # Iterate over subject directory for raw data
    for subject_path in os.listdir('../files'):
        if re.match(subject_pattern, subject_path):
            subject_id = int(subject_path.strip('S'))
            subject_dir = f"../data/{subject_id:03d}"
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)
            # Iterate over task directory for each subject
            for task_path in os.listdir(os.path.join('../files', subject_path)):
                if re.match(task_pattern, task_path):
                    path = os.path.join('../files', subject_path, task_path)
                    matches = re.search(task_id_pattern, task_path)
                    task_id = int(matches[0].strip('R'))
                    task_dir = f"../data/{subject_id:03d}/{task_id:02d}"
                    if not os.path.exists(task_dir):
                        os.makedirs(task_dir)
                    slice_task(path, subject_id, task_id)
            