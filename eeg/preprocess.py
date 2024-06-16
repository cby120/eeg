import os
import re
import mne
import numpy as np


def get_label(task_id, event_id):
    if event_id in (0, 1, 2):
        return 0
    elif task_id == (3, 4, 7, 8, 11, 12):
        if event_id == 1:
            return 1 # left fist
        else:
            return 2 # right fist
    else:
        if event_id == 1:
            return 3 # both fist
        else:
            return 4 # both feet


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
        np.savez_compressed('../data/' + str(subject_id).zfill(3) + '/' + str(task_id).zfill(2) + '/' + str(label_perT) + '_' + str(i).zfill(2) + '.npz', data=data_array)


if __name__ == "__main__":
    subject_pattern = re.compile(r'^S\d{3}$')
    task_pattern = re.compile(r'^S\d{3}R\d{2}\.edf$')
    task_id_pattern = re.compile(r'R\d{2}')
    event_list = []
    if not os.path.exists('../data'):
        os.makedirs('../data')
    for subject_path in os.listdir('../files'):
        if re.match(subject_pattern, subject_path):
            subject_id = int(subject_path.strip('S'))
            subject_dir = '../data/' + str(subject_id).zfill(3)
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)
            for task_path in os.listdir(os.path.join('../files', subject_path)):
                if re.match(task_pattern, task_path):
                    path = os.path.join('../files', subject_path, task_path)
                    matches = re.search(task_id_pattern, task_path)
                    task_id = int(matches[0].strip('R'))
                    task_dir = '../data/' + str(subject_id).zfill(3) + '/' + str(task_id).zfill(2)
                    if not os.path.exists(task_dir):
                        os.makedirs(task_dir)
                    slice_task(path, subject_id, task_id)
            