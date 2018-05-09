from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os


# Converts AffectNet to folder structure for Keras ImageDataGenerator
file = r'preprocessed.csv'
df = pd.read_csv(file)

counter = [0, 0, 0, 0, 0, 0]
save = [18486, 4747, 100571, 19117, 10528, 56106]

counter_val = [0, 0, 0, 0, 0, 0]
save_val = [18486, 4747, 100571, 19117, 10528, 56106]


for i, label in enumerate(df['label']):

    path_old = "Manually_Annotated_Images/" + df['subDirectory_filePath'][i]
    filename, file_extension = os.path.splitext(path_old)
    if i % 70 is 0:
        path_new = "images_val_full/" + str(label) + "/image" + str("%07d" % counter[label]) + file_extension
        counter_val[label] = counter_val[label] + 1

    else:
        path_new = "images_train_full/" + str(label) + "/image" + str("%07d" % counter[label]) + file_extension
        counter[label] = counter[label] + 1

    try:
        shutil.move(path_old, path_new)
    except Exception:
        print('not found')
        # continue if file not found
