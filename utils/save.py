import os
import pandas as pd
import os
import argparse

def save_to_dataframe(acc,args):
  # path to the dataframe csv file
  path = args.save_dir + '/' + args.save_name
  # new dataframe for appending (accuracy tensor in mean)
  new_data = {
      'Dataset': [args.dataset],
      'Shots': [args.shot],
      'Normalization': [args.normalize],
      'Result': [acc.mean()]
  }

  new_df = pd.DataFrame(new_data)

  # append the new dataframe to the existing csv
  new_df.to_csv(path, mode='a', header=False, index=False)

def save_to_file_directory(acc,args):
  data_dir= args.save_dir
  dataset_name= args.dataset
  shot_dir = args.shot
  normalize_data=args.normalize

  path = os.path.join(data_dir, dataset_name, str(shot_dir) + '-shot')
  os.makedirs(path,exist_ok=True)

  with open(os.path.join(path, 'results_sam_proto_neg.txt'), 'w') as f:
    f.write(dataset_name + '\n')
    f.write(str(shot_dir) + '\n')
    f.write(str(normalize_data) + '\n')
    f.write(str(acc) + '\n')
    f.write(str(acc.mean()))
