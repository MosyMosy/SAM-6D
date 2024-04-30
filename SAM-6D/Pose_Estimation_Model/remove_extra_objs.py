
# read a commo delimatted csv file and remove the extra rows from the file
# and write the new file to a new file
# extra rows are the rows that the third column is not 5

import csv
import os
import csv
import json

def remove_extra_objs(input_file, output_file):
    with open(input_file, 'r') as in_file:
        with open(output_file, 'w') as out_file:
            writer = csv.writer(out_file)
            for row in csv.reader(in_file):
                if row[2] == '5':
                    writer.writerow(row) 
                    

                    




def keep_highest_score(input_file, output_file):
    rows = []
    with open(input_file, 'r') as in_file:
        reader = csv.reader(in_file)
        header = next(reader)
        rows.append(header)
        for row in reader:
            rows.append(row)
    
    unique_rows = []
    seen = set()
    for row in rows:
        key = tuple(row[:3])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
        else:
            index = next(i for i, r in enumerate(unique_rows) if tuple(r[:3]) == key)
            if float(row[3]) > float(unique_rows[index][3]):
                unique_rows[index] = row
    
    with open(output_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(unique_rows)
         

def convert_csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # header = next(reader)  # Skip the header row
        for row in reader:
            scene_id = int(row[0])
            im_id = int(row[1])
            obj_id = int(row[2])
            inst_count = 1
            data.append({
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": obj_id,
                "inst_count": inst_count
            })

    with open(json_file, 'w') as file:
        json.dump(data, file)        
         
         

         
input_file = 'log/pose_estimation_model_base_id_fastsam/tless_eval_iter000000/result_tless.csv' 
output_file = 'log/fastsam.csv'

remove_extra_objs(input_file, output_file)     
keep_highest_score('log/fastsam.csv', 'log/fastsam_highest_score.csv')
convert_csv_to_json('log/fastsam_highest_score.csv', "log/fastsam_highest_score.json")


