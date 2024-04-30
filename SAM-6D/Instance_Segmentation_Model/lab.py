import json

def claculate_time(file_path):
    # Read data from file
    
    with open(file_path, 'r') as file:
        data = file.read()

    # Parse JSON data
    json_data = json.loads(data)

    # Initialize variables
    total_time = 0

    # Iterate over each item and sum the times
    for item in json_data:
        total_time += item['time']

    # Calculate the average
    if len(json_data) > 0:
        average_time = total_time / len(json_data)
    else:
        average_time = 0

    print("Total time:", total_time)
    print("Average time:", average_time)


claculate_time("log/fastsam_with_template/result_tless.json")