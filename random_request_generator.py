import json
import random
from pprint import pprint

rand_data_filepath= fr"request_post_data.json"
# Read JSON file
with open(rand_data_filepath) as file:
    data = json.load(file)

# Store JSON data in a dictionary
data_dict = dict(data).get('data')

# Print the dictionary
pprint(data_dict)

for key, val in data_dict.items():
    data_dict[key] = round(random.random())

# Write dictionary to a JSON file
with open(rand_data_filepath, 'w') as file:
    json.dump(data, file, indent=4)