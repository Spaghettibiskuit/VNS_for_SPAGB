import pickle
from pathlib import Path

import problem_data

folder = Path("instances")

if not folder.exists():
    folder.mkdir()

filename = "generic_3_30.pkl"

instance_path = folder / filename


problem_data.save_projects_and_students_instance(
    num_projects=3, num_students=30, instance_path=instance_path
)

with instance_path.open("rb") as f:
    created_instance = pickle.load(f)

print(created_instance[0])
print(created_instance[1])
