import pickle
from pathlib import Path

import problem_data

folder = Path("instances")

if not folder.exists():
    folder.mkdir()

# H

project_quantities = [3, 4, 5]
student_quantities = [30, 40, 50]

for project_quantity in project_quantities:
    for student_quantity in student_quantities:
        filename = f"generic_{project_quantity}_{student_quantity}.pkl"
        instance_path = folder / filename
        problem_data.save_projects_and_students_instance(
            num_projects=project_quantity, num_students=student_quantity, instance_path=instance_path
        )

        with instance_path.open("rb") as f:
            created_instance = pickle.load(f)

        print(created_instance[0])
        print(created_instance[1])
