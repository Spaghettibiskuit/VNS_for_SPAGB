import json
import time as t
from pathlib import Path

import pandas as pd

from vns_on_student_assignment import VariableNeighborhoodSearch

vns_solution_developments = {}
project_quantities = [3, 4, 5]
student_quantities = [30, 40, 50]
instances_per_combination = 10


folder_projects = Path("instances_projects")
folder_students = Path("instances_students")

for project_quantity in project_quantities:
    for student_quantity in student_quantities:
        dimension = f"{project_quantity}_{student_quantity}"
        dimension_subfolder = f"{dimension}_instances"
        for instance in range(instances_per_combination - 9):
            start = t.time()
            filename_projects = f"generic_{dimension}_projects_{instance}.csv"
            filename_students = f"generic_{dimension}_students_{instance}.csv"
            filepath_projects = folder_projects / dimension_subfolder / filename_projects
            filepath_students = folder_students / dimension_subfolder / filename_students
            projects_info = pd.read_csv(filepath_projects)
            students_info = pd.read_csv(filepath_students)
            students_info["fav_partners"] = students_info["fav_partners"].apply(json.loads)
            students_info["project_prefs"] = students_info["project_prefs"].apply(lambda x: tuple(json.loads(x)))
            vns = VariableNeighborhoodSearch(projects_info, students_info)
            vns_solution_developments[f"generic_{dimension}_{instance}"] = vns.run_general_vns_best_improvement(
                benchmarking=True, time_limit=60, max_neighborhood=6, seed=100
            )
            print(f"Done with instance {project_quantity}_{student_quantity}_{instance}!")
            print(f"It took {t.time() - start} seconds")
            if vns.check_solution():
                print("SOMETHING WENT WRONG!")
            # if vns.objective_value != (actual_objective_value := vns.current_objective_value()):
            #     print(
            #         f"THE OBJECTIVE WAS WRONGLY CALCULATED AS {vns.objective_value}. "
            #         f"ACTUAL OBJ: {actual_objective_value}"
            #     )
        print(f"Done with student quantitiy {student_quantity}!")
    print(f"Done with project quantity {project_quantity}!")

with open("vns_benchmarks_60s_every_dim_once_fixed.json", "w", encoding="utf-8") as f:
    json.dump(vns_solution_developments, f, indent=4)
