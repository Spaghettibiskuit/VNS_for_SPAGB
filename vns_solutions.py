import json

# import random as rd
from pathlib import Path

# rd.seed(100)
import pandas as pd

from vns_on_student_assignment import VariableNeighborhoodSearch

vns_solution_developments = {}
project_quantities = [3, 4, 5]
student_quantities = [30, 40, 50]
instances_per_combination = 10


folder_projects = Path("instances_projects")
folder_students = Path("instances_students")

for project_quantity in project_quantities[1:]:
    for student_quantity in student_quantities[:1]:
        dimension = f"{project_quantity}_{student_quantity}"
        dimension_subfolder = f"{dimension}_instances"
        for instance in range(instances_per_combination - 9):
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

with open("vns_benchmarks.json", "w", encoding="utf-8") as f:
    json.dump(vns_solution_developments, f, indent=4)
