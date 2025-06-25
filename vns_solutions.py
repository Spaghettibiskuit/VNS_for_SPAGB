import json
import time as t
from pathlib import Path

import pandas as pd

from settings import BenchmarkSettingsVNS
from vns_on_student_assignment import VariableNeighborhoodSearch


def benchmark_vns(
    filename_results: str,
    filename_error_logs: str,
    project_quantities: list[int],
    student_quantities: list[int],
    instances_per_dimension: int = 10,
    time_limit: int = 300,
    seed: int = 100,
):
    results_path = Path(filename_results)
    if results_path.is_file():
        raise ValueError(f"{results_path} already exists!")
    folder_error_logs = Path("error_logs")
    folder_error_logs.mkdir(exist_ok=True)
    error_log_path = folder_error_logs / filename_error_logs
    if error_log_path.is_file():
        raise ValueError(f"{error_log_path} already exists!")

    vns_solution_developments = {}
    folder_projects = Path("instances_projects")
    folder_students = Path("instances_students")

    for project_quantity in project_quantities:
        for student_quantity in student_quantities:
            dimension = f"{project_quantity}_{student_quantity}"
            dimension_subfolder = f"{dimension}_instances"
            for instance_number in range(instances_per_dimension):
                start = t.time()
                filename_projects = f"generic_{dimension}_projects_{instance_number}.csv"
                filename_students = f"generic_{dimension}_students_{instance_number}.csv"
                filepath_projects = folder_projects / dimension_subfolder / filename_projects
                filepath_students = folder_students / dimension_subfolder / filename_students
                projects_info = pd.read_csv(filepath_projects)
                students_info = pd.read_csv(filepath_students)
                students_info["fav_partners"] = students_info["fav_partners"].apply(json.loads)
                students_info["project_prefs"] = students_info["project_prefs"].apply(lambda x: tuple(json.loads(x)))
                vns = VariableNeighborhoodSearch(projects_info, students_info)
                vns_solution_developments[f"generic_{dimension}_{instance_number}"] = (
                    vns.run_general_vns_best_improvement(benchmarking=True, time_limit=time_limit, seed=seed)
                )
                print(f"Done with instance {dimension}_{instance_number}!")
                print(f"It took {t.time() - start} seconds")

                if not (error_report := vns.check_solution()):
                    results_path.write_text(json.dumps(vns_solution_developments, indent=4), encoding="utf-8")
                    continue
                error_description_elements = [f"Seed: {seed}; Instance: {dimension}_{instance_number}"]
                if "claimed_obj" in error_report:
                    error_description_elements.append(
                        f"The objective value calculated with deltas is: {error_report["claimed_obj"]}; "
                        f"The actual objective value obtained by complete recalculation is {error_report["actual_obj"]}"
                    )
                for issue in ["groups_too_small", "groups_too_big", "too_many_groups"]:
                    if issue in error_report:
                        error_description_elements += list(error_report[issue])
                if "inconsistency_students" in error_report:
                    error_description_elements.append("There is a inconsistency in how students are distributed;")
                with error_log_path.open("a", encoding="utf-8") as f:
                    f.write("; ".join(error_description_elements) + "\n")

            print(f"Done with student quantitiy {student_quantity}!")
        print(f"Done with project quantity {project_quantity}!")


if __name__ == "__main__":
    settings = BenchmarkSettingsVNS()
    benchmark_vns(
        filename_results=settings.filename_results,
        filename_error_logs=settings.filename_error_logs,
        project_quantities=settings.project_quantities,
        student_quantities=settings.student_quantities,
        instances_per_dimension=settings.instances_per_dimension,
        time_limit=settings.time_limit,
        seed=settings.seed,
    )
