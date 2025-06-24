import json
from pathlib import Path

import pandas as pd

from problem_data import generate_throwaway_instance
from settings import DemonstrationSettings
from vns_on_student_assignment import VariableNeighborhoodSearch


def report_current_solution(vns_instance: VariableNeighborhoodSearch):
    for project in vns_instance.projects:
        print(f"\nThese are the groups in the project {project.name}")
        for group in project.groups:
            print("\n")
            for student in group.students:
                print(student.name, student.student_id)
    if vns_instance.unassigned_students:
        print("\nThese students were not assigned:")
        print(vns_instance.unassigned_students)
        for student in vns_instance.unassigned_students:
            print(student.name, student.student_id)
    else:
        print("\nAll Students were assigned.")
    print(f"The objective value is: {vns_instance.objective_value}")


def demonstrate_vns(
    num_projects: int,
    num_students: int,
    instance_number: int | None,
    seed_instance_generation: int | None,
    seed_vns_run: int | None,
    iterations: int,
):
    if instance_number != None:
        if not 0 <= instance_number <= 9:
            raise ValueError("Only instance numbers from 0 to 9 are valid!")
        dimension = f"{num_projects}_{num_students}"
        dimension_subfolder = f"{dimension}_instances"
        folder_projects = Path("instances_projects")
        filename_projects = f"generic_{dimension}_projects_{instance_number}.csv"
        filepath_projects = folder_projects / dimension_subfolder / filename_projects
        folder_students = Path("instances_students")
        filename_students = f"generic_{dimension}_students_{instance_number}.csv"
        filepath_students = folder_students / dimension_subfolder / filename_students
        projects_df = pd.read_csv(filepath_projects)
        students_df = pd.read_csv(filepath_students)
        students_df["fav_partners"] = students_df["fav_partners"].apply(json.loads)
        students_df["project_prefs"] = students_df["project_prefs"].apply(lambda x: tuple(json.loads(x)))
    else:
        projects_df, students_df = generate_throwaway_instance(
            num_projects=num_projects, num_students=num_students, seed=seed_instance_generation
        )
    pd.set_option("display.max_columns", None)
    print(projects_df)
    print(students_df)
    vns_run = VariableNeighborhoodSearch(
        projects_df,
        students_df,
    )
    print("\nTHIS IS THE INITIAL SOLUTION:")
    report_current_solution(vns_run)
    vns_run.run_general_vns_best_improvement(seed=seed_vns_run, iteration_limit=iterations, demonstrating=True)
    print("\nTHIS IS THE FINAL SOLUTION:")
    report_current_solution(vns_run)


if __name__ == "__main__":
    settings = DemonstrationSettings()
    demonstrate_vns(
        settings.num_projects,
        settings.num_students,
        settings.instance_number,
        settings.seed_instance_generation,
        settings.seed_vns_run,
        settings.iterations,
    )
