"""Contains a function that demonstrates VNS on a specified or random instance."""

import json
from pathlib import Path

import pandas as pd

from problem_data import generate_throwaway_instance
from settings import DemonstrationSettings
from vns_on_student_assignment import VariableNeighborhoodSearch


def report_current_solution(vns_instance: VariableNeighborhoodSearch):
    """Prints the incumbent solution of VNS."""
    for project in vns_instance.projects:
        print(f"\nThese are the groups in the project {project.name}")
        for group in project.groups:
            print("\n")
            for student in group.students:
                print(student.name, student.student_id)
    if vns_instance.unassigned_students:
        print("\nThese students were not assigned:")
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
    """Demonstrates VNS on a specific or random instance.

    First it prints the problem data and the initial solution.
    During optimization with VNS it prints out updates on the
    solution process every iteration in the loop of
    run_general_vns_best_improvement of VariableNeighborhoodSearch.
    At the end it prints the final solution.

    Args:
        num_projects: The number of projects in the problem.
        num_students: The number of students in the problem.
        instance_number: Specifies the exact instance among
            the instances with matching num_projects and
            num_students. Only necessary when a specific
            instance should be solved.
        seed_instance_generation: The random seed with which
            a random instance will be generated.
        seed_vns_run: random seed passed to
            run_general_vns_best_improvement method of
            VariableNeighborhoodSearch.
        iterations: number of iterations in main loop of
            run_general_vns_best_improvement of
            VariableNeighborhoodSearch before optimization
            stops and final results are printed.
    """
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
