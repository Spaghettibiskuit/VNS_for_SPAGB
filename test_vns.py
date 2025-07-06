"""Contains the function for testing the correctness of the VNS solving."""

from pathlib import Path

from problem_data import generate_throwaway_instance
from settings import TestSettings
from vns_on_student_assignment import VariableNeighborhoodSearch


def test_vns(
    min_num_projects: int,
    max_num_projects: int,
    step_num_projects: int,
    min_num_students: int,
    max_num_students: int,
    step_num_students: int,
    iteration_limit: int,
    starting_random_seed: int,
    line_limit: int,
    filename: str,
) -> None:
    """Tests whether miscalculations or invalid solutions occur.

    Lets instances be generated in a reproducible way and lets them be solved by
    the run_general_vns_best_improvement method of VariableNeighborhoodSearch
    in vns_on_student_assignment in testing mode (testing=True). In
    testing mode it is checked after every founding or dissolving of a
    group, shake or VND whether the objective value is still correct and
    whether the solution is valid. If not, run_general_vns_best_improvement
    stops and returns an error report detailing what went wrong, as well as
    when and where it happened. The error report is then saved as one line of
    a designated text file. This way, any found bugs are reproducible and can be
    located more easily.

    Args:
        min_num_projects: The minimum number of projects.
        max_num_projects: The maximum number of projects.
        step_num_projects: The step in terms of numbers of projects
            as the number of projects per instance is increased.
        min_num_students: The minimum number of students.
        max_num_students: The maximum number of students.
        step_num_students: The step in terms of numbers of students
            as the number of students per instance is increased.
        iteration_limit: The number of iterations performed within
            the run_general_vns_best_improvement method of
            VariableNeighborhoodSearch per instance.
        starting_random_seed: The first random seed passed to the
            function which generates the instances as well as the
            run_general_vns_best_improvement method of
            VariableNeighborhoodSearch. After one instance for all
            possible combinations of numbers of students and numbers
            of projects defined by the respective arguments ran for the
            specified number of iterations, the random seed
            is increased.
        line_limit: The number of error reports in the designated text
            file after which the loop breaks. Each error report is on one line.
        filename: The name of the text file in which the error reports ought
            to be saved.
    """
    folder = Path("error_logs")
    folder.mkdir(exist_ok=True)
    error_log_path = folder / filename
    if error_log_path.is_file():
        raise ValueError(f"{error_log_path} already exists!")
    current_random_seed = starting_random_seed
    lines_written = 0
    while lines_written < line_limit:
        for num_projects in range(min_num_projects, max_num_projects + 1, step_num_projects):
            for num_students in range(min_num_students, max_num_students + 1, step_num_students):
                projects_info, students_info = generate_throwaway_instance(
                    num_projects=num_projects, num_students=num_students, seed=current_random_seed
                )
                vns_run = VariableNeighborhoodSearch(
                    projects_info,
                    students_info,
                )
                error_report = vns_run.run_general_vns_best_improvement(
                    iteration_limit=iteration_limit,
                    testing=True,
                    seed=current_random_seed,
                )
                if not error_report:
                    print(
                        f"Success for {num_projects} projects, {num_students} students and seed {current_random_seed}!"
                    )
                    continue
                error_description_elements = [
                    f"Seed: {current_random_seed}; Number of Projects: {num_projects}; "
                    f"Number of Students: {num_students}; Iteration: {error_report["iteration"]}; "
                    f"Point: {error_report["point"]}; Neighborhood: {error_report["neighborhood"]}"
                ]
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
                lines_written += 1
                if lines_written >= line_limit:
                    break
            if lines_written >= line_limit:
                break
        if lines_written >= line_limit:
            break
        with error_log_path.open("a", encoding="utf-8") as f:
            f.write(f"Finished testing for random seed {current_random_seed}\n")
        lines_written += 1
        current_random_seed += 1


if __name__ == "__main__":
    settings = TestSettings()
    test_vns(
        min_num_projects=settings.min_num_projects,
        max_num_projects=settings.max_num_projects,
        step_num_projects=settings.step_num_projects,
        min_num_students=settings.min_num_students,
        max_num_students=settings.max_num_students,
        step_num_students=settings.step_num_students,
        iteration_limit=settings.iteration_limit,
        starting_random_seed=settings.starting_random_seed,
        line_limit=settings.line_limit,
        filename=settings.filename,
    )
