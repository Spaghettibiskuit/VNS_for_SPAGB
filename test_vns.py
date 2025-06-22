from pathlib import Path

from problem_data import generate_throwaway_instance
from vns_on_student_assignment import VariableNeighborhoodSearch


def test_vns(
    min_num_projects,
    max_num_projects,
    step_num_projects,
    min_num_students,
    max_num_students,
    step_num_students,
    iteration_limit,
    max_neighborhood,
    starting_random_seed,
    line_limit,
    filename="most_recent_error_log",
    extension=".txt",
):
    folder = Path("error_logs")
    if not folder.exists():
        folder.mkdir()
    error_log_path = folder / (filename + extension)
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
                    max_neighborhood=max_neighborhood,
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
    test_vns(
        min_num_projects=3,
        max_num_projects=6,
        step_num_projects=1,
        min_num_students=20,
        max_num_students=60,
        step_num_students=5,
        iteration_limit=100,
        max_neighborhood=6,
        starting_random_seed=0,
        line_limit=1000,
        filename="error_log_18",
        extension=".txt",
    )
