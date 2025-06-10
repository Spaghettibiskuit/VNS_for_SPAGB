import random
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
        random.seed(current_random_seed)
        for num_projects in range(min_num_projects, max_num_projects + 1, step_num_projects):
            for num_students in range(min_num_students, max_num_students + 1, step_num_students):
                problem_instance = generate_throwaway_instance(num_projects=num_projects, num_students=num_students)
                vns_run = VariableNeighborhoodSearch(
                    problem_instance,
                    reward_bilateral_interest_collaboration=2,
                    penalty_student_not_assigned=3,
                )
                error_report = vns_run.run_general_vns_best_improvement(
                    iteration_limit=40,
                    max_neighborhood=6,
                    assignment_bias=10,
                    unassignment_probability=0.05,
                    testing=True,
                )
                if not error_report:
                    continue
                error_description_elements = [
                    f"Seed: {current_random_seed}; Number of Projects: {num_projects}; "
                    f"Number of Students: {num_students}; Iteration: {error_report["iteration"]}; "
                    f"Point: {error_report["point"]}"
                ]
                if "claimed_obj" in error_report:
                    error_description_elements.append(
                        f"The objective value calculated with deltas is: {error_report["claimed_obj"]}; "
                        f"The actual objective value obtained by complete recalculation is {error_report["actual_obj"]}"
                    )
                if "groups_too_small" in error_report:
                    error_description_elements += [
                        group_to_small_description for group_to_small_description in error_report["groups_too_small"]
                    ]
                if "groups_too_big" in error_report:
                    error_description_elements += [
                        group_to_big_description for group_to_big_description in error_report["groups_too_big"]
                    ]
                if "too_many_groups" in error_report:
                    error_description_elements += [
                        too_many_groups_description for too_many_groups_description in error_report["too_many_groups"]
                    ]
                if "inconsistency_students" in error_report:
                    error_description_elements.append("There is a inconsistency in how students are distributed;\n")

                with error_log_path.open("a", encoding="utf-8") as f:
                    f.write("; ".join(error_description_elements))
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
        starting_random_seed=0,
        line_limit=1000,
        filename="error_log_5",
        extension=".txt",
    )
