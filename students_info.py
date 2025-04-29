import itertools
import random as rd

import pandas as pd

import common_names


def indexes_to_names(
    tuples_of_indexes: list[tuple[int, int, int]],
) -> list[str]:
    """Return list of full names specified by specific values in tuples."""
    indicated_names = []
    for tuple_of_indexes in tuples_of_indexes:
        sex_indicator, index_first_name, index_last_name = tuple_of_indexes
        if sex_indicator == 0:
            first_name = common_names.GIRLS_NAMES[index_first_name]
        else:
            first_name = common_names.BOYS_NAMES[index_first_name]
        last_name = common_names.LAST_NAMES[index_last_name]
        indicated_names.append(f"{first_name} {last_name}")

    return indicated_names


def random_unique_names(num_students: int) -> list[str]:
    """Generate unique names combining girs and boys names with last names."""
    if (num_first_names := len(common_names.GIRLS_NAMES)) != len(
        common_names.BOYS_NAMES
    ):
        raise ValueError("GIRLS_NAMES and BOYS_NAMES have different length.")
    num_sexes_considered = 2
    all_combos_indexes = list(
        itertools.product(
            range(num_sexes_considered),
            range(num_first_names),
            range(len(common_names.LAST_NAMES)),
        )
    )
    selected_combos_indexes = rd.sample(all_combos_indexes, num_students)

    return indexes_to_names(selected_combos_indexes)


def random_partner_preferences(
    num_students: int,
    percentage_reciprocity: float,
    num_partner_preferences: int,
) -> list[list[int]]:
    """Generates a list of partner preferences which show a specified degree of reciprocity."""
    students_partner_preferences: list[list[int]] = []
    chosen_by: dict[list[int]] = {}
    student_ids = set(range(num_students))
    for student_id in range(num_students):
        all_other_student_ids = student_ids.difference({student_id})
        if (id_chosen_by := chosen_by.get(student_id)) is not None:
            applicable_for_reciprocity = rd.sample(
                id_chosen_by,
                min(len(id_chosen_by), num_partner_preferences),
            )
            reciprocal_preferences = [
                other_students_id
                for other_students_id in applicable_for_reciprocity
                if rd.random() <= percentage_reciprocity
            ]
            if reciprocal_preferences:
                student_partner_preferences = reciprocal_preferences
            else:
                student_partner_preferences = []
            num_missing_preferences = num_partner_preferences - len(
                student_partner_preferences
            )
            if num_missing_preferences > 0:
                left_options = list(
                    all_other_student_ids.difference(
                        set(student_partner_preferences)
                    )
                )
                student_partner_preferences += rd.sample(
                    left_options, num_missing_preferences
                )

        else:
            student_partner_preferences = rd.sample(
                list(all_other_student_ids), num_partner_preferences
            )

        students_partner_preferences.append(student_partner_preferences)

        for partner_preference in student_partner_preferences:
            if partner_preference > student_id:
                if partner_preference not in chosen_by:
                    chosen_by[partner_preference] = [student_id]
                else:
                    chosen_by[partner_preference].append(student_id)

    return students_partner_preferences


def _create_prefs_dict(
    desired_partners: list[int], project_preferences_so_far: list[list[int]]
) -> dict[int]:
    sums_preferences = {"num_with_preferences": 0}
    num_students_with_preferences = len(project_preferences_so_far)
    for desired_partner in desired_partners:
        if desired_partner < num_students_with_preferences:
            for project, preference in enumerate(
                project_preferences_so_far[desired_partner]
            ):
                if project not in sums_preferences:
                    sums_preferences[project] = preference
                else:
                    sums_preferences[project] += preference
            sums_preferences["num_with_preferences"] += 1

    return sums_preferences


def random_project_preferences(
    students_desired_partners,
    num_projects,
    perc_proj_pref_overlap,
    min_project_preference,
    max_project_preference,
) -> list[tuple[int]]:
    """Project preference values based on chance and preferences made by desired partners."""
    students_project_preferences: list[tuple[int]] = []
    for student_desired_partners in students_desired_partners:
        sums_preferences = _create_prefs_dict(
            student_desired_partners, students_project_preferences
        )
        if (
            num_already_decided := sums_preferences["num_with_preferences"]
        ) > 0:
            student_project_preferences = tuple(
                (
                    round(
                        (
                            perc_proj_pref_overlap
                            * sums_preferences[project_id]
                            / num_already_decided
                        )
                        + ((1 - perc_proj_pref_overlap) * rd.uniform(0, 3))
                    )
                    if project_id in sums_preferences
                    else round(
                        rd.uniform(
                            min_project_preference, max_project_preference
                        )
                    )
                )
                for project_id in range(num_projects)
            )
        else:
            student_project_preferences = tuple(
                round(
                    rd.uniform(min_project_preference, max_project_preference)
                )
                for _ in range(num_projects)
            )
        students_project_preferences.append(student_project_preferences)

    return students_project_preferences


def create_random_students_df(
    num_projects: int,
    num_students: int,
    num_partner_preferences: int = 3,
    percentage_reciprocity: float = 0.7,
    perc_proj_pref_overlap: float = 0.7,
    min_project_preference: int = 0,
    max_project_preference: int = 3,
) -> pd.DataFrame:
    """Return relevant information on students."""
    students_names = random_unique_names(num_students)
    desired_partners = random_partner_preferences(
        num_students, percentage_reciprocity, num_partner_preferences
    )
    desired_projects = random_project_preferences(
        desired_partners,
        num_projects,
        perc_proj_pref_overlap,
        min_project_preference,
        max_project_preference,
    )
    data_students = {
        "name": students_names,
        "fav_partners": desired_partners,
        "project_prefs": desired_projects,
    }
    return pd.DataFrame(data_students)


# print(
#     create_random_students_df(
#         num_projects=5,
#         num_students=40,
#         percentage_reciprocity=1,
#         perc_proj_pref_overlap=1,
#         num_partner_preferences=1,
#     )
# )
