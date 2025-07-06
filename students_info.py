"""Contains functions to generate a dataframe on random students."""

import itertools
import random as rd

import pandas as pd

import common_names


def random_unique_names(num_students: int) -> list[str]:
    """Returns unique random full names for males and females.

    Args:
        num_students: The number of students in the problem.

    Raises:
        ValueError: GIRLS_NAMES and BOYS_NAMES have different lengths.
    """
    if (num_first_names := len(common_names.GIRLS_NAMES)) != len(common_names.BOYS_NAMES):
        raise ValueError("GIRLS_NAMES and BOYS_NAMES have different lengths.")
    num_genders_considered = 2

    return [
        (
            f"{common_names.GIRLS_NAMES[index_first_name]} {common_names.LAST_NAMES[index_last_name]}"
            if not male_indicator
            else f"{common_names.BOYS_NAMES[index_first_name]} {common_names.LAST_NAMES[index_last_name]}"
        )
        for male_indicator, index_first_name, index_last_name in rd.sample(
            list(
                itertools.product(
                    range(num_genders_considered),
                    range(num_first_names),
                    range(len(common_names.LAST_NAMES)),
                )
            ),
            num_students,
        )
    ]


def random_partner_preferences(
    num_students: int,
    percentage_reciprocity: float,
    num_partner_preferences: int,
) -> list[list[int]]:
    """Returns partner preferences for all students.

    Args:
        num_students: The number of students in the problem.
        percentage_reciprocity: Roughly the probability that a student
            specifies another student as a partner preference if that
            student specified him/her as a partner preference before.
        num_partner_preferences: The number of partner preferences
            each student specifies with the ID of the students he/she
            wants to  work together with the most.
    """
    students_partner_preferences: list[list[int]] = []
    chosen_by: dict[int : list[int]] = {}
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
            num_missing_preferences = num_partner_preferences - len(reciprocal_preferences)
            if num_missing_preferences > 0:
                left_options = list(all_other_student_ids.difference(set(reciprocal_preferences)))
                student_partner_preferences = reciprocal_preferences + rd.sample(left_options, num_missing_preferences)
            else:
                student_partner_preferences = reciprocal_preferences
        else:
            student_partner_preferences = rd.sample(list(all_other_student_ids), num_partner_preferences)

        students_partner_preferences.append(student_partner_preferences)

        for partner_preference in student_partner_preferences:
            if partner_preference > student_id:
                if partner_preference not in chosen_by:
                    chosen_by[partner_preference] = [student_id]
                else:
                    chosen_by[partner_preference].append(student_id)

    return students_partner_preferences


def average_preferences(desired_partners: list[int], project_preferences_so_far: list[tuple[int]]) -> dict[int:int]:
    """Returns the average available project preferences among a student's partner preferences.

    Available means that the partner preference i.e., one of the students the student in
    question wants to work with the most has specified his/her project preferences before.

    Args:
        desired_partners: The students the student in question wants to work with the most.
        project_preferences_so_far: The project preferences made so far. The index position is
            the ID of the student who specified the project preferences.
    """
    sums_preferences = {}
    num_students_with_preferences = len(project_preferences_so_far)
    desired_partners_with_preferences = [
        desired_partner for desired_partner in desired_partners if desired_partner < num_students_with_preferences
    ]
    for desired_partner in desired_partners_with_preferences:
        for project_id, preference in enumerate(project_preferences_so_far[desired_partner]):
            if project_id not in sums_preferences:
                sums_preferences[project_id] = preference
            else:
                sums_preferences[project_id] += preference

    return {
        project_id: sum_preferences / len(desired_partners_with_preferences)
        for project_id, sum_preferences in sums_preferences.items()
    }


def random_project_preferences(
    num_projects: int,
    students_desired_partners: list[list[int]],
    percentage_project_preference_overlap: float,
    min_project_preference: int,
    max_project_preference: int,
) -> list[tuple[int]]:
    """Returns project preference values for all students in the problem.

    Args:
        num_projects: The number of projects in the problem.
        students_desired_partners: The IDs of the students a student wants
            to work with the most for every student in the problem
        percentage_project_preference_overlap: To what degree the
            student's preference value for a specific project is the
            average preference for that project among those that are
            partner preferences and already have specified their
            project preferences.
        min_project_preference: The lowest possible project preference.
        max_project_preference: The highest possible project preference.
    """
    students_project_preferences: list[tuple[int]] = []
    for student_desired_partners in students_desired_partners:
        average_project_preferences_desired_partners = average_preferences(
            student_desired_partners, students_project_preferences
        )
        if average_project_preferences_desired_partners:
            student_project_preferences = tuple(
                (
                    round(
                        (
                            percentage_project_preference_overlap
                            * average_project_preferences_desired_partners[project_id]
                        )
                        + (
                            (1 - percentage_project_preference_overlap)
                            * rd.uniform(min_project_preference - 0.5, max_project_preference + 0.5)
                        )
                    )
                )
                for project_id in range(num_projects)
            )
        else:
            student_project_preferences = tuple(
                round(rd.uniform(min_project_preference - 0.5, max_project_preference + 0.5))
                for _ in range(num_projects)
            )
        students_project_preferences.append(student_project_preferences)

    return students_project_preferences


def random_students_df(
    num_projects: int,
    num_students: int,
    num_partner_preferences: int = 3,
    percentage_reciprocity: float = 0.7,
    percentage_project_preference_overlap: float = 0.7,
    min_project_preference: int = 0,
    max_project_preference: int = 3,
) -> pd.DataFrame:
    """Returns random students with partner and project preferences.

    Args:
        num_projects: The number of projects in the problem instance.
        num_students: The number of students in the problem instance.
        num_partner_preferences: The number of partner preferences
            each student specifies with the ID of the students he/she
            wants to  work together with the most.
        percentage_reciprocity: Roughly the probability that a student
            specifies another student as a partner preference if that
            student specified him/her as a partner preference before.
        percentage_project_preference_overlap: To what degree the
            student's preference value for a specific project is the
            average preference for that project among those that are
            partner preferences and already have specified their
            project preferences.
        min_project_preference: The lowest possible project preference.
        max_project_preference: The highest possible project preference.

    Returns:
        The project preferences for all projects and the partner preferences
        i.e., the students a student wants to work with the most, for all
        students in the problem instance. Project preferences and partner preferences
        are influenced by each other to a specifiable degree. Otherwise values
        are random within bounds set by the arguments. THE INDEX POSITION IN THE
        DATAFRAME LATER BECOMES THE STUDENT'S ID.
    """
    students_names = random_unique_names(num_students)
    desired_partners = random_partner_preferences(num_students, percentage_reciprocity, num_partner_preferences)
    desired_projects = random_project_preferences(
        num_projects,
        desired_partners,
        percentage_project_preference_overlap,
        min_project_preference,
        max_project_preference,
    )
    data_students = {
        "name": students_names,
        "fav_partners": desired_partners,
        "project_prefs": desired_projects,
    }
    return pd.DataFrame(data_students)
