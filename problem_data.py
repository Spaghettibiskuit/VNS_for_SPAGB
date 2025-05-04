"""Prevents crash because num_projects do not match by enforcing equality."""

import pickle
from functools import partial
from pathlib import Path

import pandas as pd

from projects_info import random_projects_df
from students_info import random_students_df


def generate_projects_and_students_data(
    num_projects: int,
    num_students: int,
    save_as_pickle: bool,
    instance_path: Path | None,
    # specifications for project data
    min_desired_num_groups: int = 2,
    max_desired_num_groups: int = 4,
    min_manageable_surplus_groups: int = 1,
    max_manageable_surplus_groups: int = 3,
    min_ideal_group_size: int = 2,
    max_ideal_group_size: int = 4,
    min_tolerable_group_size_deficit: int = 1,
    max_tolerable_group_size_deficit: int = 3,
    min_tolerable_group_size_surplus: int = 1,
    max_tolerable_group_size_surplus: int = 3,
    min_pen_num_groups: int = 1,
    max_pen_num_groups: int = 3,
    min_pen_group_size: int = 1,
    max_pen_group_size: int = 3,
    # specifications for student data
    num_partner_preferences: int = 3,
    percentage_reciprocity: float = 0.7,
    perc_proj_pref_overlap: float = 0.7,
    min_project_preference: int = 0,
    max_project_preference: int = 3,
) -> tuple[pd.DataFrame]:
    """Return random projects and students data in one tuple."""
    instance_tuple = tuple(
        (
            random_projects_df(
                num_projects=num_projects,
                min_desired_num_groups=min_desired_num_groups,
                max_desired_num_groups=max_desired_num_groups,
                min_manageable_surplus_groups=min_manageable_surplus_groups,
                max_manageable_surplus_groups=max_manageable_surplus_groups,
                min_ideal_group_size=min_ideal_group_size,
                max_ideal_group_size=max_ideal_group_size,
                min_tolerable_group_size_deficit=min_tolerable_group_size_deficit,
                max_tolerable_group_size_deficit=max_tolerable_group_size_deficit,
                min_tolerable_group_size_surplus=min_tolerable_group_size_surplus,
                max_tolerable_group_size_surplus=max_tolerable_group_size_surplus,
                min_pen_num_groups=min_pen_num_groups,
                max_pen_num_groups=max_pen_num_groups,
                min_pen_group_size=min_pen_group_size,
                max_pen_group_size=max_pen_group_size,
            ),
            random_students_df(
                num_projects=num_projects,
                num_students=num_students,
                num_partner_preferences=num_partner_preferences,
                percentage_reciprocity=percentage_reciprocity,
                perc_proj_pref_overlap=perc_proj_pref_overlap,
                min_project_preference=min_project_preference,
                max_project_preference=max_project_preference,
            ),
        )
    )
    if save_as_pickle:
        with instance_path.open("wb") as f:
            pickle.dump(instance_tuple, f)
    else:
        return instance_tuple


save_projects_and_students_instance = partial(
    generate_projects_and_students_data, save_as_pickle=True
)

generate_throwaway_instance = partial(
    generate_projects_and_students_data,
    save_as_pickle=False,
    instance_path=None,
)
