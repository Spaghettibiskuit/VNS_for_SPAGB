"""Randomly generate a dataframe with relevant information on offered projects"""

import random as rd

import pandas as pd

import disciplines


def generate_projects_df(
    num_projects: int,
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
) -> pd.DataFrame:
    """Randomly generate a dataframe describing requirements of projects."""

    names_projects = rd.sample(
        disciplines.BUSINESS_DISCIPLINES, k=num_projects
    )
    desired_nums_groups = [
        rd.randint(min_desired_num_groups, max_desired_num_groups)
        for _ in names_projects
    ]
    max_nums_groups = [
        desired_num_groups
        + rd.randint(
            min_manageable_surplus_groups, max_manageable_surplus_groups
        )
        for desired_num_groups in desired_nums_groups
    ]
    ideal_group_sizes = [
        rd.randint(min_ideal_group_size, max_ideal_group_size)
        for _ in names_projects
    ]
    min_group_sizes = [
        max(
            1,
            ideal_group_size
            - rd.randint(
                min_tolerable_group_size_deficit,
                max_tolerable_group_size_deficit,
            ),
        )
        for ideal_group_size in ideal_group_sizes
    ]
    max_group_sizes = [
        ideal_group_size
        + rd.randint(
            min_tolerable_group_size_surplus, max_tolerable_group_size_surplus
        )
        for ideal_group_size in ideal_group_sizes
    ]

    data_projects = {
        "name": names_projects,
        "desired#groups": desired_nums_groups,
        "max#groups": max_nums_groups,
        "ideal_group_size": ideal_group_sizes,
        "min_group_size": min_group_sizes,
        "max_group_size": max_group_sizes,
    }

    return pd.DataFrame(data_projects)


# test_df = generate_projects_df(10)
# print(test_df)
