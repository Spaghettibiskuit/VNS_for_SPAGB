import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


def get_gurobi_best_obj_60sec_and_obj_bound(
    projects,
    students,
    bilateral_reward,
    unassignment_penalty,
):
    m = gp.Model()

    project_ids = range(len(projects))
    student_ids = range(len(students))

    projects_group_ids = {project_id: range(projects["max#groups"][project_id]) for project_id in project_ids}

    x = m.addVars(
        (
            (project_id, group_id, student_id)
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
            for student_id in student_ids
        ),
        vtype=GRB.BINARY,
        name="assign",
    )
    m.update()

    y = m.addVars(
        ((project_id, group_id) for project_id in project_ids for group_id in projects_group_ids[project_id]),
        vtype=GRB.BINARY,
        name="establish_group",
    )

    favorite_partners_students = students["fav_partners"].tolist()
    mutual_pairs = set()

    for student_id, favorite_partners in enumerate(favorite_partners_students):
        for partner_id in favorite_partners:
            if partner_id <= student_id:
                continue
            if student_id in favorite_partners_students[partner_id]:
                mutual_pairs.add((student_id, partner_id))

    z = m.addVars(mutual_pairs, vtype=GRB.BINARY, name="not_realize_bilateral_cooperation_wish")

    v = m.addVars(student_ids, name="student_unassigned")

    gs_surplus = m.addVars(
        ((project_id, group_id) for project_id in project_ids for group_id in projects_group_ids[project_id]),
        name="group_size_surplus",
    )

    gs_deficit = m.addVars(
        ((project_id, group_id) for project_id in project_ids for group_id in projects_group_ids[project_id]),
        name="group_size_deficit",
    )

    project_preferences = {
        (student_id, project_id): students["project_prefs"][student_id][project_id]
        for student_id in student_ids
        for project_id in project_ids
    }

    rewards_student_project_preference = gp.quicksum(
        project_preferences[student_id, project_id] * x[project_id, group_id, student_id]
        for project_id in project_ids
        for group_id in projects_group_ids[project_id]
        for student_id in student_ids
    )

    rewards_bilateral_cooperation_wish_realized = bilateral_reward * gp.quicksum(
        1 - z[*mutual_pair] for mutual_pair in mutual_pairs
    )

    penalties_student_unassigned = unassignment_penalty * gp.quicksum(v.values())

    penalties_more_groups_than_offered = gp.quicksum(
        projects["pen_groups"][project_id] * y[project_id, group_id]
        for project_id in project_ids
        for group_id in projects_group_ids[project_id]
        if group_id >= projects["desired#groups"][project_id]
    )

    penalties_not_ideal_group_size = gp.quicksum(
        projects["pen_size"][project_id] * (gs_surplus[project_id, group_id] + gs_deficit[project_id, group_id])
        for project_id in project_ids
        for group_id in projects_group_ids[project_id]
    )

    m.setObjective(
        rewards_student_project_preference
        + rewards_bilateral_cooperation_wish_realized
        - penalties_student_unassigned
        - penalties_more_groups_than_offered
        - penalties_not_ideal_group_size,
        sense=GRB.MAXIMIZE,
    )

    m.addConstrs(
        (x.sum("*", "*", student_id) + v[student_id] == 1 for student_id in student_ids),
        name="penalty_if_unassigned",
    )
    m.update()

    m.addConstrs(
        (
            y[project_id, group_id] <= y[project_id, group_id - 1]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
            if group_id > 0
        ),
        name="only_consecutive_group_ids",
    )
    m.update()

    m.addConstrs(
        (
            x.sum(project_id, group_id, "*") >= projects["min_group_size"][project_id] * y[project_id, group_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="ensure_min_group_size",
    )
    m.update()

    m.addConstrs(
        (
            x.sum(project_id, group_id, "*") <= projects["max_group_size"][project_id] * y[project_id, group_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="cap_group_size_at_max",
    )
    m.update()

    m.addConstrs(
        (
            gs_surplus[project_id, group_id]
            >= x.sum(project_id, group_id, "*") - projects["ideal_group_size"][project_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="ensure_correct_group_size_surplus",
    )
    m.update()

    m.addConstrs(
        (
            gs_deficit[project_id, group_id]
            >= projects["ideal_group_size"][project_id]
            - x.sum(project_id, group_id, "*")
            - projects["max_group_size"][project_id] * (1 - y[project_id, group_id])
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="ensure_correct_group_size_deficit",
    )
    m.update()

    max_num_groups = max(projects["max#groups"])

    unique_group_identifiers = {
        (project_id, group_id): project_id + group_id / max_num_groups
        for project_id in project_ids
        for group_id in projects_group_ids[project_id]
    }

    num_projects = len(projects)

    m.addConstrs(
        (
            z[first_student_id, second_student_id] * num_projects
            >= sum(
                unique_group_identifiers[project_id, group_id]
                * (x[project_id, group_id, first_student_id] - x[project_id, group_id, second_student_id])
                for project_id in project_ids
                for group_id in projects_group_ids[project_id]
            )
            for (first_student_id, second_student_id) in mutual_pairs
        ),
        name="ensure_correct_inidicator_different_group_1",
    )
    m.update()

    m.addConstrs(
        (
            z[first_student_id, second_student_id] * num_projects
            >= sum(
                unique_group_identifiers[project_id, group_id]
                * (x[project_id, group_id, second_student_id] - x[project_id, group_id, first_student_id])
                for project_id in project_ids
                for group_id in projects_group_ids[project_id]
            )
            for (first_student_id, second_student_id) in mutual_pairs
        ),
        name="ensure_correct_inidicator_different_group_2",
    )
    m.update()

    m.Params.TimeLimit = 60
    m.optimize()

    best_obj_60sec_and_obj_bound = {"best_obj": m.ObjVal, "obj_bound": m.ObjBound}

    m.dispose()
    gp.disposeDefaultEnv()

    return best_obj_60sec_and_obj_bound


if __name__ == "__main__":
    best_objs_60sec_obj_bounds = {}
    project_quantities = [3, 4, 5]
    student_quantities = [30, 40, 50]
    INSTANCES_PER_COMBINATION = 10

    reward_bilateral = 2
    penalty_unassignment = 3

    folder_projects = Path("instances_projects")
    folder_students = Path("instances_students")

    for project_quantity in project_quantities:
        for student_quantity in student_quantities:
            dimension = f"{project_quantity}_{student_quantity}"
            dimension_subfolder = f"{dimension}_instances"
            for instance in range(INSTANCES_PER_COMBINATION):
                filename_projects = f"generic_{dimension}_projects_{instance}.csv"
                filename_students = f"generic_{dimension}_students_{instance}.csv"
                filepath_projects = folder_projects / dimension_subfolder / filename_projects
                filepath_students = folder_students / dimension_subfolder / filename_students
                projects_info = pd.read_csv(filepath_projects)
                students_info = pd.read_csv(filepath_students)
                students_info["fav_partners"] = students_info["fav_partners"].apply(json.loads)
                students_info["project_prefs"] = students_info["project_prefs"].apply(lambda x: tuple(json.loads(x)))
                best_objs_60sec_obj_bounds[f"generic_{dimension}_{instance}"] = (
                    get_gurobi_best_obj_60sec_and_obj_bound(
                        projects_info,
                        students_info,
                        reward_bilateral,
                        penalty_unassignment,
                    )
                )
    with open("gurobi_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(best_objs_60sec_obj_bounds, f, indent=4)
