"""Used to benchmark Gurobi."""

import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from settings import BenchmarkSettingsGurobi


class SolutionsRecorder:
    """Collects and returns Gurobi's intermediate and final solutions.

    Attributes:
        model: problem to be optimized
        solutions_with_runtime: solutions recorded in callback.
        time_limit: Limit on time spent optimizing.
    """

    def __init__(self, model, time_limit):
        self.model: gp.Model = model
        self.solutions_with_runtime: list = []
        self.time_limit: int = time_limit

    def _record_solutions(self, model, where):
        if where == GRB.Callback.MIPSOL:
            self.solutions_with_runtime.append(
                {
                    "incumbent_obj": model.cbGet(GRB.Callback.MIPSOL_OBJ),
                    "obj_bound": model.cbGet(GRB.Callback.MIPSOL_OBJBND),
                    "runtime": model.cbGet(GRB.Callback.RUNTIME),
                }
            )

    def solutions_with_bound(self):
        """Returns intermediate and final solutions."""
        self.solutions_with_runtime.clear()
        self.model.Params.TimeLimit = self.time_limit
        self.model.optimize(self._record_solutions)
        self.solutions_with_runtime.append(
            {"incumbent_obj": self.model.ObjVal, "obj_bound": self.model.ObjBound, "runtime": self.model.Runtime}
        )
        return self.solutions_with_runtime


def get_gurobi_model(
    projects: pd.DataFrame, students: pd.DataFrame, reward_bilateral: int, penalty_unassignment: int
) -> gp.Model:
    """Returns Gurobi model.

    Args:
        projects: The project names with each project's guidelines, whishes
            and penalties regarding the number of groups and group sizes
        students: The project preferences for all projects and the partner
            preferences i.e., the students a student wants to work with
            the most for all students in the problem.
        reward_bilateral: The fixed reward for every occurrence in the solution
            of two students who have specified each other as partner
            preferences being in the same group.
        penalty_unassignment: The fixed penalty for every student in the solution
            who is not assigned to a group.
    """
    model = gp.Model()

    project_ids = range(len(projects))
    student_ids = range(len(students))

    projects_group_ids = {project_id: range(projects["max#groups"][project_id]) for project_id in project_ids}

    x = model.addVars(
        (
            (project_id, group_id, student_id)
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
            for student_id in student_ids
        ),
        vtype=GRB.BINARY,
        name="assign",
    )
    model.update()

    y = model.addVars(
        ((project_id, group_id) for project_id in project_ids for group_id in projects_group_ids[project_id]),
        vtype=GRB.BINARY,
        name="establish_group",
    )
    favorite_partners_students = students["fav_partners"].tolist()
    mutual_pairs = {
        (student_id, partner_id)
        for student_id, favorite_partners in enumerate(favorite_partners_students)
        for partner_id in favorite_partners
        if student_id < partner_id and student_id in favorite_partners_students[partner_id]
    }

    z = model.addVars(mutual_pairs, vtype=GRB.BINARY, name="not_realize_bilateral_cooperation_wish")

    v = model.addVars(student_ids, name="student_unassigned")

    gs_surplus = model.addVars(
        ((project_id, group_id) for project_id in project_ids for group_id in projects_group_ids[project_id]),
        name="group_size_surplus",
    )

    gs_deficit = model.addVars(
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

    rewards_bilateral_cooperation_wish_realized = reward_bilateral * gp.quicksum(
        1 - z[*mutual_pair] for mutual_pair in mutual_pairs
    )

    penalties_student_unassigned = penalty_unassignment * gp.quicksum(v.values())

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

    model.setObjective(
        rewards_student_project_preference
        + rewards_bilateral_cooperation_wish_realized
        - penalties_student_unassigned
        - penalties_more_groups_than_offered
        - penalties_not_ideal_group_size,
        sense=GRB.MAXIMIZE,
    )

    model.addConstrs(
        (x.sum("*", "*", student_id) + v[student_id] == 1 for student_id in student_ids),
        name="penalty_if_unassigned",
    )
    model.update()

    model.addConstrs(
        (
            y[project_id, group_id] <= y[project_id, group_id - 1]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
            if group_id > 0
        ),
        name="only_consecutive_group_ids",
    )
    model.update()

    model.addConstrs(
        (
            x.sum(project_id, group_id, "*") >= projects["min_group_size"][project_id] * y[project_id, group_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="ensure_min_group_size",
    )
    model.update()

    model.addConstrs(
        (
            x.sum(project_id, group_id, "*") <= projects["max_group_size"][project_id] * y[project_id, group_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="cap_group_size_at_max",
    )
    model.update()

    model.addConstrs(
        (
            gs_surplus[project_id, group_id]
            >= x.sum(project_id, group_id, "*") - projects["ideal_group_size"][project_id]
            for project_id in project_ids
            for group_id in projects_group_ids[project_id]
        ),
        name="ensure_correct_group_size_surplus",
    )
    model.update()

    model.addConstrs(
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
    model.update()

    max_num_groups = max(projects["max#groups"])

    unique_group_identifiers = {
        (project_id, group_id): project_id + group_id / max_num_groups
        for project_id in project_ids
        for group_id in projects_group_ids[project_id]
    }

    num_projects = len(projects)

    model.addConstrs(
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
        name="ensure_correct_indicator_different_group_1",
    )
    model.update()

    model.addConstrs(
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
        name="ensure_correct_indicator_different_group_2",
    )
    model.update()

    return model


def benchmark_gurobi(
    project_quantities: list[int],
    student_quantities: list[int],
    instances_per_dimension: int,
    reward_bilateral: int,
    penalty_unassignment: int,
    filename: str,
    time_limit: int,
) -> None:
    """Saves benchmarks of Gurobi in designated JSON.

    Loads one instance after another. For every problem instance:
    Initializes an instance of SolutionsRecorder, which keeps
    track of intermediate and final results.

    Args:
        project_quantities: Only instances with a number of projects
            that matches one of the numbers will be benchmarked.
        student_quantities: Only instances with a number of students
            that matches one of the numbers will be benchmarked.
        instances_per_dimension: How many instances for each
            combination of number of projects and number of students
            will be benchmark. SHOULD NOT EXCEED THE NUMBER OF
            INSTANCES EXISTENT PER COMBINATION.
        reward_bilateral: The fixed reward for every occurrence in the solution
            of two students who have specified each other as partner
            preferences being in the same group.
        penalty_unassignment: The fixed penalty for every student in the solution
            who is not assigned to a group.
        filename: The name of the JSON in which the results
            of the benchmark run are supposed to be saved.
        time_limit: Limit on time spent optimizing one instance.

    Raises:
        ValueError: Already a file at path of filename.
    """
    solution_and_bound_developments = {}
    folder_projects = Path("instances_projects")
    folder_students = Path("instances_students")
    results_path = Path(filename)
    if results_path.is_file():
        raise ValueError(f"{results_path} already exists!")
    for project_quantity in project_quantities:
        for student_quantity in student_quantities:
            dimension = f"{project_quantity}_{student_quantity}"
            dimension_subfolder = f"{dimension}_instances"
            for instance in range(instances_per_dimension):
                filename_projects = f"generic_{dimension}_projects_{instance}.csv"
                filename_students = f"generic_{dimension}_students_{instance}.csv"
                filepath_projects = folder_projects / dimension_subfolder / filename_projects
                filepath_students = folder_students / dimension_subfolder / filename_students
                projects_info = pd.read_csv(filepath_projects)
                students_info = pd.read_csv(filepath_students)
                students_info["fav_partners"] = students_info["fav_partners"].apply(json.loads)
                students_info["project_prefs"] = students_info["project_prefs"].apply(lambda x: tuple(json.loads(x)))
                model = get_gurobi_model(
                    projects=projects_info,
                    students=students_info,
                    reward_bilateral=reward_bilateral,
                    penalty_unassignment=penalty_unassignment,
                )
                solution_and_bound_developments[f"generic_{dimension}_{instance}"] = SolutionsRecorder(
                    model, time_limit
                ).solutions_with_bound()
                results_path.write_text(json.dumps(solution_and_bound_developments, indent=4), encoding="utf-8")
                model.dispose()
                gp.disposeDefaultEnv()


if __name__ == "__main__":
    settings = BenchmarkSettingsGurobi()
    benchmark_gurobi(
        project_quantities=settings.project_quantities,
        student_quantities=settings.student_quantities,
        instances_per_dimension=settings.instances_per_dimension,
        reward_bilateral=settings.reward_bilateral,
        penalty_unassignment=settings.penalty_unassignment,
        filename=settings.filename,
        time_limit=settings.time_limit,
    )
