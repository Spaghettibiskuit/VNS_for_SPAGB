"""Main file of VNS implementation for a student assignment problem."""

import pickle
from pathlib import Path

import pandas as pd

from problem_data import generate_throwaway_instance
from project import Project
from student import Student


class VarNeighborhoodSearch:
    """Applies VNS on a student assignment problem."""

    def __init__(
        self,
        problem_data: tuple[pd.DataFrame],
        reward_bilateral: int,
        penalty_non_assignment: int,
    ):
        self.projects_info, self.students_info = problem_data
        self.reward_bilateral = reward_bilateral
        self.penalty_non_assignment = penalty_non_assignment
        self.projects: list[Project] = []
        self.students: list[Student] = []
        self.unassigned: list[Student] = []

        self._initialize_projects()
        self._initialize_students()
        self._initial_solution()

    def _initialize_projects(self):
        for row in self.projects_info.itertuples():
            (
                project_id,
                name,
                offered_num_groups,
                max_num_groups,
                ideal_group_size,
                min_group_size,
                max_group_size,
                pen_groups,
                pen_size,
            ) = row
            self.projects.append(
                Project(
                    project_id,
                    name,
                    offered_num_groups,
                    max_num_groups,
                    ideal_group_size,
                    min_group_size,
                    max_group_size,
                    pen_groups,
                    pen_size,
                )
            )

        self.projects = tuple(self.projects)

    def _initialize_students(self):
        for row in self.students_info.itertuples():
            student_id, name, fav_partners, projects_prefs = row
            self.students.append(
                Student(student_id, name, fav_partners, projects_prefs)
            )

        self.students = tuple(self.students)

    def _initial_solution(self):
        project_waitlists = self._build_initial_projects_waitlists()
        num_students_unassigned = len(self.students)
        any_group_added = True
        while num_students_unassigned > 0 and any_group_added:
            any_group_added = False
            for project in self.projects:
                unassigned_students = [
                    student
                    for student in project_waitlists[project.project_id]
                    if not student.assigned
                ]
                if (
                    project.num_groups < project.offered_num_groups
                    and len(unassigned_students) >= project.ideal_group_size
                ):
                    now_assigned_students = unassigned_students[
                        : project.ideal_group_size
                    ]
                    project.add_initial_group_ideal_size(now_assigned_students)
                    for student in now_assigned_students:
                        student.assigned = True
                    num_students_unassigned -= project.ideal_group_size
                    any_group_added = True

        self.unassigned = [
            student for student in self.students if not student.assigned
        ]

    def _build_initial_projects_waitlists(self) -> dict[list[Student]]:
        projects_waitlists = {
            proj_id: [] for proj_id in range(len(self.projects))
        }
        min_pref_val, max_pref_val = self._min_max_pref_val()
        target_val = max_pref_val
        while target_val >= min_pref_val:
            for student in self.students:
                for proj_id, pref_val in enumerate(student.projects_prefs):
                    if pref_val == target_val:
                        projects_waitlists[proj_id].append(student)
            target_val -= 1

        return projects_waitlists

    def _min_max_pref_val(self):
        min_val = float("inf")
        max_val = -float("inf")
        for student in self.students:
            for pref_val in student.projects_prefs:
                min_val = min(min_val, pref_val)
                max_val = max(max_val, pref_val)
        return tuple((min_val, max_val))

    def report_input_data(self):
        """Print the data frames that constitute the problem data."""
        print(self.projects_info)
        print(self.students_info)

    def report_num_projects_and_students(self):
        """Return number of projects."""
        print(
            f"Number of Projects: {len(self.projects)}\nNumber of Students: {len(self.students)}"
        )

    def report_current_solution(self):
        """Currently reports who is in which group."""
        for project in self.projects:
            print(f"\nThese are the groups in the project {project.name}")
            for group in project.groups:
                print("\n")
                for student in group.students:
                    print(student.name, student.student_id)
        print("\nThese students were not assigned:")
        for student in self.unassigned:
            print(student.name, student.student_id)


if __name__ == "__main__":
    solve_specific_instance = True
    if solve_specific_instance:
        folder = Path("instances")
        filename = "generic_3_20.pkl"
        instance_path = folder / filename
        with instance_path.open("rb") as f:
            problem_instance = pickle.load(f)
    else:
        problem_instance = generate_throwaway_instance(
            num_projects=3, num_students=30
        )

    vns_run = VarNeighborhoodSearch(
        problem_instance,
        reward_bilateral=2,
        penalty_non_assignment=3,
    )
    vns_run.report_input_data()
    vns_run.report_num_projects_and_students()
    vns_run.report_current_solution()
