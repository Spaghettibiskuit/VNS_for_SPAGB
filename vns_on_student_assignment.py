"""Main file of VNS implementation for a student assignment problem."""

import copy
import pickle
import random as rd
from pathlib import Path

import pandas as pd

from problem_data import generate_throwaway_instance
from project import Project
from project_group import ProjectGroup
from student import Student

# rd.seed(100)
rd.seed(3869)


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
        self.objective_value = 0
        self.before_shake = {}

        self._initialize_projects()
        self._initialize_students()
        self.num_students = len(self.students)
        self._initial_solution()
        self._calculate_current_objective_value()

    def run_basic_vns_first_improv(self, iteration_limit: int):
        min_neighborhood = 1
        max_neigborhood = 5
        unassigned_bias = 3
        current_neighborhood = min_neighborhood
        current_iteration = 0
        while current_iteration < iteration_limit:
            self._save_state_before_shake()
            self._shake(current_neighborhood, unassigned_bias)
            if self.before_shake["objective_value"] > self.objective_value:
                self._recreate_state_before_shake()
            current_iteration += 1

    def _save_state_before_shake(self):
        self.before_shake["projects"] = copy.deepcopy(self.projects)
        self.before_shake["unassigned"] = copy.deepcopy(self.unassigned)
        self.before_shake["objective_value"] = copy.deepcopy(
            self.objective_value
        )

    def _recreate_state_before_shake(self):
        self.projects = self.before_shake["projects"]
        self.unassigned = self.before_shake["unassigned"]
        self.objective_value = self.before_shake["objective_value"]

    def _shake(self, neighborhood: int, unassigned_bias: float | int):

        shake_departures = self._shake_departures(
            neighborhood, unassigned_bias
        )
        departure_deltas = 0
        arrival_deltas = 0
        for shake_departure in shake_departures:
            departure_deltas += self._calculate_leaving_delta(shake_departure)
            shake_arrival = self._shake_arrival(
                neighborhood, shake_departure, unassigned_bias
            )
            arrival_deltas += self._calculate_arrival_delta(shake_arrival)
            self._move_student(shake_departure, shake_arrival)

        self.objective_value += departure_deltas + arrival_deltas

    def _move_student(
        self,
        shake_departure: (
            tuple[bool, Student] | tuple[Project, ProjectGroup, Student]
        ),
        shake_arrival: (
            tuple[str, Student] | tuple[Project, ProjectGroup, Student]
        ),
    ) -> None:
        if shake_departure[-1] is not shake_arrival[-1]:
            raise ValueError("No clarity which Student should be moved!")
        student_was_unassigned = isinstance(shake_departure[0], bool)
        student_will_be_unassigned = isinstance(shake_arrival[0], str)
        if student_was_unassigned and student_will_be_unassigned:
            return
        moving_student = shake_departure[-1]
        # print(f"{moving_student.name} {moving_student.student_id} is moving!")
        if student_was_unassigned and not student_will_be_unassigned:
            self.unassigned.remove(moving_student)
            arrival_group = shake_arrival[-2]
            arrival_group.accept_student(moving_student)
            return
        if not student_was_unassigned and student_will_be_unassigned:
            departure_group = shake_departure[-2]
            departure_group.release_student(moving_student)
            self.unassigned.append(moving_student)
            return
        departure_group = shake_departure[-2]
        arrival_group = shake_arrival[-2]
        departure_group.release_student(moving_student)
        arrival_group.accept_student(moving_student)

    def _calculate_arrival_delta(
        self,
        shake_arrival: (
            tuple[str, Student] | tuple[Project, ProjectGroup, Student]
        ),
    ) -> int:
        if isinstance(shake_arrival[0], str):
            return -self.penalty_non_assignment
        arrival_project, arrival_group, arriving_student = shake_arrival
        preference_gain = arriving_student.projects_prefs[
            arrival_project.project_id
        ]
        bilateral_reward_gain = sum(
            self.reward_bilateral
            for student in arrival_group.students
            if student.student_id in arriving_student.fav_partners
            and arriving_student.student_id in student.fav_partners
        )
        # Establishment of a new group not yet implemented.
        if arrival_group.size < arrival_project.ideal_group_size:
            delta_group_size = arrival_project.pen_size
        else:
            delta_group_size = -arrival_project.pen_size
        return preference_gain + bilateral_reward_gain + delta_group_size

    def _shake_arrival(
        self,
        neighborhood: int,
        shake_departure: (
            tuple[Project, ProjectGroup, Student] | tuple[True, Student]
        ),
        unassigned_bias: int,
    ) -> tuple[Project, ProjectGroup, Student] | tuple[str, Student]:
        if neighborhood == 1:
            if isinstance(shake_departure[0], bool):
                student = shake_departure[1]
                candidate_projects = [
                    project
                    for project in self.projects
                    if any(
                        group.size < project.max_group_size
                        for group in project.groups
                    )
                ]
                if not candidate_projects:
                    return tuple(("keep unassigned", student))
                chosen_project: Project = rd.choice(candidate_projects)
                candidate_groups = [
                    group
                    for group in chosen_project.groups
                    if group.size < chosen_project.max_group_size
                ]
                chosen_group: ProjectGroup = rd.choice(candidate_groups)
                return tuple((chosen_project, chosen_group, student))

            if (
                rd.random()
                < len(self.unassigned) / self.num_students / unassigned_bias
            ):
                return tuple(("unassign", student))

            project, current_group, student = shake_departure
            candidate_groups = [
                group
                for group in project.groups
                if group is not current_group
                and group.size < project.max_group_size
            ]
            if not candidate_groups:
                return tuple(("unassign", student))
            chosen_group = rd.choice(candidate_groups)
            return tuple((project, chosen_group, student))

        return "something"

    def _calculate_leaving_delta(
        self,
        shake_departure: (
            tuple[Project, ProjectGroup, Student] | tuple[True, Student]
        ),
    ) -> int:
        if isinstance(shake_departure[0], bool):
            return self.penalty_non_assignment
        project, group, student = shake_departure
        preference_loss = student.projects_prefs[project.project_id]
        bilateral_reward_loss = sum(
            self.reward_bilateral
            for bilateral_preference in group.bilateral_preferences
            if student.student_id in bilateral_preference
        )
        # currently no mechanism for deleting groups.
        # if project.num_groups > project.offered_num_groups and group.size == 1:
        #     reduced_penalty_num_groups = project.pen_groups
        # else:
        #     reduced_penalty_num_groups = 0
        if group.size > project.ideal_group_size:
            delta_group_size = project.pen_size
        else:
            delta_group_size = -project.pen_size
        return (
            -preference_loss
            - bilateral_reward_loss
            # + reduced_penalty_num_groups
            + delta_group_size
        )

    def _shake_departures(
        self, neighborhood: int, unassigned_bias: float | int
    ) -> list[tuple[bool, Student] | tuple[Project, ProjectGroup, Student]]:
        match neighborhood:
            case 1:
                unassigned_student_chosen = False
                if (
                    num_unassigned := len(self.unassigned)
                ) >= neighborhood and (
                    rd.random()
                    > num_unassigned / self.num_students / unassigned_bias
                ):
                    unassigned_student_chosen = True
                    student_to_move = rd.choice(self.unassigned)
                    return [
                        tuple(
                            (
                                unassigned_student_chosen,
                                student_to_move,
                            )
                        )
                    ]

                candidate_projects = [
                    project
                    for project in self.projects
                    if any(
                        group.size - project.min_group_size >= neighborhood
                        for group in project.groups
                    )
                ]
                if not candidate_projects:
                    raise ValueError("No student can be moved!")
                chosen_project: Project = rd.choice(candidate_projects)
                candidate_groups = [
                    group
                    for group in chosen_project.groups
                    if (group.size - neighborhood)
                    >= chosen_project.min_group_size
                ]
                chosen_group = rd.choice(candidate_groups)
                student_to_move = rd.choice(chosen_group.students)
                return [
                    tuple(
                        (
                            chosen_project,
                            chosen_group,
                            student_to_move,
                        )
                    )
                ]

            case _:
                raise ValueError(
                    f"Invalid neigborhood indicator ({neighborhood})"
                )

    def _calculate_current_objective_value(self):
        self.objective_value = (
            self._sum_preferences()
            + self._sum_join_rewards()
            - self._sum_no_assignment_penalties()
            - self._sum_group_surplus_penalties()
            - self._sum_group_size_penalties()
        )

    def _sum_preferences(self):
        return sum(
            student.projects_prefs[project.project_id]
            for project in self.projects
            for group in project.groups
            for student in group.students
        )

    def _sum_join_rewards(self):
        return sum(
            self.reward_bilateral * len(group.bilateral_preferences)
            for project in self.projects
            for group in project.groups
        )

    def _sum_no_assignment_penalties(self):
        return len(self.unassigned) * self.penalty_non_assignment

    def _sum_group_surplus_penalties(self):
        return sum(
            max(0, project.num_groups - project.offered_num_groups)
            * project.pen_groups
            for project in self.projects
        )

    def _sum_group_size_penalties(self):
        return sum(
            abs(group.size - project.ideal_group_size) * project.pen_size
            for project in self.projects
            for group in project.groups
        )

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
                    now_assigned_students: list[Student] = unassigned_students[
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
        print(f"The objective value is {self.objective_value}")


if __name__ == "__main__":
    solve_specific_instance = False
    if solve_specific_instance:
        folder = Path("instances")
        filename = "generic_3_30.pkl"
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
    vns_run.run_basic_vns_first_improv(300)
    vns_run.report_current_solution()
    vns_run._calculate_current_objective_value()
    print(vns_run.objective_value)
    print(vns_run.unassigned)
