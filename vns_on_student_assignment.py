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
        problem_data: tuple[pd.DataFrame, pd.DataFrame],
        reward_bilateral: int,
        penalty_non_assignment: int,
    ):
        self.projects_info, self.students_info = problem_data
        self.reward_bilateral = reward_bilateral
        self.penalty_non_assignment = penalty_non_assignment
        self.projects = tuple(
            (Project(*row) for row in self.projects_info.itertuples())
        )
        self.students = tuple(
            (Student(*row) for row in self.students_info.itertuples())
        )
        self.unassigned: list[Student] = []
        self.objective_value = 0
        self.previous_state = {}
        self.neigborhood_visit_counter = {}
        self.num_students = len(self.students)
        self._initial_solution()
        self.calculate_current_objective_value()

    def run_basic_vns_first_improv(
        self,
        iteration_limit: int,
        assignment_bias: float | int,
        unassignment_prob: float,
    ):
        if not 0 <= unassignment_prob <= 1:
            raise ValueError("A probability must be between 0 and 1.")
        min_neighborhood = 1
        max_neigborhood = 4
        self.neigborhood_visit_counter = {
            neighborhood: 0
            for neighborhood in range(min_neighborhood, max_neigborhood + 1)
        }
        current_iteration = 0
        current_neighborhood = min_neighborhood

        while current_iteration < iteration_limit:
            current_iteration += 1
            self._save_state()

            match current_neighborhood:
                case 1:
                    num_to_move = 1
                    across_projects = False

                case 2:
                    num_to_move = 2
                    across_projects = False

                case 3:
                    num_to_move = 1
                    across_projects = True

                case 4:
                    num_to_move = 2
                    across_projects = True

            self.neigborhood_visit_counter[current_neighborhood] += 1
            self._shake(
                num_to_move,
                across_projects,
                assignment_bias,
                unassignment_prob,
            )

            if self.objective_value > self.previous_state["objective_value"]:
                current_neighborhood = min_neighborhood
                continue

            self._recreate_state()
            if current_neighborhood == max_neigborhood:
                current_neighborhood = min_neighborhood
            else:
                current_neighborhood += 1

    def _save_state(self):
        self.previous_state["projects"] = copy.deepcopy(self.projects)
        self.previous_state["unassigned"] = copy.deepcopy(self.unassigned)
        self.previous_state["objective_value"] = copy.deepcopy(
            self.objective_value
        )

    def _recreate_state(self):
        self.projects = self.previous_state["projects"]
        self.unassigned = self.previous_state["unassigned"]
        self.objective_value = self.previous_state["objective_value"]

    def _shake(
        self,
        num_to_move: int,
        across_projects: bool,
        assignment_bias: float | int,
        unassignment_prob: float,
    ):

        shake_departures = self._shake_departures(num_to_move, assignment_bias)
        departure_deltas = 0
        arrival_deltas = 0
        for shake_departure in shake_departures:
            departure_deltas += self._calculate_leaving_delta(shake_departure)
            shake_arrival = self._shake_arrival(
                shake_departure, across_projects, unassignment_prob
            )
            arrival_deltas += self._calculate_arrival_delta(shake_arrival)
            self._move_student(shake_departure, shake_arrival)

        self.objective_value += departure_deltas + arrival_deltas

    def _move_student(
        self,
        departure_point: (
            tuple[bool, Student] | tuple[Project, ProjectGroup, Student]
        ),
        arrival_point: (
            tuple[str, Student] | tuple[Project, ProjectGroup, Student]
        ),
    ) -> None:
        if departure_point[-1] is not arrival_point[-1]:
            raise ValueError("No clarity which student should be moved!")
        student_was_unassigned = isinstance(departure_point[0], bool)
        student_will_be_unassigned = isinstance(arrival_point[0], str)
        if student_was_unassigned and student_will_be_unassigned:
            return
        moving_student = departure_point[-1]
        # print(f"{moving_student.name} {moving_student.student_id} is moving!")
        if student_was_unassigned and not student_will_be_unassigned:
            # print(
            #     f"{moving_student.name} {moving_student.student_id} is moving!"
            # )
            self.unassigned.remove(moving_student)
            arrival_group = arrival_point[-2]
            arrival_group.accept_student(moving_student)
            return
        if not student_was_unassigned and student_will_be_unassigned:
            departure_group = departure_point[-2]
            departure_group.release_student(moving_student)
            self.unassigned.append(moving_student)
            return
        departure_group = departure_point[-2]
        arrival_group = arrival_point[-2]
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
        if arrival_group.size() < arrival_project.ideal_group_size:
            delta_group_size = arrival_project.pen_size
        else:
            delta_group_size = -arrival_project.pen_size
        return preference_gain + bilateral_reward_gain + delta_group_size

    def _shake_arrival(
        self,
        shake_departure: (
            tuple[Project, ProjectGroup, Student] | tuple[True, Student]
        ),
        across_projects: bool,
        unassignment_prob: float,
    ) -> tuple[Project, ProjectGroup, Student] | tuple[str, Student]:
        if isinstance(shake_departure[0], bool):
            student = shake_departure[1]
            candidate_projects = [
                project
                for project in self.projects
                if any(
                    group.size() < project.max_group_size
                    for group in project.groups
                )
            ]
            if not candidate_projects:
                return ("keep unassigned", student)
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group
                for group in chosen_project.groups
                if group.size() < chosen_project.max_group_size
            ]
            chosen_group: ProjectGroup = rd.choice(candidate_groups)
            return (chosen_project, chosen_group, student)

        if rd.random() < unassignment_prob:
            student = shake_departure[-1]
            return ("unassign", student)

        current_project, current_group, student = shake_departure

        if not across_projects:
            chosen_project = current_project
        else:
            candidate_projects = [
                project
                for project in self.projects
                if project is not current_project
                and any(
                    group.size() < project.max_group_size
                    for group in project.groups
                )
            ]

            if not candidate_projects:
                return ("unassign", student)

            chosen_project = rd.choice(candidate_projects)

        candidate_groups = [
            group
            for group in chosen_project.groups
            if group is not current_group
            and group.size() < chosen_project.max_group_size
        ]

        if not candidate_groups:
            return ("unassign", student)
        chosen_group = rd.choice(candidate_groups)
        return (chosen_project, chosen_group, student)

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
        # if project.num_groups() > project.offered_num_groups and group.size() == 1:
        #     reduced_penalty_num_groups = project.pen_groups
        # else:
        #     reduced_penalty_num_groups = 0
        if group.size() > project.ideal_group_size:
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
        self, num_to_move: int, assignment_bias: float | int
    ) -> list[tuple[bool, Student] | tuple[Project, ProjectGroup, Student]]:
        departures_specs = []
        departing_students: list[Student] = []
        for _ in range(num_to_move):
            unassigned_student_was_chosen = False
            unassigned_remaining = [
                student
                for student in self.unassigned
                if student not in departing_students
            ]

            if (num_unassigned_remaining := len(unassigned_remaining)) and (
                rd.random()
                > num_unassigned_remaining
                / self.num_students
                / assignment_bias
            ):
                student_to_move: Student = rd.choice(unassigned_remaining)
                departing_students.append(student_to_move)
                unassigned_student_was_chosen = True
                departures_specs.append(
                    (
                        unassigned_student_was_chosen,
                        student_to_move,
                    )
                )
                continue

            candidate_projects = [
                project
                for project in self.projects
                if any(
                    group.remaining_size(departing_students)
                    > project.min_group_size
                    for group in project.groups
                )
            ]
            if not candidate_projects:
                raise ValueError("No student can be moved!")
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group
                for group in chosen_project.groups
                if group.remaining_size(departing_students)
                > chosen_project.min_group_size
            ]
            chosen_group = rd.choice(candidate_groups)
            student_to_move = rd.choice(
                chosen_group.remaining_students(departing_students)
            )
            departing_students.append(student_to_move)
            departures_specs.append(
                (
                    chosen_project,
                    chosen_group,
                    student_to_move,
                )
            )
        return departures_specs

    def calculate_current_objective_value(self):
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
        for project in self.projects:
            for group in project.groups:
                group.populate_bilateral_preferences_set()
        return sum(
            self.reward_bilateral * len(group.bilateral_preferences)
            for project in self.projects
            for group in project.groups
        )

    def _sum_no_assignment_penalties(self):
        return len(self.unassigned) * self.penalty_non_assignment

    def _sum_group_surplus_penalties(self):
        return sum(
            max(0, project.num_groups() - project.offered_num_groups)
            * project.pen_groups
            for project in self.projects
        )

    def _sum_group_size_penalties(self):
        return sum(
            abs(group.size() - project.ideal_group_size) * project.pen_size
            for project in self.projects
            for group in project.groups
        )

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
                    project.num_groups() < project.offered_num_groups
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

    def _build_initial_projects_waitlists(self) -> dict[int, list[Student]]:
        return {
            (proj_id := project.project_id): sorted(
                self.students,
                key=lambda student: student.projects_prefs[proj_id],
                reverse=True,
            )
            for project in self.projects
        }

    def report_input_data(self):
        """Print the data frames that constitute the problem data."""
        pd.set_option("display.max_columns", None)
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
    vns_run.run_basic_vns_first_improv(1000, 3, 0.05)
    vns_run.report_current_solution()
    vns_run.calculate_current_objective_value()
    print(vns_run.objective_value)
    print(vns_run.unassigned)
    for neigborhood, num_visits in vns_run.neigborhood_visit_counter.items():
        print(f"{neigborhood}: {num_visits}")
