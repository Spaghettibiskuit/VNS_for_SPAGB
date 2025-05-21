"""Main file of VNS implementation for a student assignment problem."""

import itertools as it
import pickle
import random as rd
from pathlib import Path

import pandas as pd

from problem_data import generate_throwaway_instance
from project import Project
from project_group import ProjectGroup
from student import Student

# rd.seed()
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
        self.projects = tuple((Project(*row) for row in self.projects_info.itertuples()))
        self.students = tuple((Student(*row) for row in self.students_info.itertuples()))
        self.unassigned: list[Student] = []
        self.objective_value = 0
        # self.before_shake = {}
        self.neigborhood_visit_counter = {}
        self.num_students = len(self.students)
        self._initial_solution()
        self.calculate_current_objective_value()
        self.best_objective_value = self.objective_value
        self.move_reversals = []

    def run_gen_vns_best_improvement(
        self,
        iteration_limit: int,
        assignment_bias: float | int,
        unassignment_prob: float,
    ):
        if not 0 <= unassignment_prob <= 1:
            raise ValueError("A probability must be between 0 and 1.")
        min_neighborhood = 1
        max_neigborhood = 8
        self.neigborhood_visit_counter = {
            neighborhood: 0 for neighborhood in range(min_neighborhood, max_neigborhood + 1)
        }
        current_iteration = 0
        current_neighborhood = min_neighborhood

        while current_iteration < iteration_limit:

            current_iteration += 1

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

                case 5:
                    num_to_move = 3
                    across_projects = False

                case 6:
                    num_to_move = 3
                    across_projects = True

                case 7:
                    num_to_move = 4
                    across_projects = True

                case 8:
                    num_to_move = 5
                    across_projects = True

            self.neigborhood_visit_counter[current_neighborhood] += 1

            print(f"\nIteration {current_iteration}/{iteration_limit}")
            print("\nThe current neighborhood is:", current_neighborhood)

            self._shake(
                num_to_move,
                across_projects,
                assignment_bias,
                unassignment_prob,
            )

            print("The objective value after the shake is:", self.objective_value)

            self._var_neigborhood_descent(num_to_move, across_projects)

            curr_obj_val = (
                self._sum_preferences()
                + self._sum_join_rewards()
                - self._sum_no_assignment_penalties()
                - self._sum_group_surplus_penalties()
                - self._sum_group_size_penalties()
            )

            print("The objective value after VND is:", curr_obj_val)
            print("The stated current objective value after VND is:", self.objective_value)

            if self.objective_value > self.best_objective_value:
                self.best_objective_value = self.objective_value
                current_neighborhood = min_neighborhood
            else:
                for move_reversal in reversed(self.move_reversals):
                    self._move_student(*move_reversal)
                self.objective_value = self.best_objective_value

                if current_neighborhood == max_neigborhood:
                    current_neighborhood = min_neighborhood
                else:
                    current_neighborhood += 1

            self.move_reversals = []

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
            shake_arrival = self._shake_arrival(shake_departure, across_projects, unassignment_prob)
            arrival_deltas += self._calculate_arrival_delta(shake_arrival)
            self._move_student(shake_departure, shake_arrival)
            self.move_reversals.append((shake_arrival, shake_departure))

        self.objective_value += departure_deltas + arrival_deltas

    def _var_neigborhood_descent(self, max_to_move: int, across_projects: bool, min_to_move: int = 1):
        destinations, current_locations = self._prerequisites_vnd()

        num_to_move = min_to_move
        while num_to_move <= max_to_move:

            # if max_to_move == 3 and across_projects is False:
            #     print("\nThe num of students to move is:", num_to_move)

            best_moves_local, delta = self._local_search_best_improvement(
                destinations, current_locations, across_projects, num_to_move
            )
            if delta > 0:
                self.objective_value += delta
                for best_move_local in best_moves_local:
                    # print(f"\nThe best move is {best_move_local}")
                    departure, arrival = best_move_local
                    if departure[-1] is not arrival[-1]:
                        raise ValueError("No clarity which student to be moved.")

                    self._move_student(departure, arrival)
                    self.move_reversals.append((arrival, departure))

                    if arrival[0] is not self.unassigned:
                        arrival: tuple[Project, ProjectGroup, Student]
                        project, group, student = arrival
                        current_locations[student.student_id] = (
                            project,
                            group,
                        )

                    else:
                        student = arrival[-1]
                        current_locations[student.student_id] = self.unassigned

                num_to_move = min_to_move

            else:
                num_to_move += 1

    def _local_search_best_improvement(
        self,
        destinations: tuple[tuple[Project, ProjectGroup] | list[Student]],
        current_locations: dict[int : tuple[Project, ProjectGroup] | list[Student]],
        across_projects: bool,
        num_to_move: int,
    ) -> tuple[
        tuple[
            tuple[
                tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
            ]
        ],
        int,
    ]:
        best_move_combination = None
        best_delta = -10e6

        combinations_student_ids = tuple(it.combinations(range(self.num_students), num_to_move))
        cartestian_product_destinations = tuple(it.product(destinations, repeat=num_to_move))

        for combination_ids in combinations_student_ids:
            if not self._all_in_combination_can_leave(combination_ids, current_locations):
                continue

            # Everybody should move
            valid_destination_combinations = tuple(
                filter(
                    lambda destinations_for_combination, combination_ids=combination_ids: all(
                        (
                            current_locations[combination_ids[i]][1] is not destinations_for_combination[i][1]
                            if (current_locations[combination_ids[i]] is not self.unassigned)
                            and (destinations_for_combination[i] is not self.unassigned)
                            # This filters out from unassigned to unassigned
                            else current_locations[combination_ids[i]] is not destinations_for_combination[i]
                        )
                        for i in range(num_to_move)
                    ),
                    cartestian_product_destinations,
                )
            )

            if not across_projects:
                # Everybody should stay in their project or be unassigned
                valid_destination_combinations = tuple(
                    filter(
                        lambda destinations_for_combination, combination_ids=combination_ids: all(
                            (
                                (current_locations[combination_ids[i]][0] is destinations_for_combination[i][0])
                                if (current_locations[combination_ids[i]] is not self.unassigned)
                                and (destinations_for_combination[i] is not self.unassigned)
                                else True
                            )
                            for i in range(num_to_move)
                        ),
                        valid_destination_combinations,
                    )
                )

            corresponding_departures = tuple(
                (
                    (
                        (
                            *student_location,
                            self.students[student_id],
                        )
                        if (student_location := current_locations[student_id]) is not self.unassigned
                        else (self.unassigned, self.students[student_id])
                    )
                    for student_id in combination_ids
                )
            )

            moving_students = tuple(
                (corresponding_departure[-1] for corresponding_departure in corresponding_departures)
            )

            best_arrival_combination, delta = self._find_best_moves_combination(
                valid_destination_combinations,
                corresponding_departures,
                moving_students,
            )

            if delta > best_delta:

                best_delta = delta
                best_move_combination = tuple(
                    (corresponding_departures[i], best_arrival_combination[i]) for i in range(num_to_move)
                )

        return best_move_combination, best_delta

    def _all_in_combination_can_leave(
        self,
        combination_ids: tuple[int],
        current_locations_all_students: dict[int : tuple[Project, ProjectGroup] | list[Student]],
    ):
        current_locations_comb = [current_locations_all_students[student_id] for student_id in combination_ids]
        for current_location in current_locations_comb:
            if current_location is not self.unassigned:
                num_occurences = current_locations_comb.count(current_location)
                current_location: tuple[Project, ProjectGroup]
                project, group = current_location
                if group.size() - num_occurences < project.min_group_size:
                    return False
        return True

    def _find_best_moves_combination(
        self,
        destination_combinations: tuple[tuple[tuple[Project, ProjectGroup] | list[Student]]],
        corresponding_departures: tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]],
        moving_students: tuple[Student],
    ) -> tuple[
        tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]],
        int,
    ]:
        best_arrival_combination = 0
        best_delta_combination = -10e6

        for destination_combination in destination_combinations:
            delta_combination = 0
            invalid_destination = False
            reversal_moves = []
            for i, destination in enumerate(destination_combination):
                departure_point = corresponding_departures[i]

                student = departure_point[-1]

                if student is not moving_students[i]:
                    raise ValueError("There is a mix up of students!")

                delta_combination += self._calculate_leaving_delta(departure_point)

                if destination is not self.unassigned:
                    destination: tuple[Project, ProjectGroup]
                    project, group = destination
                    if group.size() == project.max_group_size:
                        invalid_destination = True
                        break
                    arrival = (project, group, student)
                else:
                    arrival = (self.unassigned, student)
                delta_combination += self._calculate_arrival_delta(arrival)
                self._move_student(departure_point, arrival)
                reversal_moves.append((arrival, departure_point))

            if invalid_destination:
                for move in reversal_moves:
                    self._move_student(*move)
                continue

            if delta_combination > best_delta_combination:
                best_delta_combination = delta_combination
                best_arrival_combination = tuple(
                    (
                        (*destination, moving_students[i])
                        if destination is not self.unassigned
                        else (self.unassigned, moving_students[i])
                    )
                    for i, destination in enumerate(destination_combination)
                )
            for move in reversal_moves:
                self._move_student(*move)

        return best_arrival_combination, best_delta_combination

    def _prerequisites_vnd(
        self,
    ) -> tuple[
        tuple[tuple[Project, ProjectGroup] | list[Student]],
        dict[int : tuple[Project, ProjectGroup] | list[Student]],
    ]:
        destinations = tuple((project, group) for project in self.projects for group in project.groups) + (
            self.unassigned,
        )

        current_locations = {
            student.student_id: (project, group)
            for project in self.projects
            for group in project.groups
            for student in group.students
        }

        for student in self.unassigned:
            current_locations[student.student_id] = self.unassigned

        return (destinations, current_locations)

    def _move_student(
        self,
        departure_point: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
        arrival_point: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
    ) -> None:
        if departure_point[-1] is not arrival_point[-1]:
            raise ValueError("No clarity which student should be moved!")
        student_was_unassigned = departure_point[0] is self.unassigned
        student_will_be_unassigned = arrival_point[0] is self.unassigned
        if student_was_unassigned and student_will_be_unassigned:
            return
        moving_student = departure_point[-1]

        if student_was_unassigned and not student_will_be_unassigned:
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
        shake_arrival: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
    ) -> int:
        if shake_arrival[0] is self.unassigned:
            return -self.penalty_non_assignment
        arrival_project, arrival_group, arriving_student = shake_arrival
        preference_gain = arriving_student.projects_prefs[arrival_project.project_id]
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
        shake_departure: tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
        across_projects: bool,
        unassignment_prob: float,
    ) -> tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]:
        if shake_departure[0] is self.unassigned:
            student = shake_departure[-1]
            candidate_projects = [
                project
                for project in self.projects
                if any(group.size() < project.max_group_size for group in project.groups)
            ]
            if not candidate_projects:
                return (self.unassigned, student)
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group for group in chosen_project.groups if group.size() < chosen_project.max_group_size
            ]
            chosen_group = rd.choice(candidate_groups)
            return (chosen_project, chosen_group, student)

        if rd.random() < unassignment_prob:
            student = shake_departure[-1]
            return (self.unassigned, student)

        current_project, current_group, student = shake_departure

        if not across_projects:
            chosen_project = current_project
        else:
            candidate_projects = [
                project
                for project in self.projects
                if project is not current_project
                and any(group.size() < project.max_group_size for group in project.groups)
            ]

            if not candidate_projects:
                return (self.unassigned, student)

            chosen_project = rd.choice(candidate_projects)

        candidate_groups = [
            group
            for group in chosen_project.groups
            if group is not current_group and group.size() < chosen_project.max_group_size
        ]

        if not candidate_groups:
            return (self.unassigned, student)
        chosen_group = rd.choice(candidate_groups)
        return (chosen_project, chosen_group, student)

    def _calculate_leaving_delta(
        self,
        departure_point: tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
    ) -> int:
        if departure_point[0] is self.unassigned:
            return self.penalty_non_assignment
        project, group, student = departure_point
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
    ) -> list[tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student]]:
        departures_specs = []
        departing_students: list[Student] = []
        for _ in range(num_to_move):
            unassigned_remaining = [student for student in self.unassigned if student not in departing_students]

            if (num_unassigned_remaining := len(unassigned_remaining)) and (
                rd.random() < num_unassigned_remaining / self.num_students * assignment_bias
            ):
                student_to_move: Student = rd.choice(unassigned_remaining)
                departing_students.append(student_to_move)
                departures_specs.append(
                    (
                        self.unassigned,
                        student_to_move,
                    )
                )
                continue

            candidate_projects = [
                project
                for project in self.projects
                if any(group.remaining_size(departing_students) > project.min_group_size for group in project.groups)
            ]
            if not candidate_projects:
                raise ValueError("No student can be moved!")
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group
                for group in chosen_project.groups
                if group.remaining_size(departing_students) > chosen_project.min_group_size
            ]
            chosen_group = rd.choice(candidate_groups)
            student_to_move = rd.choice(chosen_group.remaining_students(departing_students))
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
            max(0, project.num_groups() - project.offered_num_groups) * project.pen_groups for project in self.projects
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
                # I currently do not update the student.unassigned parameter at later points.
                unassigned_students = [
                    student for student in project_waitlists[project.project_id] if not student.assigned
                ]
                if (
                    project.num_groups() < project.offered_num_groups
                    and len(unassigned_students) >= project.ideal_group_size
                ):
                    now_assigned_students = unassigned_students[: project.ideal_group_size]
                    project.add_initial_group_ideal_size(now_assigned_students)
                    for student in now_assigned_students:
                        student.assigned = True
                    num_students_unassigned -= project.ideal_group_size
                    any_group_added = True

        self.unassigned = [student for student in self.students if not student.assigned]

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
        print(f"Number of Projects: {len(self.projects)}\nNumber of Students: {len(self.students)}")

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
        problem_instance = generate_throwaway_instance(num_projects=3, num_students=30)

    vns_run = VarNeighborhoodSearch(
        problem_instance,
        reward_bilateral=2,
        penalty_non_assignment=3,
    )
    vns_run.report_input_data()
    vns_run.report_num_projects_and_students()
    vns_run.report_current_solution()
    vns_run.run_gen_vns_best_improvement(18, 10, 0.05)
    vns_run.report_current_solution()
    vns_run.calculate_current_objective_value()
    print("The objective after complete recalculation:", vns_run.objective_value)
    print("The list of unassigned students:", vns_run.unassigned)
    for neigborhood, num_visits in vns_run.neigborhood_visit_counter.items():
        print(f"{neigborhood}: {num_visits}")


# parameters with vnd introduction:  50, 10, 0.05 1000, 10, 0.05  Fehler mit vnd:  16, 10, 0.05 3869
# Improve at last: 40, 10, 0.05 neighborhood 6
