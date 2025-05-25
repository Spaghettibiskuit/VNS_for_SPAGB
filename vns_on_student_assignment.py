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

# rd.seed(100)
# rd.seed(3869)


class VariableNeighborhoodSearch:
    """Applies VNS on a student assignment problem."""

    def __init__(
        self,
        problem_data: tuple[pd.DataFrame, pd.DataFrame],
        reward_bilateral_interest_collaboration: int,
        penalty_student_not_assigned: int,
    ):
        self.projects_info, self.students_info = problem_data
        self.reward_bilateral_interest_collaboration = reward_bilateral_interest_collaboration
        self.penalty_student_not_assigned = penalty_student_not_assigned
        self.projects = tuple((Project(*row) for row in self.projects_info.itertuples()))
        self.students = tuple((Student(*row) for row in self.students_info.itertuples()))
        self.unassigned_students: list[Student] = []
        self.objective_value = 0
        self.neigborhood_visit_counter = {}
        self._initial_solution()
        self.calculate_current_objective_value()
        self.best_objective_value = self.objective_value
        self.move_reversals = []

    def run_general_vns_best_improvement(
        self,
        iteration_limit: int,
        max_neighborhood: int,
        assignment_bias: float | int,
        unassignment_probability: float,
        min_neighborhood: int = 1,
    ):
        if not 0 <= unassignment_probability <= 1:
            raise ValueError("A probability must be between 0 and 1.")
        self.neigborhood_visit_counter = {
            neighborhood: 0 for neighborhood in range(min_neighborhood, max_neighborhood + 1)
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
                unassignment_probability,
            )

            print("The objective value after the shake is:", self.objective_value)

            self._var_neigborhood_descent(num_to_move, across_projects)

            curr_obj_val = (
                self._sum_preferences()
                + self._sum_join_rewards()
                - self._sum_missing_assignment_penalties()
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

                if current_neighborhood == max_neighborhood:
                    current_neighborhood = min_neighborhood
                else:
                    current_neighborhood += 1

            self.move_reversals = []

    def _shake(
        self,
        num_to_move: int,
        across_projects: bool,
        assignment_bias: float | int,
        unassignment_probability: float,
    ):

        shake_departures = self._shake_departures(num_to_move, assignment_bias)
        departure_deltas = 0
        arrival_deltas = 0
        for shake_departure in shake_departures:
            departure_deltas += self._calculate_leaving_delta(shake_departure)
            shake_arrival = self._shake_arrival(shake_departure, across_projects, unassignment_probability)
            arrival_deltas += self._calculate_arrival_delta(shake_arrival)
            self._move_student(shake_departure, shake_arrival)
            self.move_reversals.append((shake_arrival, shake_departure))

        self.objective_value += departure_deltas + arrival_deltas

    def _var_neigborhood_descent(self, max_to_move: int, across_projects: bool, min_to_move: int = 1):
        destinations, locations_students_by_id = self._prerequisites_vnd()

        num_to_move = min_to_move
        while num_to_move <= max_to_move:
            best_moves_local, delta = self._local_search_best_improvement(
                destinations, locations_students_by_id, across_projects, num_to_move
            )
            if delta > 0:
                self.objective_value += delta
                for best_move_local in best_moves_local:
                    departure, arrival = best_move_local
                    if departure[-1] is not arrival[-1]:
                        raise ValueError("No clarity which student to be moved.")

                    self._move_student(departure, arrival)
                    self.move_reversals.append((arrival, departure))

                    if arrival[0] is not self.unassigned_students:
                        arrival: tuple[Project, ProjectGroup, Student]
                        project, group, student = arrival
                        locations_students_by_id[student.student_id] = (
                            project,
                            group,
                        )

                    else:
                        student = arrival[-1]
                        locations_students_by_id[student.student_id] = self.unassigned_students

                num_to_move = min_to_move

            else:
                num_to_move += 1

    def _local_search_best_improvement(
        self,
        destinations: tuple[tuple[Project, ProjectGroup] | list[Student]],
        locations_students_by_id: dict[int : tuple[Project, ProjectGroup] | list[Student]],
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

        combinations_student_ids = tuple(it.combinations(range(len(self.students)), num_to_move))
        all_ordered_n_tuples_destinations = tuple(it.product(destinations, repeat=num_to_move))

        for combination_ids in combinations_student_ids:
            if not self._all_in_combination_can_leave(combination_ids, locations_students_by_id):
                continue

            corresponding_departures = tuple(
                (
                    (
                        (
                            *student_location,
                            self.students[student_id],
                        )
                        if (student_location := locations_students_by_id[student_id]) is not self.unassigned_students
                        else (self.unassigned_students, self.students[student_id])
                    )
                    for student_id in combination_ids
                )
            )

            ordered_n_tuples_destinations_corresponding_student_moving = filter(
                lambda ordered_n_tuple_destinations, combination_ids=combination_ids: all(
                    (
                        locations_students_by_id[combination_ids[i]][1] is not ordered_n_tuple_destinations[i][1]
                        if (locations_students_by_id[combination_ids[i]] is not self.unassigned_students)
                        and (ordered_n_tuple_destinations[i] is not self.unassigned_students)
                        else locations_students_by_id[combination_ids[i]] is not ordered_n_tuple_destinations[i]
                    )
                    for i in range(num_to_move)
                ),
                all_ordered_n_tuples_destinations,
            )

            if across_projects:
                best_arrival_combination, delta = self._find_best_moves_combination(
                    ordered_n_tuples_destinations_corresponding_student_moving,
                    corresponding_departures,
                )

            else:
                ordered_n_tuples_destinations_also_moves_only_within_project = filter(
                    lambda ordered_n_tuple_destinations, combination_ids=combination_ids: all(
                        (
                            (locations_students_by_id[combination_ids[i]][0] is ordered_n_tuple_destinations[i][0])
                            if (locations_students_by_id[combination_ids[i]] is not self.unassigned_students)
                            and (ordered_n_tuple_destinations[i] is not self.unassigned_students)
                            else True
                        )
                        for i in range(num_to_move)
                    ),
                    ordered_n_tuples_destinations_corresponding_student_moving,
                )

                best_arrival_combination, delta = self._find_best_moves_combination(
                    ordered_n_tuples_destinations_also_moves_only_within_project,
                    corresponding_departures,
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
        locations_students_by_id: dict[int : tuple[Project, ProjectGroup] | list[Student]],
    ):
        locations_students_combination = [locations_students_by_id[student_id] for student_id in combination_ids]
        for location_student in locations_students_combination:
            if location_student is not self.unassigned_students:
                num_occurences = locations_students_combination.count(location_student)
                location_student: tuple[Project, ProjectGroup]
                project, group = location_student
                if group.size() - num_occurences < project.min_group_size:
                    return False
        return True

    def _find_best_moves_combination(
        self,
        ordered_n_tuples_destinations: tuple[tuple[tuple[Project, ProjectGroup] | list[Student]]],
        corresponding_departures: tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]],
    ) -> tuple[
        tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]],
        int,
    ]:
        best_arrival_combination = None
        best_delta_combination = -10e6

        for ordered_n_tuple_destinations in ordered_n_tuples_destinations:
            delta_combination = 0
            invalid_destination = False
            reversal_moves = []
            for i, destination in enumerate(ordered_n_tuple_destinations):
                corresponding_departure = corresponding_departures[i]

                student = corresponding_departure[-1]

                delta_combination += self._calculate_leaving_delta(corresponding_departure)

                if destination is self.unassigned_students:
                    arrival = (self.unassigned_students, student)
                else:
                    destination: tuple[Project, ProjectGroup]
                    project, group = destination
                    if group.size() == project.max_group_size:
                        invalid_destination = True
                        break
                    arrival = (project, group, student)
                delta_combination += self._calculate_arrival_delta(arrival)
                self._move_student(corresponding_departure, arrival)
                reversal_moves.append((arrival, corresponding_departure))

            if invalid_destination:
                for move in reversal_moves:
                    self._move_student(*move)
                continue

            if delta_combination > best_delta_combination:
                best_delta_combination = delta_combination
                best_arrival_combination = tuple(
                    (
                        (*destination, corresponding_departures[i][-1])
                        if destination is not self.unassigned_students
                        else (self.unassigned_students, corresponding_departures[i][-1])
                    )
                    for i, destination in enumerate(ordered_n_tuple_destinations)
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
            self.unassigned_students,
        )

        locations_students_by_id = {
            student.student_id: (project, group)
            for project in self.projects
            for group in project.groups
            for student in group.students
        }

        for student in self.unassigned_students:
            locations_students_by_id[student.student_id] = self.unassigned_students

        return (destinations, locations_students_by_id)

    def _move_student(
        self,
        departure_specifications: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
        arrival_specifications: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
    ) -> None:
        if (moving_student := departure_specifications[-1]) is not arrival_specifications[-1]:
            raise ValueError("No clarity which student should be moved!")
        student_was_unassigned = departure_specifications[0] is self.unassigned_students
        student_will_be_unassigned = arrival_specifications[0] is self.unassigned_students
        if student_was_unassigned and student_will_be_unassigned:
            return

        if student_was_unassigned and not student_will_be_unassigned:
            self.unassigned_students.remove(moving_student)
            arrival_group = arrival_specifications[-2]
            arrival_group.accept_student(moving_student)
            return
        if not student_was_unassigned and student_will_be_unassigned:
            departure_group = departure_specifications[-2]
            departure_group.release_student(moving_student)
            self.unassigned_students.append(moving_student)
            return
        departure_group = departure_specifications[-2]
        arrival_group = arrival_specifications[-2]
        departure_group.release_student(moving_student)
        arrival_group.accept_student(moving_student)

    def _calculate_arrival_delta(
        self,
        shake_arrival: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
    ) -> int:
        if shake_arrival[0] is self.unassigned_students:
            return -self.penalty_student_not_assigned
        arrival_project, arrival_group, arriving_student = shake_arrival
        preference_gain = arriving_student.preference_value(arrival_project)
        bilateral_reward_gain = sum(
            self.reward_bilateral_interest_collaboration
            for student in arrival_group.students
            if student.student_id in arriving_student.fav_partners
            and arriving_student.student_id in student.fav_partners
        )
        # Establishment of a new group not yet implemented.
        if arrival_group.size() < arrival_project.ideal_group_size:
            delta_group_size = arrival_project.penalty_deviation_from_ideal_group_size
        else:
            delta_group_size = -arrival_project.penalty_deviation_from_ideal_group_size
        return preference_gain + bilateral_reward_gain + delta_group_size

    def _shake_arrival(
        self,
        shake_departure: tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
        across_projects: bool,
        unassignment_probability: float,
    ) -> tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]:
        if shake_departure[0] is self.unassigned_students:
            student = shake_departure[-1]
            candidate_projects = [
                project
                for project in self.projects
                if any(group.size() < project.max_group_size for group in project.groups)
            ]
            if not candidate_projects:
                return (self.unassigned_students, student)
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group for group in chosen_project.groups if group.size() < chosen_project.max_group_size
            ]
            chosen_group = rd.choice(candidate_groups)
            return (chosen_project, chosen_group, student)

        if rd.random() < unassignment_probability:
            student = shake_departure[-1]
            return (self.unassigned_students, student)

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
                return (self.unassigned_students, student)

            chosen_project = rd.choice(candidate_projects)

        candidate_groups = [
            group
            for group in chosen_project.groups
            if group is not current_group and group.size() < chosen_project.max_group_size
        ]

        if not candidate_groups:
            return (self.unassigned_students, student)
        chosen_group = rd.choice(candidate_groups)
        return (chosen_project, chosen_group, student)

    def _calculate_leaving_delta(
        self,
        departure_point: tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
    ) -> int:
        if departure_point[0] is self.unassigned_students:
            return self.penalty_student_not_assigned
        project, group, student = departure_point
        preference_loss = student.preference_value(project)
        bilateral_reward_loss = sum(
            self.reward_bilateral_interest_collaboration
            for bilateral_preference in group.bilateral_preferences
            if student.student_id in bilateral_preference
        )
        # currently no mechanism for deleting groups.
        # if project.num_groups() > project.offered_num_groups and group.size() == 1:
        #     reduced_penalty_num_groups = project.penalty_extra_group
        # else:
        #     reduced_penalty_num_groups = 0
        if group.size() > project.ideal_group_size:
            delta_group_size = project.penalty_deviation_from_ideal_group_size
        else:
            delta_group_size = -project.penalty_deviation_from_ideal_group_size
        return (
            -preference_loss
            - bilateral_reward_loss
            # + reduced_penalty_num_groups
            + delta_group_size
        )

    def _shake_departures(
        self, num_to_move: int, assignment_bias: float | int
    ) -> list[tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student]]:
        departures_specifications = []
        departing_students: list[Student] = []
        for i in range(num_to_move):
            unassigned_students_remaining = [
                student for student in self.unassigned_students if student not in departing_students
            ]

            if unassigned_students_remaining and (
                rd.random() < len(unassigned_students_remaining) / len(self.students) * assignment_bias
            ):
                student_to_move: Student = rd.choice(unassigned_students_remaining)
                departing_students.append(student_to_move)
                departures_specifications.append(
                    (
                        self.unassigned_students,
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
                if unassigned_students_remaining:
                    students_to_move = rd.choices(
                        unassigned_students_remaining, k=min(len(unassigned_students_remaining), num_to_move - i)
                    )
                    departures_specifications += [(self.unassigned_students, student) for student in students_to_move]
                break

            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group
                for group in chosen_project.groups
                if group.remaining_size(departing_students) > chosen_project.min_group_size
            ]
            chosen_group = rd.choice(candidate_groups)
            student_to_move = rd.choice(chosen_group.remaining_students(departing_students))
            departing_students.append(student_to_move)
            departures_specifications.append(
                (
                    chosen_project,
                    chosen_group,
                    student_to_move,
                )
            )
        return departures_specifications

    def calculate_current_objective_value(self):
        self.objective_value = (
            self._sum_preferences()
            + self._sum_join_rewards()
            - self._sum_missing_assignment_penalties()
            - self._sum_group_surplus_penalties()
            - self._sum_group_size_penalties()
        )

    def _sum_preferences(self):
        return sum(
            student.preference_value(project)
            for project in self.projects
            for group in project.groups
            for student in group.students
        )

    def _sum_join_rewards(self):
        for project in self.projects:
            for group in project.groups:
                group.populate_bilateral_preferences_set()
        return sum(
            self.reward_bilateral_interest_collaboration * len(group.bilateral_preferences)
            for project in self.projects
            for group in project.groups
        )

    def _sum_missing_assignment_penalties(self):
        return len(self.unassigned_students) * self.penalty_student_not_assigned

    def _sum_group_surplus_penalties(self):
        return sum(
            max(0, project.num_groups() - project.offered_num_groups) * project.penalty_extra_group
            for project in self.projects
        )

    def _sum_group_size_penalties(self):
        return sum(
            abs(group.size() - project.ideal_group_size) * project.penalty_deviation_from_ideal_group_size
            for project in self.projects
            for group in project.groups
        )

    def _initial_solution(self):
        projects_waitlists = self._build_initial_projects_waitlists()
        assigned_students = []
        any_group_added = True
        num_students = len(self.students)
        while len(assigned_students) < num_students and any_group_added:
            any_group_added = False
            for project in self.projects:
                unassigned_descending_preference = [
                    student for student in projects_waitlists[project.project_id] if student not in assigned_students
                ]
                if (
                    project.num_groups() < project.offered_num_groups
                    and len(unassigned_descending_preference) >= project.ideal_group_size
                ):
                    now_assigned_students = unassigned_descending_preference[: project.ideal_group_size]
                    project.add_initial_group_ideal_size(now_assigned_students)
                    assigned_students += now_assigned_students
                    any_group_added = True

        self.unassigned_students = [student for student in self.students if student not in assigned_students]

    def _build_initial_projects_waitlists(self) -> dict[int, list[Student]]:
        return {
            project.project_id: sorted(
                self.students,
                key=lambda student, project=project: student.preference_value(project),
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
        for student in self.unassigned_students:
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

    vns_run = VariableNeighborhoodSearch(
        problem_instance,
        reward_bilateral_interest_collaboration=2,
        penalty_student_not_assigned=3,
    )
    vns_run.report_input_data()
    vns_run.report_num_projects_and_students()
    vns_run.report_current_solution()
    vns_run.run_general_vns_best_improvement(20, 4, 10, 0.05)
    vns_run.report_current_solution()
    vns_run.calculate_current_objective_value()
    print("The objective after complete recalculation:", vns_run.objective_value)
    print("The list of unassigned students:", vns_run.unassigned_students)
    for neigborhood, num_visits in vns_run.neigborhood_visit_counter.items():
        print(f"{neigborhood}: {num_visits}")


# parameters with vnd introduction:  50, 10, 0.05 1000, 10, 0.05  Fehler mit vnd:  16, 10, 0.05 3869
# Improve at last: 40, 10, 0.05 neighborhood 6
