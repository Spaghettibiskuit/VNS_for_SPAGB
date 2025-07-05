"""Contains the class which controls the entire VNS solving process."""

import itertools as it
import random as rd
import time as t
from collections import Counter
from collections.abc import Iterator

import pandas as pd

from project import Project
from project_group import ProjectGroup
from student import Student


class VariableNeighborhoodSearch:
    """Solves solutions for instances of the SPAGP.

    SPAGP := Student-Project-Allocation with Group Building Problem

    Capable of solving instances, but also checking solutions for
    their validity and checking whether the objective has been
    calculated correctly.

    Attributes:
        projects_info: The names of the projects with each project's
            guidelines, whishes and penalties regarding the number of
            groups and group sizes.
        students_info: The project preferences for all projects and the
            partner preferences i.e., the students a student wants to
            work with the most for all students in the problem instance.
        reward_bilateral_interest_collaboration: The fixed reward for every
            occurrence in the solution of two students who have specified
            each other as partner preferences being in the same group.
        penalty_student_not_assigned: The fixed penalty for every student
            in the solution who is not assigned to a group.
        projects: All instances of Project in the problem instance.
        students: All instances of Student in the problem instance.
        time_data_loaded: The point in time when all data was loaded.
        num_students: The number of students in the problem instance.
        unassigned_students: The students who are not assigned to a
            group at any point in time.
        bilateral_pairs: A set of tuples with two student IDS belonging
            to two students who specified each other as partner
            preferences i.e., one of the students they want to work
            with the most.
        objective_value: The objective value of the solution at any
            point in time.
        best_objective_value: The objective value of the best solution
            found thus far at any point in time.
        move_reversals: Moves of students made during a try to improve
            the best_objective_value which have to be reversed if the
            try fails.
        combinations_student_ids_by_dimension: combinations of student IDs
            with different numbers of IDs per combination which are used
            during local search.
    """

    def __init__(
        self,
        projects_info: pd.DataFrame,
        students_info: pd.DataFrame,
        reward_bilateral_interest_collaboration: int = 2,
        penalty_student_not_assigned: int = 3,
    ):
        """Initializes the instance with the data of the problem instance.

        Args:
            projects_info: The names of the projects with each project's
                guidelines, whishes and penalties regarding the number of
                groups and group sizes.
            students_info: The project preferences for all projects and the
                partner preferences i.e., the students a student wants to
                work with the most for all students in the problem instance.
            reward_bilateral_interest_collaboration: The fixed reward for every
                occurrence in the solution of two students who have specified
                each other as partner preferences being in the same group.
            penalty_student_not_assigned: The fixed penalty for every student
                in the solution who is not assigned to a group.
        """
        self.projects_info, self.students_info = projects_info, students_info
        self.reward_bilateral_interest_collaboration = reward_bilateral_interest_collaboration
        self.penalty_student_not_assigned = penalty_student_not_assigned
        self.projects = tuple((Project(*row) for row in self.projects_info.itertuples()))
        self.students = tuple((Student(*row) for row in self.students_info.itertuples()))
        self.time_data_loaded = t.time()
        self.num_students = len(self.students)
        self.unassigned_students: list[Student] = []
        self.bilateral_pairs = self._get_bilateral_pairs()
        self._initial_solution()
        self.objective_value = self.current_objective_value()
        self.best_objective_value = self.objective_value
        self.move_reversals = []
        self.combinations_student_ids_by_dimension = {1: list(it.combinations(range(self.num_students), 1))}

    def _get_bilateral_pairs(self) -> set[tuple[int, int]]:
        favorite_partners_students = self.students_info["fav_partners"].tolist()
        return {
            (student_id, partner_id)
            for student_id, favorite_partners in enumerate(favorite_partners_students)
            for partner_id in favorite_partners
            if student_id < partner_id and student_id in favorite_partners_students[partner_id]
        }

    def run_general_vns_best_improvement(
        self,
        max_neighborhood: int = 6,
        assignment_bias: float | int = 10,
        unassignment_probability: float = 0.05,
        min_neighborhood: int = 1,
        testing: bool = False,
        benchmarking: bool = False,
        demonstrating: bool = False,
        iteration_limit: int = 40,
        time_limit: int = 300,
        seed: int | None = None,
    ):
        """Solves or tests the General VNS solution process.

        Performs an heuristic problem-solving algorithm based on the
        philosophy of the General Variable Neighborhood Search.

        As long as its main loop does not break, the algorithm continuously tries
        to improve the incumbent solution based on a different
        defined neighborhoods.

        In each neighborhood the solution is first perturbed and then this perturbed
        solution is improved with Variable Neighborhood Descent (VND). Perturbation is
        done in two ways. The first way is to randomly move students. This means changing
        the group a student is assigned to, unassign him, or assign him to any group if
        he was unassigned before. This way of perturbation is called "shake".
        The second way is to found or dissolve a group.

        The neighborhoods are numbered from 1-6. If no improvement is found in one
        neighborhood, the neighborhood with the next biggest number is tried in the
        next iteration in the loop. If no improvement is found in the neighborhood
        with the biggest number, neighborhood with the smallest number is tried again
        and so forth. If an improvement is found in any neighborhood, the neighborhood
        with the smallest number is visited next.

        Args:
            max_neighborhood: The biggest neighborhood by number that is visited.
            assignment bias: How much more likely it is for an unassigned student
                to be moved during the shake compared to an assigned student.
            unassignment_probability: Probability that an assigned student that is
                moved during the shake is unassigned instead of randomly moved into
                another group.
            min_neighborhood: The smallest neighborhood by number that is visited.
            testing: Set to True if it should be checked after every shake, every
                founding or dissolving of a group and every VND if the solution is
                valid and the objective is calculated correctly. If anything goes
                wrong, an error report is returned.
            benchmarking: Set to True if every solution that improves upon the best
                found before should be saved in a dictionary in a list of dictionaries
                which is returned once the time limit is reached.
            demonstrating: Set to True if information on the solution process should
                be printed in the terminal.
            iteration_limit: The number of iterations before solving stops if
                benchmarking is not set to True
            time_limit: The time after which the main loop breaks if benchmarking is
                set to True
            seed: The random seed.

        Returns:
            If testing == True: If after any shake, founding or dissolving of a group
            or VND either the objective is not correctly calculated or the solution
            is not valid an error report is returned detailing what went wrong and at
            what point of the solution process.

            if benchmarking == True: List of dictionaries with solutions which improved
            on the best found solution before. Each dictionary includes the objective
            value, runtime until the solution was found and the neighborhood in which it
            was found. Neighborhood 0 is the initial solution.

        Raises:
            ValueError: unassignment_probability is not between 0 and 1.
        """
        if seed != None:
            rd.seed(seed)
        if not 0 <= unassignment_probability <= 1:
            raise ValueError("A probability must be between 0 and 1.")
        if benchmarking:
            best_solutions = [
                {"obj": self.best_objective_value, "runtime": t.time() - self.time_data_loaded, "neighborhood": 0}
            ]
        current_iteration = 0
        current_neighborhood = min_neighborhood
        while True:
            current_iteration += 1
            match current_neighborhood:
                case 1:
                    num_to_move = 1
                    across_projects = False
                    shake = True
                    found_or_dissolve_group = False
                case 2:
                    num_to_move = 2
                    across_projects = False
                    shake = True
                    found_or_dissolve_group = False
                case 3:
                    num_to_move = 2
                    across_projects = False
                    shake = False
                    found_or_dissolve_group = True
                case 4:
                    num_to_move = 1
                    across_projects = True
                    shake = True
                    found_or_dissolve_group = False
                case 5:
                    num_to_move = 2
                    across_projects = True
                    shake = True
                    found_or_dissolve_group = False
                case 6:
                    num_to_move = 2
                    across_projects = True
                    shake = False
                    found_or_dissolve_group = True

            if demonstrating:
                print(f"\nIteration {current_iteration}/{iteration_limit}")
                print("The incumbent best objective value is:", self.best_objective_value)
                print("The neighborhood is:", current_neighborhood)
                start_time = t.time()

            if found_or_dissolve_group:
                self._found_or_dissolve_one_group()
                if testing:
                    if error_report := self.check_solution():
                        error_report["iteration"] = current_iteration
                        error_report["neighborhood"] = current_neighborhood
                        error_report["point"] = "founding or dissolution"
                        return error_report
                if demonstrating:
                    print("The objective value after founding or dissoving a group is:", self.objective_value)

            if shake:
                self._shake(
                    num_to_move,
                    across_projects,
                    assignment_bias,
                    unassignment_probability,
                )
                if testing:
                    if error_report := self.check_solution():
                        error_report["iteration"] = current_iteration
                        error_report["neighborhood"] = current_neighborhood
                        error_report["point"] = "shake"
                        return error_report
                if demonstrating:
                    print("The objective value after shaking is:", self.objective_value)

            self._variable_neighborhood_descent(num_to_move, across_projects)
            if testing:
                if error_report := self.check_solution():
                    error_report["iteration"] = current_iteration
                    error_report["neighborhood"] = current_neighborhood
                    error_report["point"] = "VND"
                    return error_report
            if demonstrating:
                print("The objective value after VND is:", self.objective_value)
                print(f"Visiting the neighborhood took {t.time() - start_time} seconds.")

            if self.objective_value > self.best_objective_value:
                self.best_objective_value = self.objective_value
                if benchmarking:
                    best_solutions.append(
                        {
                            "obj": self.best_objective_value,
                            "runtime": t.time() - self.time_data_loaded,
                            "neighborhood": current_neighborhood,
                        }
                    )
                current_neighborhood = min_neighborhood
            else:
                for move_reversal in reversed(self.move_reversals):
                    self._move_student(*move_reversal)
                self.objective_value = self.best_objective_value

                if current_neighborhood >= max_neighborhood:
                    current_neighborhood = min_neighborhood
                else:
                    current_neighborhood += 1

            self.move_reversals = []
            for project in self.projects:
                project.groups = [group for group in project.groups if group.students]

            if benchmarking and t.time() - self.time_data_loaded > time_limit:
                break
            if not benchmarking and current_iteration >= iteration_limit:
                break

        if benchmarking:
            return best_solutions

    def check_solution(self) -> dict:
        """Returns any errors in the solution process.

        If the solution is invalid it is specified why. If the objective
        value calculated with deltas deviates from the one obtained by
        complete recalculation, both are specified.
        """
        errors_validity = self._check_validity()
        error_objective_value = self._check_objective_value()
        return errors_validity | error_objective_value

    def _check_validity(self):
        errors_validity = {}
        groups_too_small = [
            f"Group {project.groups.index(group)} in project {project.name} {project.project_id} "
            f"has {group.size()} students. The minimum is {project.min_group_size}"
            for project in self.projects
            for group in project.groups
            if group.students and group.size() < project.min_group_size
        ]
        if groups_too_small:
            errors_validity["groups_too_small"] = groups_too_small
        groups_too_big = [
            f"Group {project.groups.index(group)} in project {project.name} {project.project_id} "
            f"has {group.size()} students. The maximum is {project.max_group_size}"
            for project in self.projects
            for group in project.groups
            if group.size() > project.max_group_size
        ]
        if groups_too_big:
            errors_validity["groups_too_big"] = groups_too_big
        projects_too_many_groups = [
            f"{project.name} {project.project_id} has {project.num_non_empty_groups()} "
            f"groups. Allowed are {project.max_num_groups}."
            for project in self.projects
            if project.num_non_empty_groups() > project.max_num_groups
        ]
        if projects_too_many_groups:
            errors_validity["too_many_groups"] = projects_too_many_groups
        students_anywhere = sorted(
            [student for project in self.projects for group in project.groups for student in group.students]
            + self.unassigned_students,
            key=lambda student: student.student_id,
        )
        if not all(a is b for a, b in zip(self.students, students_anywhere)):
            errors_validity["inconsistency_students"] = True
        return errors_validity

    def _check_objective_value(self):
        if self.objective_value != (actual_objective_value := self.current_objective_value()):
            return {"claimed_obj": self.objective_value, "actual_obj": actual_objective_value}
        return {}

    def _found_or_dissolve_one_group(self):
        founding_options_moves_and_deltas = self._get_founding_options()
        dissolution_options_moves_and_deltas = self._get_dissolution_options()
        all_options_moves_and_deltas = founding_options_moves_and_deltas + dissolution_options_moves_and_deltas
        max_delta = max(all_options_moves_and_deltas, key=lambda moves_and_delta: moves_and_delta[-1])[-1]
        max_delta_options_moves = [
            option_moves_and_delta[0]
            for option_moves_and_delta in all_options_moves_and_deltas
            if option_moves_and_delta[-1] == max_delta
        ]
        moves = rd.choice(max_delta_options_moves)
        for move in moves:
            self._move_student(*move)

        self.move_reversals += [move[::-1] for move in moves]
        self.objective_value += max_delta

    def _get_founding_options(
        self,
    ) -> list[
        tuple[
            list[
                tuple[
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                ]
            ],
            int,
        ]
    ]:
        projects_applicable_for_group_founding = (
            project for project in self.projects if project.num_groups() < project.max_num_groups
        )
        founding_options = []

        for project in projects_applicable_for_group_founding:
            new_group, founding_delta = project.get_new_empty_group_and_initial_delta()
            moves_made = []

            while new_group.size() < project.max_group_size:
                addition_options_departures = [
                    (project, group, student)
                    for project in self.projects
                    for group in project.groups
                    for student in group.students
                    if group.size() > project.min_group_size and group is not new_group
                ]
                addition_options_departures += [
                    (self.unassigned_students, student) for student in self.unassigned_students
                ]
                if not addition_options_departures:
                    break
                addition_options_leaving_deltas = (
                    self._calculate_leaving_delta(addition_option) for addition_option in addition_options_departures
                )
                addition_options_arrivals = [
                    (project, new_group, addition_option_departure[-1])
                    for addition_option_departure in addition_options_departures
                ]
                addition_options_arrival_deltas = (
                    self._calculate_arrival_delta(addition_option_arrival)
                    for addition_option_arrival in addition_options_arrivals
                )
                addition_options_deltas = [
                    leaving_delta + arrival_delta
                    for leaving_delta, arrival_delta in zip(
                        addition_options_leaving_deltas, addition_options_arrival_deltas
                    )
                ]
                max_addition_delta = max(addition_options_deltas)
                if max_addition_delta < 0 and new_group.size() >= project.min_group_size:
                    break
                max_addition_delta_moves = [
                    addition_move
                    for i, addition_move in enumerate(zip(addition_options_departures, addition_options_arrivals))
                    if addition_options_deltas[i] == max_addition_delta
                ]
                addition_move = rd.choice(max_addition_delta_moves)
                self._move_student(*addition_move)
                founding_delta += max_addition_delta
                moves_made.append(addition_move)

            if new_group.size() >= project.min_group_size:
                founding_options.append((moves_made, founding_delta))
            for move in moves_made:
                self._move_student(*move[::-1])

        return founding_options

    def _get_dissolution_options(self) -> list[
        tuple[
            list[
                tuple[
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                ]
            ],
            int,
        ]
    ]:
        dissolution_options = []
        dissolution_candidates = (
            (project, group) for project in self.projects for group in project.groups if group.students
        )
        destinations_with_free_capacity = [
            (project, group)
            for project in self.projects
            for group in project.groups
            if group.students and group.size() < project.max_group_size
        ]

        for project, group in dissolution_candidates:
            project: Project
            group: ProjectGroup
            moves_made = []
            dissolution_delta = self._initial_dissolution_delta(project, group)
            arrivals_with_deltas = [
                ((*destination, student), self._calculate_arrival_delta((*destination, student)))
                for destination in destinations_with_free_capacity
                for student in group.students
                if not all(x is y for x, y in zip(destination, (project, group)))
            ]
            arrivals_with_deltas += [
                ((self.unassigned_students, student), -self.penalty_student_not_assigned) for student in group.students
            ]
            while group.students:
                max_delta = max(arrivals_with_deltas, key=lambda arrival_with_delta: arrival_with_delta[-1])[-1]
                max_delta_arrivals = [
                    arrival_with_delta[0]
                    for arrival_with_delta in arrivals_with_deltas
                    if arrival_with_delta[-1] == max_delta
                ]
                arrival = rd.choice(max_delta_arrivals)
                move = ((project, group, (student_in_move := arrival[-1])), arrival)
                self._move_student(*move)
                dissolution_delta += max_delta
                moves_made.append(move)
                if arrival[0] is self.unassigned_students:
                    arrivals_with_deltas = [
                        arrival_with_delta
                        for arrival_with_delta in arrivals_with_deltas
                        if arrival_with_delta[0][-1] is not student_in_move
                    ]
                    continue
                project_student_moved_to, group_student_moved_to, _ = arrival
                if group_student_moved_to.size() >= project_student_moved_to.max_group_size:
                    arrivals_with_deltas = [
                        arrival_with_delta
                        for arrival_with_delta in arrivals_with_deltas
                        if (arrival_with_delta[0][-1] is not student_in_move)
                        and (arrival_with_delta[0][1] is not group_student_moved_to)
                    ]
                    continue
                arrivals_with_deltas = [
                    (
                        (arrival, self._calculate_arrival_delta(arrival))
                        if (arrival := arrival_with_delta[0])[1] is group_student_moved_to
                        else arrival_with_delta
                    )
                    for arrival_with_delta in (
                        arrival_with_delta
                        for arrival_with_delta in arrivals_with_deltas
                        if arrival_with_delta[0][-1] is not student_in_move
                    )
                ]

            for move in moves_made:
                self._move_student(*move[::-1])
            dissolution_options.append((moves_made, dissolution_delta))

        return dissolution_options

    def _initial_dissolution_delta(self, project: Project, group: ProjectGroup) -> int:
        preference_loss = sum(student.preference_value(project) for student in group.students)
        bilateral_reward_loss = sum(
            self.reward_bilateral_interest_collaboration
            for id_pair in it.combinations(sorted([student.student_id for student in group.students]), 2)
            if id_pair in self.bilateral_pairs
        )
        reward_one_less_group = (
            project.penalty_extra_group if project.num_non_empty_groups() > project.offered_num_groups else 0
        )
        reward_removal_group_not_ideal_size = (
            abs(group.size() - project.ideal_group_size) * project.penalty_deviation_from_ideal_group_size
        )
        return -preference_loss - bilateral_reward_loss + reward_one_less_group + reward_removal_group_not_ideal_size

    def _shake(
        self,
        num_to_move: int,
        across_projects: bool,
        assignment_bias: float | int,
        unassignment_probability: float,
    ):

        shake_departures = self._shake_departures(num_to_move, assignment_bias)
        sum_departure_deltas = 0
        sum_arrival_deltas = 0
        for shake_departure in shake_departures:
            sum_departure_deltas += self._calculate_leaving_delta(shake_departure)
            shake_arrival = self._shake_arrival(shake_departure, across_projects, unassignment_probability)
            sum_arrival_deltas += self._calculate_arrival_delta(shake_arrival)
            self._move_student(shake_departure, shake_arrival)
            self.move_reversals.append((shake_arrival, shake_departure))

        self.objective_value += sum_departure_deltas + sum_arrival_deltas

    def _variable_neighborhood_descent(self, max_to_move: int, across_projects: bool, min_to_move: int = 1):
        destinations = [(project, group) for project in self.projects for group in project.groups if group.students]
        destinations.append(self.unassigned_students)

        if max_to_move > (highest_saved_dimension := max(self.combinations_student_ids_by_dimension.keys())):
            for num_in_combination in range(highest_saved_dimension + 1, max_to_move + 1):
                self.combinations_student_ids_by_dimension[num_in_combination] = list(
                    it.combinations(range(self.num_students), num_in_combination)
                )

        ordered_n_tuples_destinations_by_dimension = {
            n: list(it.product(destinations, repeat=n)) for n in range(min_to_move, max_to_move + 1)
        }

        locations_students_by_id = {
            student.student_id: (project, group)
            for project in self.projects
            for group in project.groups
            for student in group.students
        }
        for student in self.unassigned_students:
            locations_students_by_id[student.student_id] = self.unassigned_students

        num_to_move = min_to_move
        while num_to_move <= max_to_move:
            best_moves_local, delta = self._local_search_best_improvement(
                ordered_n_tuples_destinations_by_dimension[num_to_move],
                locations_students_by_id,
                across_projects,
                num_to_move,
            )
            if delta > 0:
                self.objective_value += delta
                for departure, arrival in best_moves_local:
                    self._move_student(departure, arrival)

                    if arrival[0] is not self.unassigned_students:
                        project, group, student = arrival
                        locations_students_by_id[student.student_id] = (project, group)

                    else:
                        student = arrival[-1]
                        locations_students_by_id[student.student_id] = self.unassigned_students
                self.move_reversals += [move[::-1] for move in best_moves_local]
                num_to_move = min_to_move

            else:
                num_to_move += 1

    def _local_search_best_improvement(
        self,
        ordered_n_tuples_destinations: list[tuple[tuple[Project, ProjectGroup] | list[Student]]],
        locations_students_by_id: dict[int : tuple[Project, ProjectGroup] | list[Student]],
        across_projects: bool,
        num_to_move: int,
    ) -> (
        tuple[
            list[
                tuple[
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                    tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
                ]
            ],
            int,
        ]
        | tuple[None, 0]
    ):
        best_move_combination = None
        best_delta = 0

        for combination_ids in self.combinations_student_ids_by_dimension[num_to_move]:

            locations_assigned_students_combination = [
                location_student
                for student_id in combination_ids
                if (location_student := locations_students_by_id[student_id]) is not self.unassigned_students
            ]

            locations_num_occurences = Counter(
                (id(project), id(group)) for project, group in locations_assigned_students_combination
            )

            locations_assigned_students_combination: list[tuple[Project, ProjectGroup]]

            if not all(
                group.size() - locations_num_occurences[(id(project), id(group))] >= project.min_group_size
                for project, group in locations_assigned_students_combination
            ):
                group_size_insufficiencies = {
                    (id(project), id(group)): quantity_insuffiency
                    for project, group in locations_assigned_students_combination
                    if (
                        quantity_insuffiency := -(
                            group.size() - locations_num_occurences[(id(project), id(group))] - project.min_group_size
                        )
                        > 0
                    )
                }
                if not all(
                    num_to_move - locations_num_occurences[(id(project), id(group))]
                    >= group_size_insufficiencies[(id(project), id(group))]
                    for project, group in locations_assigned_students_combination
                    if (id(project), id(group)) in group_size_insufficiencies
                ):
                    continue

            else:
                group_size_insufficiencies = {}

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
                        student_location[1] is not student_destination[1]
                        if (student_location := locations_students_by_id[combination_ids[i]])
                        is not self.unassigned_students
                        and (student_destination := ordered_n_tuple_destinations[i]) is not self.unassigned_students
                        else student_location is not ordered_n_tuple_destinations[i]
                    )
                    for i in range(num_to_move)
                ),
                ordered_n_tuples_destinations,
            )

            if across_projects:
                best_arrivals_combination, delta = self._find_best_arrivals_combination(
                    ordered_n_tuples_destinations_corresponding_student_moving,
                    corresponding_departures,
                    group_size_insufficiencies,
                    num_to_move,
                )

            else:
                ordered_n_tuples_destinations_within_project = filter(
                    lambda ordered_n_tuple_destinations, combination_ids=combination_ids: all(
                        (
                            student_location[0] is student_destination[0]
                            if (student_location := locations_students_by_id[combination_ids[i]])
                            is not self.unassigned_students
                            and (student_destination := ordered_n_tuple_destinations[i])
                            is not self.unassigned_students
                            else True
                        )
                        for i in range(num_to_move)
                    ),
                    ordered_n_tuples_destinations_corresponding_student_moving,
                )

                best_arrivals_combination, delta = self._find_best_arrivals_combination(
                    ordered_n_tuples_destinations_within_project,
                    corresponding_departures,
                    group_size_insufficiencies,
                    num_to_move,
                )

            if delta > best_delta:

                best_delta = delta
                best_move_combination = [
                    (corresponding_departures[i], best_arrivals_combination[i]) for i in range(num_to_move)
                ]

        return best_move_combination, best_delta

    def _find_best_arrivals_combination(
        self,
        ordered_n_tuples_destinations: Iterator[tuple[tuple[Project, ProjectGroup] | list[Student]]],
        corresponding_departures: tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]],
        group_size_insufficiencies: dict[tuple[int, int] : int],
        num_to_move: int,
    ) -> tuple[
        tuple[tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student]] | None,
        int,
    ]:
        best_arrivals_combination = None
        moving_students = tuple(corresponding_departure[-1] for corresponding_departure in corresponding_departures)
        group_departures = [
            departure for departure in corresponding_departures if departure[0] is not self.unassigned_students
        ]
        combined_departure_delta = (num_to_move - len(group_departures)) * self.penalty_student_not_assigned
        for project, group, student in group_departures:
            combined_departure_delta += self._calculate_leaving_delta((project, group, student))
            group.release_student(student)

        arrival_delta_to_surpass = -combined_departure_delta

        ordered_n_tuples_destinations = filter(
            lambda destinations: all(
                group.size() < project.max_group_size
                for project, group in (
                    destination for destination in destinations if destination is not self.unassigned_students
                )
            ),
            ordered_n_tuples_destinations,
        )

        for destinations in ordered_n_tuples_destinations:
            group_arrivals = [
                (*destination, moving_students[i])
                for i, destination in enumerate(destinations)
                if destination is not self.unassigned_students
            ]
            if group_size_insufficiencies:
                group_destination_occurences = Counter(
                    (id(project), id(group)) for project, group, _ in group_arrivals
                )
                if not all(
                    group_destination_occurences.get(ids, -1) >= insufficiency
                    for ids, insufficiency in group_size_insufficiencies.items()
                ):
                    continue
            if len({(id(project), id(group)) for project, group, _ in group_arrivals}) == (
                num_group_arrivals := len(group_arrivals)
            ):
                combined_arrival_delta = (
                    sum(self._calculate_arrival_delta(arrival) for arrival in group_arrivals)
                    - (num_to_move - num_group_arrivals) * self.penalty_student_not_assigned
                )

            else:
                group_student_acceptances = []
                combined_arrival_delta = 0
                invalid_destinations = False
                for project, group, student in group_arrivals:
                    if group.size() >= project.max_group_size:
                        invalid_destinations = True
                        break
                    combined_arrival_delta += self._calculate_arrival_delta((project, group, student))
                    group.accept_student(student)
                    group_student_acceptances.append((group, student))

                for group, student in group_student_acceptances:
                    group: ProjectGroup
                    group.release_student(student)

                if invalid_destinations:
                    continue

                combined_arrival_delta -= (num_to_move - num_group_arrivals) * self.penalty_student_not_assigned

            if combined_arrival_delta > arrival_delta_to_surpass:
                arrival_delta_to_surpass = combined_arrival_delta
                best_arrivals_combination = tuple(
                    (
                        (*destination, moving_students[i])
                        if destination is not self.unassigned_students
                        else (self.unassigned_students, moving_students[i])
                    )
                    for i, destination in enumerate(destinations)
                )

        for _, group, student in group_departures:
            group.accept_student(student)

        return best_arrivals_combination, arrival_delta_to_surpass + combined_departure_delta

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
        arrival_specifications: tuple[list[Student], Student] | tuple[Project, ProjectGroup, Student],
    ) -> int:
        if arrival_specifications[0] is self.unassigned_students:
            return -self.penalty_student_not_assigned
        project, group, student = arrival_specifications
        preference_gain = student.preference_value(project)
        bilateral_reward_gain = (
            sum(
                (
                    (arriving_id, present_id) in self.bilateral_pairs
                    if (arriving_id := student.student_id) < (present_id := group_member.student_id)
                    else (present_id, arriving_id) in self.bilateral_pairs
                )
                for group_member in group.students
            )
            * self.reward_bilateral_interest_collaboration
        )
        if group.size() < project.ideal_group_size:
            delta_group_size = project.penalty_deviation_from_ideal_group_size
        else:
            delta_group_size = -project.penalty_deviation_from_ideal_group_size
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
                if any((group.students and group.size() < project.max_group_size) for group in project.groups)
            ]
            if not candidate_projects:
                return (self.unassigned_students, student)
            chosen_project: Project = rd.choice(candidate_projects)
            candidate_groups = [
                group
                for group in chosen_project.groups
                if group.students and group.size() < chosen_project.max_group_size
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
                and any((group.students and group.size() < project.max_group_size) for group in project.groups)
            ]

            if not candidate_projects:
                return (self.unassigned_students, student)

            chosen_project = rd.choice(candidate_projects)

        candidate_groups = [
            group
            for group in chosen_project.groups
            if group is not current_group and group.students and group.size() < chosen_project.max_group_size
        ]

        if not candidate_groups:
            return (self.unassigned_students, student)
        chosen_group = rd.choice(candidate_groups)
        return (chosen_project, chosen_group, student)

    def _calculate_leaving_delta(
        self,
        departure_specifications: tuple[Project, ProjectGroup, Student] | tuple[list[Student], Student],
    ) -> int:
        if departure_specifications[0] is self.unassigned_students:
            return self.penalty_student_not_assigned
        project, group, student = departure_specifications
        preference_loss = student.preference_value(project)
        bilateral_reward_loss = (
            sum(
                (
                    (leaving_id, remaining_id) in self.bilateral_pairs
                    if (leaving_id := student.student_id) < (remaining_id := group_member.student_id)
                    else (remaining_id, leaving_id) in self.bilateral_pairs
                )
                for group_member in group.students
                if group_member is not student
            )
            * self.reward_bilateral_interest_collaboration
        )
        if group.size() > project.ideal_group_size:
            delta_group_size = project.penalty_deviation_from_ideal_group_size
        else:
            delta_group_size = -project.penalty_deviation_from_ideal_group_size
        return -preference_loss - bilateral_reward_loss + delta_group_size

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
                rd.random() < len(unassigned_students_remaining) / self.num_students * assignment_bias
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
                    departures_specifications += [
                        (self.unassigned_students, student)
                        for student in rd.choices(
                            unassigned_students_remaining, k=min(len(unassigned_students_remaining), num_to_move - i)
                        )
                    ]
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

    def current_objective_value(self) -> int:
        """Calculates the current objective value.

        During solving the objective value is only calculated by
        deltas of moves. This complete calculation is used for
        evaluating the initial solution and checking whether the
        calculation by delta went wrong.
        """
        return (
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
        return sum(
            self.reward_bilateral_interest_collaboration
            for project in self.projects
            for group in project.groups
            for id_pair in it.combinations(sorted([student.student_id for student in group.students]), 2)
            if id_pair in self.bilateral_pairs
        )

    def _sum_missing_assignment_penalties(self):
        return len(self.unassigned_students) * self.penalty_student_not_assigned

    def _sum_group_surplus_penalties(self):
        return sum(
            max(0, project.num_non_empty_groups() - project.offered_num_groups) * project.penalty_extra_group
            for project in self.projects
        )

    def _sum_group_size_penalties(self):
        return sum(
            abs(group.size() - project.ideal_group_size) * project.penalty_deviation_from_ideal_group_size
            for project in self.projects
            for group in project.groups
            if group.students
        )

    def _initial_solution(self):
        projects_waitlists = {
            project.project_id: sorted(
                self.students,
                key=lambda student, project=project: student.preference_value(project),
                reverse=True,
            )
            for project in self.projects
        }
        assigned_students = []
        any_group_added = True
        while len(assigned_students) < self.num_students and any_group_added:
            any_group_added = False
            keen_projects = (project for project in self.projects if project.num_groups() < project.offered_num_groups)
            for project in keen_projects:
                unassigned_descending_preference = [
                    student for student in projects_waitlists[project.project_id] if student not in assigned_students
                ]
                if len(unassigned_descending_preference) >= project.ideal_group_size:
                    now_assigned_students = unassigned_descending_preference[: project.ideal_group_size]
                    project.add_initial_group_ideal_size(now_assigned_students)
                    assigned_students += now_assigned_students
                    any_group_added = True

        self.unassigned_students = [student for student in self.students if student not in assigned_students]
