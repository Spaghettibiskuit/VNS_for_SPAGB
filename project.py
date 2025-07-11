"""Contains the class Project."""

from project_group import ProjectGroup
from student import Student


class Project:
    """One of the projects in the problem.

    Attributes:
        project_id: ID of the project.
        name: name/topic of the project.
        offered_num_groups: How many groups the project wants to supervise.
        max_num_groups: Maximum number of groups it is willing to supervise.
        ideal_group_size: Number of students per group deemed ideal.
        min_group_size: Minimum number of students in any group.
        max_group_size: Maximum number of students in any group.
        penalty_extra_group: Penalty for every group exceeding offered_num_groups.
        penalty_deviation_from_ideal_group_size: Coefficient with which
            deviation from ideal_group_size is penalized.
        groups: Groups which are part of the project.
    """

    def __init__(
        self,
        project_id: int,
        name: str,
        offered_num_groups: int,
        max_num_groups: int,
        ideal_group_size: int,
        min_group_size: int,
        max_group_size: int,
        penalty_extra_group: int,
        penalty_deviation_from_ideal_group_size: int,
    ):
        """Initializes the project.

        Args:
            project_id: ID of the project.
            name: name/topic of the project.
            offered_num_groups: How many groups the project wants to supervise.
            max_num_groups: Maximum number of groups it is willing to supervise.
            ideal_group_size: Number of students per group deemed ideal.
            min_group_size: Minimum number of students in any group.
            max_group_size: Maximum number of students in any group.
            penalty_extra_group: Penalty for every group exceeding offered_num_groups.
            penalty_deviation_from_ideal_group_size: Coefficient with which
                deviation from ideal_group_size is penalized.
        """
        self.project_id = project_id
        self.name = name
        self.offered_num_groups = offered_num_groups
        self.max_num_groups = max_num_groups
        self.ideal_group_size = ideal_group_size
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.penalty_extra_group = penalty_extra_group
        self.penalty_deviation_from_ideal_group_size = penalty_deviation_from_ideal_group_size
        self.groups: list[ProjectGroup] = []

    def add_initial_group_ideal_size(self, unassigned_students: list[Student]):
        """Adds a group of ideal size during creation of the initial solution.

        Args:
            unassigned_students: students that will be in the group.
        """
        new_group = ProjectGroup(self.project_id, self.name, unassigned_students)
        self.groups.append(new_group)

    def num_groups(self) -> int:
        """Returns how many groups are currently in the project."""
        return len(self.groups)

    def get_new_empty_group_and_initial_delta(self) -> tuple[ProjectGroup, int]:
        """Returns new empty group in project and delta caused by penalties."""
        additional_group_delta = -self.penalty_extra_group if self.num_groups() >= self.offered_num_groups else 0
        initial_group_size_delta = -self.penalty_deviation_from_ideal_group_size * self.ideal_group_size
        new_empty_group = ProjectGroup(self.project_id, self.name, [])
        self.groups.append(new_empty_group)
        return (new_empty_group, additional_group_delta + initial_group_size_delta)

    def non_empty_groups(self) -> list[ProjectGroup]:
        """Returns non-empty groups in project."""
        return [group for group in self.groups if group.students]

    def num_non_empty_groups(self) -> int:
        """Returns how many non-empty groups are currently in the project."""
        return len(self.non_empty_groups())
