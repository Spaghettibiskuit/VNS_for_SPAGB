"""Contains the class Project."""

from project_group import ProjectGroup


class Project:
    """Contains information on a project relevant for VNS."""

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

    def add_initial_group_ideal_size(self, unassigned_students):
        """Add a group of unassigned students for initial solution."""
        new_group = ProjectGroup(self.project_id, self.name, unassigned_students)
        self.groups.append(new_group)

    def num_groups(self) -> int:
        """Return how many groups are currently in the project."""
        return len(self.groups)

    def get_new_empty_group_and_initial_delta(self) -> tuple[ProjectGroup, int]:
        additional_group_delta = -self.penalty_extra_group if self.num_groups() >= self.offered_num_groups else 0
        initial_group_size_delta = -self.penalty_deviation_from_ideal_group_size * self.ideal_group_size
        new_empty_group = ProjectGroup(self.project_id, self.name, [])
        self.groups.append(new_empty_group)
        return (new_empty_group, additional_group_delta + initial_group_size_delta)

    def non_empty_groups(self) -> list[ProjectGroup]:
        return [group for group in self.groups if group.students]

    def num_non_empty_groups(self) -> int:
        return len(self.non_empty_groups())
