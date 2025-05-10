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
        pen_groups: int,
        pen_size: int,
    ):
        self.project_id = project_id
        self.name = name
        self.offered_num_groups = offered_num_groups
        self.max_num_groups = max_num_groups
        self.ideal_group_size = ideal_group_size
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.pen_groups = pen_groups
        self.pen_size = pen_size
        self.groups: list[ProjectGroup] = []

    def add_initial_group_ideal_size(self, unassigned_students):
        """Adds a group of unassigned students for initial solution."""
        new_group = ProjectGroup(
            self.project_id, self.name, unassigned_students
        )
        self.groups.append(new_group)

    def num_groups(self) -> int:
        """Return how many groups are currently in the project."""
        return len(self.groups)
