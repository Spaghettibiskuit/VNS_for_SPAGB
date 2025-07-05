"""Contains the class ProjectGroup."""

from student import Student


class ProjectGroup:
    """A group of students that is part of a project.

    Attributes:
        project_id: The ID of the project the group is part of.
        project_name: The name of the project it is part of.
        students: The students that are in it.
    """

    def __init__(self, project_id: int, project_name: str, students: list[Student]):
        """Initializes the group in a specific project.

        Args:
            project_id: ID of the project the group is part of.
            project_name: Name of the project it is part of.
            students: The students in it.
        """
        self.project_id = project_id
        self.project_name = project_name
        self.students = students

    def accept_student(self, arriving_student: Student):
        """Adds a student to the group.

        Raises:
            ValueError: Student is already in group.
        """
        if arriving_student in self.students:
            raise ValueError("Arriving student already in group!")
        self.students.append(arriving_student)

    def release_student(self, departing_student: Student):
        """Removes student from the group."""
        self.students.remove(departing_student)

    def size(self) -> int:
        """Returns the number of students in the group."""
        return len(self.students)

    def remaining_students(self, departures: list[Student]) -> list[Student]:
        """Returns the students in the group who are not yet designated to leave.

        Args:
            departures: Students that are designated to leave their group.
        """
        return [student for student in self.students if student not in departures]

    def remaining_size(self, departures: list[Student]) -> int:
        """Returns the number of students in the group who are not yet designated to leave.

        Args:
            departures: Students that are designated to leave their group.
        """
        return len(self.remaining_students(departures))
