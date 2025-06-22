"""Contains the class ProjectGroup."""

from student import Student


class ProjectGroup:
    """Contains information on a project group relevant for VNS."""

    def __init__(self, project_id: int, project_name: str, students: list[Student]):
        self.project_id = project_id
        self.project_name = project_name
        self.students = students

    def accept_student(self, arriving_student: Student):
        if arriving_student in self.students:
            raise ValueError("Arriving student already in group!")
        self.students.append(arriving_student)

    def release_student(self, departing_student: Student):
        self.students.remove(departing_student)

    def size(self) -> int:
        """Return how many students are currently in the group."""
        return len(self.students)

    def remaining_students(self, departures: list[Student]) -> list[Student]:
        return [student for student in self.students if student not in departures]

    def remaining_size(self, departures: list[Student]) -> int:
        return len(self.remaining_students(departures))
