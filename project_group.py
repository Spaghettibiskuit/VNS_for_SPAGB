"""Contains the class ProjectGroup."""

from student import Student


class ProjectGroup:
    """Contains information on a project group relevant for VNS."""

    def __init__(
        self, project_id: int, project_name: str, students: list[Student]
    ):
        self.project_id = project_id
        self.project_name = project_name
        self.students = students
        self.bilateral_preferences = set()
        self.size = len(self.students)
        self._create_bilateral_preferences_set()
        self.num_bilateral_pairs = len(self.bilateral_preferences)

    def _create_bilateral_preferences_set(self):
        student_ids_in_group = tuple(
            student.student_id for student in self.students
        )
        for student in self.students:
            for fav_partner in student.fav_partners:
                if (
                    fav_partner in student_ids_in_group
                    and tuple((student.student_id, fav_partner))
                    not in self.bilateral_preferences
                ):
                    self.bilateral_preferences.add(
                        tuple((fav_partner, student.student_id))
                    )
