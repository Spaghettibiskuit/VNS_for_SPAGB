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
        self._populate_bilateral_preferences_set()

    def _populate_bilateral_preferences_set(self):
        self.bilateral_preferences = set()
        preference_dict = {
            student.student_id: student.fav_partners
            for student in self.students
        }
        for student_id, fav_partners in preference_dict.items():
            for fav_partner in fav_partners:
                if (
                    fav_partner in preference_dict
                    and student_id in preference_dict[fav_partner]
                ):
                    bilateral_pref = tuple(
                        (fav_partner, student_id)
                        if fav_partner < student_id
                        else (student_id, fav_partner)
                    )
                    self.bilateral_preferences.add(bilateral_pref)

    def accept_student(self, arriving_student: Student):
        if arriving_student in self.students:
            raise ValueError("Arriving student already in group!")
        self.students.append(arriving_student)
        self._update_bilateral_preferences_set_arrival(arriving_student)

    def release_student(self, departing_student: Student):
        self.students.remove(departing_student)
        self._update_bilateral_preferences_set_departure(
            departing_student.student_id
        )

    def _update_bilateral_preferences_set_departure(
        self, departing_student_id
    ):
        self.bilateral_preferences = {
            bilateral_preference
            for bilateral_preference in self.bilateral_preferences
            if departing_student_id not in bilateral_preference
        }

    def _update_bilateral_preferences_set_arrival(
        self, arriving_student: Student
    ):
        arriving_student_favs_present = [
            student
            for student in self.students
            if student.student_id in arriving_student.fav_partners
        ]

        for student in arriving_student_favs_present:
            if arriving_student.student_id in student.fav_partners:
                bilateral_pref = (
                    (student.student_id, arriving_student.student_id)
                    if student.student_id < arriving_student.student_id
                    else (arriving_student.student_id, student.student_id)
                )
                self.bilateral_preferences.add(bilateral_pref)

    def size(self) -> int:
        """Return how many students are currently in the group."""
        return len(self.students)

    def remaining_students(self, departures: list[Student]) -> list[Student]:
        return [
            student for student in self.students if student not in departures
        ]

    def remaining_size(self, departures: list[Student]) -> int:
        return len(self.remaining_students(departures))
