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
        self.num_bilateral_pairs = 0
        self.size = len(self.students)
        self._calculate_bilateral_preferences_set()

    # Currently entirely recalculated with every change. Potential for optimization.
    def _calculate_bilateral_preferences_set(self):
        self.bilateral_preferences = set()
        # student_ids_in_group = tuple(
        #     student.student_id for student in self.students
        # )
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
        # for student in self.students:
        #     for fav_partner in student.fav_partners:
        #         if fav_partner in student_ids_in_group:
        #             bilateral_pref = tuple(
        #                 (fav_partner, student.student_id)
        #                 if fav_partner < student.student_id
        #                 else (student.student_id, fav_partner)
        #             )
        #             self.bilateral_preferences.add(bilateral_pref)
        self.num_bilateral_pairs = len(self.bilateral_preferences)

    def accept_student(self, arriving_student: Student):
        if arriving_student in self.students:
            raise ValueError("Arriving student already in group!")
        self.students.append(arriving_student)
        self.size += 1
        self._calculate_bilateral_preferences_set()

    def release_student(self, departing_student: Student):
        if departing_student not in self.students:
            raise ValueError("Departing student not in group!")
        self.students.remove(departing_student)
        self.size -= 1
        self._calculate_bilateral_preferences_set()
