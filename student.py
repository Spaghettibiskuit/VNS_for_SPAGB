"""Contains the class Student."""


class Student:
    """A student who seeks assignment to a project.

    Attributes:
        student_id: His/her ID
        name: His/her name
        fav_partners: IDs of students he/she wants to work with.
        projects_prefs: Preference values for every project. The index
            position of a value is the ID of the project.
    """

    def __init__(
        self,
        student_id: int,
        name: str,
        fav_partners: list[int],
        projects_prefs: tuple[int],
    ):
        """Initializes student.

        Args:
            student_id: His/her ID.
            name: His/her name.
            fav_partners: IDs of students he/she wants to work with.
            projects_prefs: Preference values for each project available at
                the index position that corresponds to the ID of the project.
        """
        self.student_id = student_id
        self.name = name
        self.fav_partners = fav_partners
        self.projects_prefs = projects_prefs

    def preference_value(self, project) -> int:
        """Returns the student's preference value for a project.

        Args:
            project: Instance of Project.
        """
        return self.projects_prefs[project.project_id]
