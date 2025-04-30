"""Contains the class Student."""


class Student:
    """Contains information on a student relevant for VNS."""

    def __init__(
        self,
        student_id: int,
        name: str,
        fav_partners: list[int],
        projects_prefs: tuple[int],
    ):
        self.student_id = student_id
        self.name = name
        self.fav_partners = fav_partners
        self.projects_prefs = projects_prefs
        self.assigned = False
