from pathlib import Path

import problem_data

folder_projects = Path("instances_projects")
folder_students = Path("instances_students")
folder_projects.mkdir()
folder_students.mkdir()


project_quantities = [3, 4, 5]
student_quantities = [30, 40, 50]
instances_per_combination = 10

for project_quantity in project_quantities:
    for student_quantity in student_quantities:
        subfolder_dimension_projects = folder_projects / f"{project_quantity}_{student_quantity}_instances"
        subfolder_dimension_students = folder_students / f"{project_quantity}_{student_quantity}_instances"
        subfolder_dimension_projects.mkdir()
        subfolder_dimension_students.mkdir()
        for instance_number in range(instances_per_combination):
            filename_projects = f"generic_{project_quantity}_{student_quantity}_projects_{instance_number}.csv"
            filename_students = f"generic_{project_quantity}_{student_quantity}_students_{instance_number}.csv"

            filepath_projects = subfolder_dimension_projects / filename_projects
            filepath_students = subfolder_dimension_students / filename_students

            problem_data.save_projects_and_students_instance(
                num_projects=project_quantity,
                num_students=student_quantity,
                filepath_projects=filepath_projects,
                filepath_students=filepath_students,
            )
