"""Contains classes which store settings.

Central place to save and change settings for running any
code whithin the folder.
"""


class DemonstrationSettings:
    "Stores all settings for demonstrating VNS."

    def __init__(self):
        """Initializes settings for demonstrating VNS."""
        # Existing instances have 3, 4 or 5 projects
        self.num_projects: int = 4
        # Existing instances have 30, 40 or 50 students
        self.num_students: int = 40
        # Existing instances are numbered from 0 to 9
        self.instance_number: int | None = 3
        # If you want to solve a randomly generated instance you may pick a seed
        self.seed_instance_generation: int | None = None
        # You may also pick a random seed for running VNS
        self.seed_vns_run: int | None = 100
        # You may choose how many iterations should be performed
        self.iterations: int = 30


class BenchmarkSettingsVNS:
    "Stores all settings for benchmarking VNS."

    def __init__(self):
        """Initializes settings for benchmarking VNS."""
        self.project_quantities = [3, 4, 5]
        self.student_quantities = [30, 40, 50]
        self.instances_per_dimension = 10
        self.time_limit = 300
        self.seed = 100
        self.filename_results = "vns_benchmarks_300s.json"
        self.filename_error_logs = "benchmark_1.txt"


class TestSettings:
    "Stores all settings for testing VNS."

    def __init__(self):
        """Initializes settings for testing VNS."""
        self.min_num_projects = 3
        self.max_num_projects = 6
        self.step_num_projects = 1
        self.min_num_students = 20
        self.max_num_students = 60
        self.step_num_students = 5
        self.iteration_limit = 100
        self.starting_random_seed = 0
        self.line_limit = 1000
        self.filename = "error_log_3.txt"


class BenchmarkSettingsGurobi:
    "Stores all settings for benchmarking Gurobi."

    def __init__(self):
        "Initializes settings for benchmarking Gurobi"
        self.project_quantities = [3, 4, 5]
        self.student_quantities = [30, 40, 50]
        self.instances_per_dimesion = 10
        self.reward_bilateral = 2
        self.penalty_unassignment = 3
        self.filename = "gurobi_benchmarks_verbose.json"
        self.time_limit = 60
