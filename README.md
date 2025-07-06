# Variable Neighborhood Search for the Student-Project-Allocation with Group Building Problem

In this problem, project allocation and group building happens simultaneously. Students specify their preference value for every project that is being offered and specify a fixed number of students they want to work with the most. Those offering the projects specify guidelines, whishes and penalties regarding the number of groups and group sizes in their specific project. There is also a fixed reward for every occurrence in the solution of two students who have specified each other as partner preferences being in the same group (a bilateral pair). Lastly, there is a fixed penalty for every student in the solution who is not assigned to a group.

For more details on the problem, check out gurobi_solutions.ipynb where the problem is modeled as an MILP.

## Project Overview

The class VariableNeighborhoodSearch in which VNS is performed is in vns_on_student_assignment. Instances of this class are created in the following scripts which serve different purposes.

vns_solutions: Used for benchmarking. For benchmarking results check out any vns_benchmarks_foo.json file.

test_vns: Used for testing. Lets VNS solve possibly endless numbers of instances in a reproducible way and logs any occurrences of invalid solutions or a wrongly calculated objective values in a designated file. Each line of the error log specifies what happened and when it happened if anything went wrong.

demonstrate_vns: Run it for a small demonstration of VNS in the terminal.

The script gurobi_solutions is used to benchmark Gurobi on problem instances. For benchmarking results check out any gurobi_benchmarks_foo.json file.

To see how my VNS implementation performed compared to Gurobi, check out the notebook benchmark_analysis.

The instances always consist of two parts. The input from the projects (found in instances_projects) and the input from the students (found in instances_students).

In projects_info and students_info you can see how instances are created semi-randomly.

You may learn what the rest of the files do from their documentation.

## Requirements

I used Python 3.13.2. The only dependencies necessary are pandas and gurobipy.

## License
[MIT](https://choosealicense.com/licenses/mit/)






