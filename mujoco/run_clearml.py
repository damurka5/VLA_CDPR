from clearml import Task

task = Task.init(
    project_name="AK_ClearML_experiments",
    task_name="CDPR Simulation with OpenVLA",
    task_type=Task.TaskTypes.testing
)

# task.set_packages(requirements_file="requirements.txt")

# Let ClearML auto-detect the environment
# No docker setup needed - it will use default images

task.execute_remotely(queue_name="default")
# Now import and run your main script
import clearml_cdpr  # Replace with your actual script name
clearml_cdpr.main()