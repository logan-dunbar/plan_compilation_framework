from setuptools import setup

setup(name='plan_compilation_framework',
      version='0.1.0',
      description="Code that compiles a planner's policy into a model-free value function which"
                  " eventually outperforms the planner through constrained exploration.",
      url='https://github.com/logan-dunbar/plan_compilation_framework',
      author='Logan Dunbar',
      author_email='logan.dunbar@gmail.com',
      license='MIT',
      packages=['plan_compilation_framework'],
      zip_safe=False)
