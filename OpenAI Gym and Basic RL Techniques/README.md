## Intro to OpenAI Gym and Basic Reinforcement Learning Techniques

###  Intro

![This](cartpole_intro.py) file has an introductory routine where cartpole environment is setup from gym and random actions are performed until episode isn't over. We calculate the average number of steps before the episode ends.

### Random Search

![This](random_search.py) file is used to solve the cartpole problem using a random search method. A random parameters array is defined and its performance is measured on the simulation. The best paramter array is chosen to be the solution. We basically search for the best hyperparameter in R^n, where n is the dimension of the parameters.

The below graph shows the number of time-steps for each parameter.

<p align="center">
  <img src="random_search.png" width="300"/>
</p>
