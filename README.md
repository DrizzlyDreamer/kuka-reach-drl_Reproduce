# kuka-reach-drl
Train kuka robot reach a point with deep rl in pybullet.

* The train process
![The train process](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/train.gif)
* The evaluate process
![The evaluate process](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/eval.gif)
* The average episode reward
![The average episode reward](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/train_results.png)


* **I strongly recommend using Conda to install the env, because you will possible encounter the mpi4py error with pip.**
* The spinningup rl library is the necessary lib.

### Installation guide (Now only support linux and macos)
first, you should install miniconda or anaconda.
second, install some dev dependencies.
```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
third, create a conda virtual environment
```bash
conda create -n spinningup python=3.6   #python 3.6 is recommended
```
```bash
#activate the env
conda activate spinningup
```

last, install spiningup,is contains almost dependencies
```bash
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```

### files guide
* the train.py file is the main train file, you can directly run it or through `python train.py --cpu 6` to run it in terminal. Please notice the parameters.
* eval.py file is the evaluate trained model file, the model is in the logs directory named model.pt. In the eval file, pybullet render is open default. **When you want to evaluate my trained model, please change the source code `ac=torch.load("logs/ppo-kuka-reach/ppo-kuka-reach_s0/pyt_save/model.pt")` to `ac=torch.load("saved_model/model.pt")` in `eval.py`**
* ppo directory is the main algorithms about ppo.
* env directory is the main pybullet env.

### view the train results through plot
```bash
python -m spinup.run plot ./logs
``` 
More detailed information please visit [plotting results](https://spinningup.openai.com/en/latest/user/plotting.html)

#### alternative installation method
Or, you can create the virtual environment directly through
```bash
conda create --name spinningup --file requirements.txt
```
but I can not ensure this method can success.


### some relative blogs and articles.

1. [spinningup docs](https://spinningup.openai.com/en/latest/user/installation.html)
2. [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6)(do not carefully read now.)
3. [some ray/rllib and other rl problems' blogs](https://www.datahubbs.com/)
4. [Action Masking with RLlib](https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505)
5. [This AI designs beautiful Forest Landscapes for Games!](https://medium.com/deepgamingai/this-ai-designs-beautiful-forest-landscapes-for-games-8675e053636e)
6. [Chintan Trivedi's homepage](https://medium.com/@chintan.t93), he writes many blogs about AI and games. It's very recommended.
7. [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://twitter.com/ericwen5986/status/1374361315100172289)
8. [Proximal Policy Optimization Tutorial (Part 2/2: GAE and PPO loss)](https://twitter.com/ericwen5986/status/1374361470859767809)
9. [Antonin Raffin](https://araffin.github.io/), he is the member of stable baseline3 project.
10. [robotics-rl-srl](https://github.com/araffin/robotics-rl-srl), S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics.
11. [zenetio/DeepRL-Robotic](https://github.com/zenetio/DeepRL-Robotic), a deep rl project using gazebo.
12. [robotology-playground/pybullet-robot-envs](https://github.com/robotology-playground/pybullet-robot-envs), a deep rl project using pybullet, it is built by a company, there are a lot can study from their codes. But their envs do not introduce images.
13. [mahyaret/kuka_rl](https://github.com/mahyaret/kuka_rl), a tutorial tells you how to implement DQN and ppo algorithms to kuka robot grasping.
14. [AutodeskRoboticsLab/RLRoboticAssembly](https://github.com/AutodeskRoboticsLab/RLRoboticAssembly), a deep rl robot assembly project build by autodesk, it uses rllib and ray.
15. [MorvanZhou/train-robot-arm-from-scratch](https://github.com/MorvanZhou/train-robot-arm-from-scratch), a deep rl robot project build by Morvan.

