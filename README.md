# cs221-project
# Unix intallation
* Install cmake: `brew install cmake`
* Install pip: `sudo easy_install pip`
* Install virtualenv: `pip install --upgrade virtualenv `
* Create a new virtual environment: `virtualenv --system-site-packages ~/tensorflow`
* activate virtualenv: `source ~/tensorflow/bin/activate ` (use `deactivate` to close it)
* Install tensorflow: https://www.tensorflow.org/install/install_mac
* Install keras: https://keras.io/#installation
* In same virtual environment, install open gym dependencies:
  * `pip install gym[atari] --ignore-installed`
    * put `gym[atari]` in quotes if using zsh
  * `pip install future --ignore-installed`
  * `pip install pydot`
# Running DQN Agent
`python riverraid.py`
# Running in Codalab
* Signup Codalab account
* upload riverraid.py
* upload riverraid_eval.py
* Paste the following command on the cli web interface on top of the page
* cl run :riverraid.py :riverraid_eval.py "python riverraid.py <your hyper parameters>" --request-docker-image gdadwal/openai-gym <--request-gpus 1>
* For Custom Markup add this to your worksheet
* % display image /reward.png
