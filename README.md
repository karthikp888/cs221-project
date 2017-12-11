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
upload riverraid.py
cl run :riverraid.py "python riverraid.py --num-episodes 5" --request-docker-image gdadwal/openai-gym
* For Custom Markup add this to your worksheet
* % schema r1
* % addschema run
* % add '|'
* % add episode /stats.json:episode
* % add reward /stats.json:reward
* % add steps /stats.json:steps
* % add secs /stats.json:secs
* % add epsilon /stats.json:epsilon
* % add predicted_action /stats.json:predicted_action
* % add random_action /stats.json:random_action
* % display table r1