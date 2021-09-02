# prep (dataset_preparation)
In this folder of the project, we record various lane change scenarios using CARLA and scenario runner. 

**annotator.py**
This file is used to annotate the dataset.

```python
self.parser.add_argument('--input_path', type=str, default="merges/", help="Path to data directory.")
self.parser.add_argument('--start', type=int, default=10230, help="Starting lane change clip ex: for 5825_201706081335 use 5825")
self.parser.add_argument('--frame_delay', type=int, default=1, help="The amount of delay (ms) between each clip")
self.parser.add_argument('--filter', type=str, default='', help="Type of maneuver to consider {branches, lanechange, merges, turns2}")
self.parser.add_argument('--risk', default=False, action='store_true', help="Set to show only risky clips")
self.parser.add_argument('--nonrisk', default=False, action='store_true', help="Set to show only non-risky clips")
```

The most important buttons are Save Score, Ignore Clip, Pause, Replay Clip, Prev Clip, Next Clip.

* Save Score: saves the risk score that you give for a clip (after entering your score you can either press the Save Score button or press enter on the keyboard)
* Ignore Clip: ignores given clip so that it is not used in the final dataset (contact louis, arnav, or brandon for types of clips that should be ignored)
* Pause: pauses the given clip
* Replay Clip: replays the given clip
* Prev Clip: goes back one clip
* Next Clip: moves to the next clip

*Ignore the other buttons.*
* For best performance run both the dataset and program locally

Troubleshooting:
* Getting a Display Error?
   * Use something similar to Xquartz or Xming
* Where are the buttons?
   * Try resizing the window
* Why is the clip not showing up?
   * Trying pressing Replay Clip
* Why is it slow?
   * Remote server latency :(

**CARLA** 
This folder contains the files to run the CARLA simulation. 

**Scenario_Runner**
This project created multiple traffic scenarios to simulate real world driving for CARLA. Scenarios include leading vehicles or pedestrians crossing. We used these scenarios while randomly choosing a vehicle to lane change.

**spawn_npc.py**
This file spawns a number of vehicles and pedestrians in the CARLA world. This is the main file we run to record lane changes. The recordings are saved into a folder called _out.

Each recording has 3 folders: raw_images, scene_raw, ss_images
 - **raw_images:** frames of the recording using rgb camera
 - **scene_raw:** information on the environment, cars, pedestrians in dictionary format
 - **ss_images:** frames of the recording using segmented segmentation camera

**sensors.py**
This file contains cameras and other sensors that attach to the ego vehicle. The camera is used to record lane changes. In addition, we created functions to extract information from the vehicle.

**lane_change_recorder.py**
This file chooses a random vehicle to lane change. It records each lane change and gathers information from the surrounding environment. 

# How to Execute
1. Download CARLA 0.9.8 - refer to https://carla.readthedocs.io/en/0.9.8/ to set up
2. Navigate to /scenario_runner and pip install -r requirements.txt
3. Run the CARLA executable
4. Run spawn_npc.py in synchronous mode to record lane changes
   - ex. python spawn_npc.py -n 100 --sync --safe
