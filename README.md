# PoseOFF Extraction Demo
This repo contains a simple, live demonstration of the PoseOFF extraction method using the [Lucas-Kanade method](https://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf) of optical flow estimation in combination with [YOLO Pose](https://docs.ultralytics.com/tasks/pose/). 

**NOTE**: This demo only shows a visualisation of the extraction method, and does not contain and end-to-end human action anticipation framework.

## Installation
Clone this repo:

``` sh
git clone https://github.com/LewisdeZ/PoseOFF_demo.git
```

Next, create the python environment and install the requirements:
#### UNIX
``` sh
cd PoseOFF_demo
python -m venv ./env
# Activate the environment
source ./env/bin/activate
# Install requirements, this may take some time!
python pip install -r requirements.txt
```

#### Windows
``` sh
cd PoseOFF_demo
python -m venv ./env
# Activate the environment, may change depending on if using CMD or PowerShell...
.\env\Scripts\Activate.ps1 # You should see (env) at the front of your prompt
# Install requirements, this may take some time!
python pip install -r requirements.txt
```


## Usage
Ensure that the virtual environment is active (`source env/bin/activate` for unix or `.\env\Scripts\Activate.ps1` for Windows).

Run the demo using default settings, simply use:

``` sh
python demo.py 
```

***Press `q` to quit the demo.***

For help on the available arguments, use:

``` sh
python demo.py -h
```

Optional arguments are as follows:
| Argument flag       | Argument description                                                                                                                             |
|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| -t, --threshold     | Confidence threshold below which pose keypoints will be discarded, between 0.0 and 1.0 (default: 0.2).                                           |
| -w, --window_size   | Width of square optical flow sampling window - must be an odd number - for example window_size=5 would result in a 5*5 pixel window (default: 5) |
| -d, --dilation      | Dilation factor of sampling window, a higher dilation means a more spread sampling window (default: 3)                                           |
| -c, --camera_number | Camera number to stream from, this may require some trial and error... (default: 0)                                                              |
| -m, --only_middle   | If passed, only draw the middle optical flow arrow on each pose keypoint -store_true (default: False)                                            |

