# Rigid versus Nonrigid Optical Flow &mdash; Stimuli for Perceptual Science
The aim of the stimulus set is to disentangle rigid and nonrigid optical flow.
Rigid optical flow describes the optical flow that can be explained by camera
motion alone: it represents where things on the screen would move
if the scene were completely static. Nonrigid optical flow is the optical flow 
that remains after subtracting rigid flow from the overall optical
flow. It thus represents the optical flow that camera motion is unable to
explain &mdash; typically due to things that move autonomously through the
scene such as humans, cars, or flowers that are moved by the wind etc.
Distinguishing between different types of optical flow is essential because
they carry fundamentally different ecological meanings.

The stimulus set includes data from various sources. Criteria for inclusion is
that the video material is ecologically realistic and that it comes with ground
truth information on optical flow, depth, and camera matrices. Examples are:

* [Sintel](http://sintel.is.tue.mpg.de/)
* [Spring](https://spring-benchmark.org/)
* [Monkaa](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Kubric](https://github.com/google-research/kubric)
* ...

We first verified that the ground truth information provided for depth, flow, 
and camera pose are consistent by focusing on video sections with static scenes
only. The provided ground truth optical flow should now be the same as the
rigid flow computed from depth and camera pose.

## Problem
We plot the distribution of ground truth optical flow (y axis) against the rigid 
flow that we compute from depth and pose (x axis). The distribution falls along
the identity for Sintel, Spring and for Monkaa. This shows that, for these datasets,
rigid flow can be computed from depth and pose.

![Sintel](https://github.com/mmbannert/rigid-v-nonrigid-flow/blob/master/images/sintel.png)
![Spring](https://github.com/mmbannert/rigid-v-nonrigid-flow/blob/master/images/spring.png)
![Monkaa](https://github.com/mmbannert/rigid-v-nonrigid-flow/blob/master/images/monkaa.png)

For Kubric, in contrast, the figure below shows a mismatch between ground truth optical flow
and the rigid flow that is computed from depth and pose. This could mean
that the scene parameters are inconsistent or that they cannot be interpreted
correctly to reconstruct rigid flow.

![Kubric](https://github.com/mmbannert/rigid-v-nonrigid-flow/blob/master/images/kubric.png)

For further details, please take a look at the accompanying [Jupyter Notebook](https://github.com/mmbannert/rigid-v-nonrigid-flow/blob/master/rigid_v_nonrigid_flow_demo.ipynb).

## Steps to reproduce

### Installation
To reproduce our results, unzip the code to an installation directory of your
choice. Within that directory, create the conda environment like so:

``` shell
conda env create --file environment.yml
```

### Data download
Download the stimulus folder from [here](https://keeper.mpdl.mpg.de/f/8d7f5791a6634d76a9b8/) and unzip it to the
installation directory. It contains the actual RGB frames that were used in the
example as well as the ground truth information and labels.

### Run the demo
Start a Jupyter Notebook.
``` shell
jupyter notebook
```
... and open the Jupyter Notebook called `rigid_v_nonrigid_flow_demo.ipynb`.
Change `base_dir` so that it points to your installation directory.