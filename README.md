# barpath-detection
Detect barpath and calculate bar velocity in a video shot from a side angle.
## Installation
Requires Node.js and Python. 

Navigate to the project directory and create a virtual environment with
```
python -m venv .venv
```
and install the required Python libraries (this may take a second)
```
python -m pip install -r requirements.txt
```

## Usage
1. Run the frontend server - navigate to the `/frontend/` folder and run in a terminal
```
npm install barbell-detector
cd barbell-detector
npm run dev
```
2. Run the backend server - in a separate terminal, navigate to the `/backend/` folder and run
```
python main.py
```

Now you can upload a video file to the app, and it will attempt to detect the barbell and calculate its velocity.

### Tips
The app is built with kilogram plates in mind and needs to see the full side view of the kilogram plate in order to accurately calculate velocity. Ensure that any uploaded video has has the red plate in full view (a small difference in angle is okay). 

Currently the website automatically tries to detect the lift type, but if there are lots of benches or other objects in view, it might have trouble. Manual lift classification is WIP. 

## How It's Made
I wanted to create this app to learn more about full-stack development and because it intersected with my hobbies of weightlifting. Velocity based training can be used to estimate one's proximity to failure, but the physical devices to estimate barbell velocity can be prohibitively expensive. My goal with this project was to make a lightweight website that a user could quickly access, upload a video of a lift, get their data, and do whatever they wanted with it.  

Through working on this project, I've been able to apply what I have learned about machine learning, and learn so much more about web dev. My work on this app has been in a few stages:

1. The model(s)
    - I used a YOLOv11 normal sized model to create the barbell detector. I got my dataset from Roboflow, then went through and reannotated where I felt it was necessary. I took a long time on data selection, because I really wanted it to be accurate and I was not sure just how thorough I would have to be. 
    - Once I got my data together, I trained the model and was happy with my accuracy. The results can be found in the `/runs/train` folder. I found that the Supervision library was very useful and intuitive to annotate the video with detections from the YOLO model, so that gave me the bar tracing functionality.
    - While planning on how to track the barbell and calculate velocity, I thought it would be helpful to be able to classify the type of lift (squat, bench, or deadlift) in order to have more accurate tracking. I created a second dataset and trained a [classification model](https://github.com/camronrule/lift-classifier) which I implemented into this project. The classification takes place directly before the barbell detection.
2. The tracker
    - The algorithms behind the tracker were the most familiar part of this project for me. The foundational piece of the tracker is the relative size of the red kilo plate in the video, which is used to scale the pixel displacement of the barbell between frames, and estimate speed, velocity, and acceleration. These measures, in the context of the 'phase' of the lift (see `/backend/detectors/BarbellPhase.py/`) are used to assume is a frame is the start of a new phase of the lift. 
3. The frontend
    - 

## Planned Features
- Cleanup temp files before closing
- Implement deadlift phase classification
- Implement multi-rep support
- Data presentation using D3.js
- View the results and download links of multiple different uploaded videos
- More concise, useful video annotation (max concentric speed)
- Explain importance of velocity based training