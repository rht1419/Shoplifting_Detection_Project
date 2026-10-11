============================= A few notes on using the dataset ================================
Chen Chen, UNC-Charlotte
https://webpages.uncc.edu/cchen62/
chen.chen@uncc.edu

Cite our paper:
Waqas Sultani, Chen Chen, Mubarak Shah, "Real-world Anomaly Detection in Surveillance Videos"
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
===============================================================================================

1. Anomaly Detection Experiment

- A) Videos:
Anomaly-Videos-Part-1 -- Part-4 (4 zip files)
Training-Normal-Videos-Part-1.zip and Training-Normal-Videos-Part-2.zip (normal videos for training)
Testing_Normal_Videos.zip (normal videos for testing)

UCF_Crimes-Train-Test-Split.zip contains the traing and testing split in our experiments
(folder: Anomaly_Detection_splits)


- B) Temporal Annotations for Testing Videos (anomaly videos)

Temporal_Anomaly_Annotation_for_Testing_Videos.txt

 Each row of 'Temporal_Anomaly_Annotation.txt' is the annotation for a video, for example:
Abuse028_x264.mp4  Abuse  165  240  -1 -1  
-	The first column is the name of the video
-	The second column is the name of the anomalous event
-	The third column is the starting frame of the event (you will have to convert each video to image frames first) 
-	The fourth column is the ending frame of the event.
-	For videos in which second  instance of event occurs, fifth and sixth contains starting and ending frames of second instance.  
    Negative number means no anomalous event instance. In this example, abuse (instance) only occurs once.

Note: Ours videos have 30 frames per second.



2. Anomaly Event Recognition Experiment

Classify 13 anomaly events and normal event

Normal_Videos_for_Event_Recognition.zip contains the normal videos we used for event recognition experiment
Note: rename the unziped folder to "Normal_Videos_event"

UCF_Crimes-Train-Test-Split.zip also contains the traing and testing split for event recognition experiment
(folder: Action_Regnition_splits)
