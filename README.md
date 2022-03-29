# LSTM_SVA
(CNN feature extractor) + (LSTM) for surgery video analysis algoritm

This code can be used for phase recognition and tool detection.

Cholec80 dataset can be downloaded from the following web-site.

http://camma.u-strasbg.fr/datasets

After downloading the dataset, exctract images at 1[fps] rate, and put the resulting 80 folders in the following folder before using the codes in this repository.

../data/cholec80/pngs


<Current results>
Total Phase Accuracy : 90.34 % 
Phase Precision : 86.71 +- 8.79 %
Phase Recall : 85.45 +- 10.09 %
mAP of Tool Detection : 81.00 +- 14.09 %




![image](https://user-images.githubusercontent.com/72535628/160593478-d336b6b5-5237-4ce2-81b9-50ba51e1dd90.png)

![image](https://user-images.githubusercontent.com/72535628/160593520-d279f574-44a2-482c-9b7d-79c8f98c21aa.png)

New surgery video analysis algoritms (based on TCNs and transformers) will be coming soon.
