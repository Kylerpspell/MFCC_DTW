import os

#For handling audio data
from pydub import AudioSegment

#MFCC generation 
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
import csv
from fastdtw import fastdtw
from numpy.linalg import norm

participantDirectory = "MelFreq-DTW/participantDirectory"

def splitAudio(filePath):
	stereo_audio = AudioSegment.from_file(filePath,format="wav")
	mono_audios = stereo_audio.split_to_mono()
	#Left(Participant)
	mono_audios[0].export(filePath.replace("stereo","left"),format="wav")
	#Right(Model)
	mono_audios[1].export(filePath.replace("stereo","right"),format="wav")
	return(filePath.replace("stereo","left"),filePath.replace("stereo","right"))

def mfccDtw(filePath):
	participant, model = splitAudio(filePath)
	#MFCC generation
	model_y, model_sr = librosa.load(model)
	pt_y, pt_sr = librosa.load(participant)
	mfccs_model = librosa.feature.mfcc(model_y, model_sr,hop_length=110,n_mfcc=25,win_length=220)
	mfccs_pt = librosa.feature.mfcc(pt_y, pt_sr,hop_length=110,n_mfcc=25,win_length=220)
	#Write to File 
	with open(model.replace('.wav','.csv'),'w') as out_m:
		csv_m = csv.writer(out_m)
		csv_m.writerows(np.rot90(mfccs_model, k=1))
	with open(participant.replace('.wav','.csv'),'w') as out_p:
		csv_m = csv.writer(out_p)
		csv_m.writerows(np.rot90(mfccs_pt, k=1))
	#DTW
	distance, path = fastdtw(mfccs_pt.T, mfccs_model.T, dist=euclidean)
	#Writes total distance to file
	with open(filePath.replace('stereo.wav','distance.csv'),'a') as out_d:
		csv_m = csv.writer(out_d)
		csv_m.writerow(["DTW Distance: ",distance])
	#Writes path to file
	with open(filePath.replace('stereo.wav','path.csv'),'a') as out_p:
		csv_m = csv.writer(out_p)
		for i in path:
			csv_m.writerow(i)
	
	#Generate plot
	x = []
	last = int()
	lowestPath = []
	for i in range(len(path)):
		if(path[i][0] != last):
			lowestPath.append(path[i])
			last = path[i][0]
	for i in range(len(lowestPath)):
		x.append(lowestPath[i][1])

	y = []
	for i in range(len(lowestPath)):
		y.append((((int(lowestPath[i][1])-int(lowestPath[i][0]))*11)/2205))

	tempx = list(map(int,x))
	x = []
	for xval in tempx:
		x.append((xval*11)/2205)
	
	plt.plot(x,y)
	plt.xlabel('Time (s)')
	plt.ylabel('Seconds Ahead (s)')
	plt.title('Frame Lag')
	plt.axhline(y=0, color='black')
	plt.savefig(filePath.replace('stereo.wav','lag.png'))
	plt.clf()

if __name__ == '__main__':

	for dirs, subdirs, files in os.walk("MelFreq-DTW/participantDirectory/"):
		for file in files:
			if file.endswith("stereo.wav"):
				mfccDtw(os.path.join(dirs, file))


	for dirs, subdirs, files in os.walk(participantDirectory):
		for dir in subdirs:
			participantTop = os.path.join(participantDirectory,dir)
			out = []
			Day1 = participantTop + "/Day 1"
			Day2 = participantTop + "/Day 2"
			Day3 = participantTop + "/Day 3"
			days = [Day1, Day2, Day3]
			pxNames = ['Px1','Px2','Px3']
			for day in days:
				num = day[-1]
				stim = day + "/SCRIPT 6 (STIM)"
				post = day + "/SCRIPT 10 (POST STIM)"
				scripts = [stim, post]
				for script in scripts:
					for dirs, subdirs, files in os.walk(script):
						#Generate Distance CSV
						distances = [0,0,0]
						paths = [[],[],[]]
						for file in files:
							if file.endswith("distance.csv"):
								Px = file[file.index('Px')+2]
								with open(os.path.join(dirs,file),'r') as f:
									reader = csv.reader(f)
									d = list(reader)
									distances[int(Px)-1] = float(d[0][1])

							if file.endswith("path.csv"):
								Px = file[file.index('Px')+2]
								with open(os.path.join(dirs, file),'r') as h:
									reader = csv.reader(h)
									paths[int(Px)-1] = list(reader)

						with open(os.path.join(dirs,"overallDistance.csv"),'w') as g:
							writer = csv.writer(g)
							writer.writerow(pxNames)
							writer.writerow(distances)

						#Generate Combined image
						plt.figure(figsize=(20,10),dpi=100)
						plt.axhline(y=0, color='black')
						plt.xlabel("Time (s)")
						plt.ylabel("Seconds ahead (s)")
						l = 0
						for path in paths:
							x = []
							last = int()
							lowestPath = []
							for i in range(len(path)):
								if(path[i][0] != last):
									lowestPath.append(path[i])
									last = path[i][0]
							for i in range(len(lowestPath)):
								x.append(lowestPath[i][1])

							y = []
							for i in range(len(lowestPath)):
								y.append((((int(lowestPath[i][1])-int(lowestPath[i][0]))*11)/2205))
								tempx = list(map(int,x))
							x = []
							for xval in tempx:
								x.append((xval*11)/2205)
							
							plt.plot(x,y,label=pxNames[l])
							l+=1
						plt.legend()
						plt.savefig(os.path.join(dirs,"combinedLag.png"))
						print(dir)
						plt.close()
