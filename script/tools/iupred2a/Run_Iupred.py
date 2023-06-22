
import pandas as pd
import numpy as np
import subprocess
import os
import shutil, errno
# from pathlib import Path
from pathlib2 import Path

Dataset_Name_F =open("./Dataset_Name.txt" ,"r")

Dataset_Name=Dataset_Name_F.readline()
Dataset_Name=Dataset_Name.strip()
print(Dataset_Name)

id_list =open("../../../../Dataset/"+Dataset_Name+"/id_list.txt" ,"r")

output_dir1 ="../../../../Dataset/"+Dataset_Name+"/iupred2a_short/"

if not os.path.exists(output_dir1):
	os.makedirs(output_dir1)

output_dir2 ="../../../../Dataset/"+Dataset_Name+"/iupred2a_long/"

if not os.path.exists(output_dir2):
	os.makedirs(output_dir2)	

for pid in id_list:
	print(pid)
	pid=pid.strip()
	bashCommand='python iupred2a.py -a ../../../../Dataset/'+Dataset_Name+'/FASTA/'+pid+'.fasta short > '+output_dir1+pid+'.txt'
	print(bashCommand)
	output = subprocess.check_output(['bash','-c', bashCommand])
	print(output)

	bashCommand='python iupred2a.py  ../../../../Dataset/'+Dataset_Name+'/FASTA/'+pid+'.fasta long > '+output_dir2+pid+'.txt'
	print(bashCommand)
	output = subprocess.check_output(['bash','-c', bashCommand])
	print(output)