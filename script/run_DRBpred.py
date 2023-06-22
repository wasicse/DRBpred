# %%
# Author: Md Wasi Ul Kabir
# Date: December 28 2020
import os
import pandas as pd
import subprocess
import pathlib 
import joblib
import pandas as pd
import numpy as np
import docker
from pathlib import Path
from optparse import OptionParser

# check if any files are missing
def checkmissing(source_dir):
    if not os.path.exists(source_dir):
        print("Missing", source_dir )
        missing_list.append(pid)
        
# check if datafarme has NAN values    
def checkforNAN(df,seqlength):
    if df.isnull().values.any():
        print("NAN")
        print(df[df.isnull().any(axis=1)])     
   
    if df.shape[0]!=seqlength:
        print("Length Mismatch", df.shape[0], seqlength)

# merge all features into one file     
def mergedata(PrintP,label):
    id_list =open("../Dataset/example/id_list.txt" ,"r")
    dnaoutput_dir ="../output/dna_merge_features/"
    rnaoutput_dir ="../output/rna_merge_features/"
    output_file_end=".csv"
    if not os.path.exists(dnaoutput_dir):
        os.makedirs(dnaoutput_dir)
    if not os.path.exists(rnaoutput_dir):
        os.makedirs(rnaoutput_dir)
    fasta_dir="../Dataset/example/FASTA/"
        
    missing_list=[]
    flag=0
    for pid in id_list:
        # print(pid)
        pid=pid.strip()
        # output_file_name=dnaoutput_dir+pid+output_file_end
        
        read_fasta= open(fasta_dir+pid+".fasta", "r")
        fasta=read_fasta.readline()
        fasta=read_fasta.readline()
        # print(type(fasta))
        seqlength=fasta[:-1].__len__()
        print(pid, seqlength)   
        #Dispredict Features      

        source_dir="../Features/Features/DisPredict/Features/"+pid+"/"+pid+".57pfeatures"                     
        dispredict_column=['O/D(1)', 'AA(1)', 'PP(1)', 'PP(2)', 'PP(3)', 'PP(4)', 'PP(5)', 'PP(6)', 'PP(7)', 'PSSM(1)', 'PSSM(2)', 'PSSM(3)', 'PSSM(4)', 'PSSM(5)', 'PSSM(6)', 'PSSM(7)', 'PSSM(8)', 'PSSM(9)', 'PSSM(10)', 'PSSM(11)', 'PSSM(12)', 'PSSM(13)', 'PSSM(14)', 'PSSM(15)', 'PSSM(16)', 'PSSM(17)', 'PSSM(18)', 'PSSM(19)', 'PSSM(20)', 'SS(1)', 'SS(2)', 'SS(3)', 'ASA(1)', 'dphi(1)', 'dpsi(1)', 'MG(1)', 'BG(1)', 'BG(2)', 'BG(3)', 'BG(4)', 'BG(5)', 'BG(6)', 'BG(7)', 'BG(8)', 'BG(9)', 'BG(10)', 'BG(11)', 'BG(12)', 'BG(13)', 'BG(14)', 'BG(15)', 'BG(16)', 'BG(17)', 'BG(18)', 'BG(19)', 'BG(20)', 'sPSEE(1)', 't(1)']
        df=pd.read_csv(source_dir,delim_whitespace=True,header=None,skiprows=1 )
        df.columns=dispredict_column
        drop_col= ["ASA(1)","dphi(1)","dpsi(1)","O/D(1)","PSSM(1)","PSSM(2)","PSSM(3)","PSSM(4)","PSSM(5)","PSSM(6)","PSSM(7)","PSSM(8)","PSSM(9)","PSSM(10)","PSSM(11)","PSSM(12)","PSSM(13)","PSSM(14)","PSSM(15)","PSSM(16)","PSSM(17)","PSSM(18)","PSSM(19)","PSSM(20)"]
        df1=df.drop(drop_col, axis=1)
        df1.columns=["Dis_"+ s for s in  df1.columns]
        if PrintP: print("Dispredict", df1.shape) 
        checkforNAN(df1,seqlength)
        if PrintP: print("Dispredict", df1.shape)  
                
        #Dispredict Probability

        source_dir="../Features/Features/DisPredict/prediction/"+pid+"/DisPredict2/"+pid+".drp"
    
        df=pd.read_csv(source_dir,delim_whitespace=True,skiprows=6 ,skipfooter=1,header=None, engine="python")
        df.columns=['Residue', 'Target','Dis_SVM_Probability']
        df2=df.drop([ "Residue","Target"], axis=1)
        if PrintP: print("DispredictProba",df2.shape)
    
    # # #PSSM Features    
        # source_dir="../Features/Features/PSSM_Parse/"+pid+".csv"  

        # df3=pd.read_csv(source_dir )
        # df3.columns=["PSI_"+ s for s in  df3.columns]
        # checkforNAN(df3,seqlength)
        # if PrintP: print("PSSM",df3.shape)
        
        # #Spider Features
        # source_dir="../Features/Features/Spider/"+pid.strip()+ ".i1"

        # df=pd.read_csv(source_dir,delim_whitespace=True)  
        # df4=df.drop([ "#","AA","SS","SS8"], axis=1)
        # df4.columns=["Spi_"+ s for s in  df4.columns]
        # checkforNAN(df4,seqlength)
        # if PrintP: print("Spider",df4.shape)

        
        # #OPAL Features
        source_dir="../Features/Features/OPAL/"+pid+".txt"
        df=pd.read_csv(source_dir,delim_whitespace=True )   
        df5=df.drop([ "No:","residues"], axis=1)
        df5.columns=["Opa_"+ s for s in  df5.columns]
        checkforNAN(df5,seqlength)
        if PrintP: print("OPAL",df5.shape)

        # #CNCC Features
        source_dir="../Features/Features/CNCC/"+pid+".csv"
        df6=pd.read_csv(source_dir)  
        checkforNAN(df6,seqlength)
        if PrintP: print("CNCC",df6.shape)
        
        #Iupred_shortAnchor
        source_dir="../Features/Features/iupred2a_short/"+pid+".txt"   
        df=pd.read_csv(source_dir,delim_whitespace=True,skiprows=7 ,header=None, engine="python")
        df.columns=["POS","RES","IUPRED2_short","ANCHOR2"]
        df7=df.drop([ "POS","RES"], axis=1)
        if PrintP: print("Iupred_shortAnchor",df7.shape)
    
        # #Iupred_long
        source_dir="../Features/Features/iupred2a_long/"+pid+".txt"

        df=pd.read_csv(source_dir,delim_whitespace=True,skiprows=7 ,header=None, engine="python")
        df.columns=["POS","RES","IUPRED2_long"]
        df8=df.drop([ "POS","RES"], axis=1)
        if PrintP: print("Iupred_long",df8.shape) 

        #SpotDisorder
        source_dir="../Features/Features/SpotDisorder/"+pid+".csv"   
        col= (   
            r"ASA,HSEa-u,HSEa-d,CN13,theta,tau,phi,psi,theta_c,tau_c,phi_c,psi_c,P(3-C),P(3-E),P(3-H),P(8-C),P(8-S),P(8-T),P(8-H),"
            r"P(8-G),P(8-I),P(8-E),P(8-B),HH_1,HH_2,HH_3,HH_4,HH_5,HH_6,HH_7,HH_8,HH_9,HH_10,HH_11,HH_12,HH_13,HH_14,HH_15,HH_16,"
            r"HH_17,HH_18,HH_19,HH_20,HH_21,HH_22,HH_23,HH_24,HH_25,HH_26,HH_27,HH_28,HH_29,HH_30,PSSMS_1,PSSMS_2,PSSMS_3,PSSMS_4,"
            r"PSSMS_5,PSSMS_6,PSSMS_7,PSSMS_8,PSSMS_9,PSSMS_10,PSSMS_11,PSSMS_12,PSSMS_13,PSSMS_14,PSSMS_15,PSSMS_16,PSSMS_17,PSSMS_18,PSSMS_19,PSSMS_20"
            )
        SpotDisorder_column=col.split(",")  
        SpotDisorder_column=[x.strip() for x in SpotDisorder_column ]
        df9=pd.read_csv(source_dir,index_col=0 )
        df9.columns=SpotDisorder_column
        checkforNAN(df9,seqlength)  
        df9.columns=["Spot_"+ s for s in  df9.columns]      
        if PrintP: print("SpotDisorder",df9.shape)  
        
        #SpotDisorder Probability
        source_dir="../Features/Features/SpotDisorderProba/"+pid+".spotd2"    
        df=pd.read_csv(source_dir,delim_whitespace=True,skiprows=1)  
        df10=df.drop([ "#","AA","O/D"], axis=1)
        df10.columns=["Spot_"+ s for s in  df10.columns]
        checkforNAN(df10,seqlength)
        if PrintP: print("SpotDisorderProba",df10.shape)

    # Target (Set Dummy Target for windowing code)    
        df0=pd.DataFrame()
        df0["Target"]=  range(df10.shape[0])
        print("Target",df0.shape)

    # code to add target
        # target_dir="../Feature_Extraction/Dataset/example/"+Angle+"target/"
        # target_dir_file_end=".csv"
        # source_dir=target_dir+pid+target_dir_file_end
        # df0=pd.read_csv(source_dir,index_col=0)   #,header=None
    
        # # df0.columns=["Target"]
        # if PrintP: print("Target",df0.shape)

        listdf=[df0,df1,df2,df5,df6,df7,df8,df9,df10]
        merged = pd.concat(listdf, axis=1)
        merged = merged.loc[:, ~merged.columns.str.contains('^Unnamed')] 
        if PrintP: print("Merge",merged.shape)  


        for dff in listdf:      
            if(dff.isnull().values.any()):
                prinf(dff)
                print("NAN")
                break;
            
        if(merged.isnull().values.any()):
            print("Merge files have NAN")

        dnaselected_feat=pd.read_csv("./DNASelected_Feat_96.csv")

        ss=["Target"]+dnaselected_feat.iloc[:,0].tolist()
  
        dnamerged = merged[ss]       
        dnamerged.to_csv(dnaoutput_dir+"/"+pid+'.csv',index=False,header=label) 

        rnaselected_feat=pd.read_csv("./RNASelected_Feat_96.csv")
        ss=["Target"]+rnaselected_feat.iloc[:,0].tolist()
   
        rnamerged = merged[ss]      
        rnamerged.to_csv(rnaoutput_dir+"/"+pid+'.csv',index=False,header=label) 
  
    return rnamerged.shape

# run windowing code
def windowing(len,startwindow_size,endwindow_size):
    
    windowintput=open("./window_param.txt","w")    
    windowintput.write(len+ '\n')
    windowintput.write(str(startwindow_size)+ '\n')
    windowintput.write(str(endwindow_size)+ '\n')
    windowintput.close()

    output_dir ="../output/dna_Windowed_file/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir ="../output/rna_Windowed_file/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bashCommand='javac run_windowing_dna.java '
    output = subprocess.check_output(['bash','-c', bashCommand])
    bashCommand='java run_windowing_dna'
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 

    bashCommand='javac run_windowing_rna.java '
    output = subprocess.check_output(['bash','-c', bashCommand])
    bashCommand='java run_windowing_rna'
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 


def getprediction():
    np.set_printoptions(precision=3)
    workspace="../output"
    pathlib.Path(workspace).mkdir(parents=True, exist_ok=True) 

    dnaX_test=pd.read_csv("../output/dna_Windowed_file/feat_96_w_11.csv",header=None).to_numpy()
    dnaX_test=dnaX_test[:,1:]

    rnaX_test=pd.read_csv("../output/rna_Windowed_file/feat_96_w_11.csv",header=None).to_numpy()
    rnaX_test=rnaX_test[:,1:]

    # scaler= joblib.load("../models/phi_scaler.pkl")
    # X_test = scaler.transform(X_test)


    dnasaved_model = joblib.load("../models/dna_model.pkl")
    dnaproba = dnasaved_model.predict_proba(dnaX_test)

    rnasaved_model = joblib.load("../models/rna_model.pkl")
    rnaproba = rnasaved_model.predict_proba(rnaX_test)
  

    id_list =open("../Dataset/example/id_list.txt" ,"r")
    start=0
    dnathreshold=0.42
    rnathreshold=0.40
    for pid in id_list:
        pid=pid.strip()
        fastafile=open("../Dataset/example/FASTA/"+pid+".fasta","r")
        fasta=fastafile.readline().strip()
        fasta=fastafile.readline().strip()    
        end=start+fasta.__len__()

        fdnaproba=dnaproba[start:end]
        dnapred = (fdnaproba[:,1] >= dnathreshold).astype(int)
      
        frnaproba=rnaproba[start:end] 
        rnapred = (frnaproba[:,1] >= rnathreshold).astype(int)   
        dnaresult=np.hstack((np.array(list(fasta)).astype(str).reshape(-1,1), np.round(fdnaproba[:,1], 3).astype(str).reshape(-1,1) ,dnapred.astype(str).reshape(-1,1))) 
        rnaresult=np.hstack((np.array(list(fasta)).astype(str).reshape(-1,1), np.round(frnaproba[:,1], 3).astype(str).reshape(-1,1) ,rnapred.astype(str).reshape(-1,1))) 
       
        with open("../output/"+pid+"_dnaPred.txt", "ab") as f:
            f.write((">"+pid+"\n").encode())
            fmt ='%s','%s','%s'
            np.savetxt(f, dnaresult, delimiter='\t',fmt=fmt)
      
        with open("../output/"+pid+"_rnaPred.txt", "ab") as f:
            f.write((">"+pid+"\n").encode())
            fmt ='%s','%s','%s'
            np.savetxt(f, rnaresult, delimiter='\t',fmt=fmt)

        start=fasta.__len__()           

# check if a container is running or not
def checkcontainerstatus(containername):
    cli = docker.APIClient()
    try:
        inspect_dict = cli.inspect_container(containername)
        state = inspect_dict['State']
        print(state)
        is_running = state['Status'] == 'running'

        if is_running:
            print("My container is running!")
            return False
    except:
        print("My container is not running!")
        return True

# collect features from docker container
def collectfeatures(containername, database_path):
    
    print("Removing old features from host")
    bashCommand="rm -rf ../Features/"
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 

    print("Pulling docker image")
    bashCommand="docker pull wasicse/featureextract:2.0"
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 

    print("Checking container status")
    bashCommand="docker ps -q -f name={"+containername+"}"
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 
    if checkcontainerstatus(containername):
        print("Creating container")
        bashCommand="docker run -itd -v "+database_path+":/opt/FeatureExtractionDocker/Databases --name "+containername+"  wasicse/featureextract:2.0"
        print(bashCommand)
        output = subprocess.check_output(['bash','-c', bashCommand])
        print(output.decode('utf-8')) 
    else:
        print("Starting container")
        bashCommand="docker start "+containername
        print(bashCommand)
        output = subprocess.check_output(['bash','-c', bashCommand])
        print(output.decode('utf-8'))   
        pass 
    
    print("Removing old features from container")
    bashCommand="docker exec "+containername+"  bash  -c \"rm -rf /opt/FeatureExtractionDocker/Dataset/example/*\""
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 

    print("Copying files to container")
    bashCommand="docker cp ../Dataset/example/ "+containername+":/opt/FeatureExtractionDocker/Dataset"
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 



    print("Running feature extraction. It might take a while to complete. and depends on the number of sequences in the input file.")
    bashCommand= "docker exec "+containername+" bash --login -c \"cd /opt/FeatureExtractionDocker/FeatureExtractTool ; /opt/.pyenv/versions/miniconda3-3.9-4.10.3/envs/feat/bin/python runScript.py\""
    print(bashCommand)
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 
    
    print("copying features from container")
    bashCommand="docker cp "+containername+":/opt/FeatureExtractionDocker/Dataset/example/ ../Features/"
    output = subprocess.check_output(['bash','-c', bashCommand])
    print(output.decode('utf-8')) 

    # print("Stop container")
    # bashCommand="docker stop "+containername
    # print(bashCommand)
    # output = subprocess.check_output(['bash','-c', bashCommand])
    # print(output.decode('utf-8'))  
    # 
    # Run Iupred local machine
    runIupred() 

def runIupred():

    id_list =open("../Dataset/example/id_list.txt" ,"r")

    output_dir1 ="../Features/Features/iupred2a_short/"

    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)

    output_dir2 ="../Features/Features/iupred2a_long/"

    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)	

    for pid in id_list:
        # print(pid)
        pid=pid.strip()
        bashCommand='python tools/iupred2a/iupred2a.py -a ../Dataset/example/FASTA/'+pid+'.fasta short > '+output_dir1+pid+'.txt'
        print(bashCommand)
        output = subprocess.check_output(['bash','-c', bashCommand])
        print(output)

        bashCommand='python tools/iupred2a/iupred2a.py  ../Dataset/example/FASTA/'+pid+'.fasta long > '+output_dir2+pid+'.txt'
        print(bashCommand)
        output = subprocess.check_output(['bash','-c', bashCommand])
        print(output)

if __name__ == "__main__":  

    parent_path = str(Path(__file__).resolve().parents[1])
    print("Parent Dir",parent_path)

    parser = OptionParser()
    parser.add_option("-f", "--containerName", dest="containerName", help="Container Name.", default="featuresv1")
    parser.add_option("-o", "--output_path", dest="output_path", help="Path to output.", default=parent_path+'/output/')
    parser.add_option("-d", "--database_path", dest="database_path", help="Path to database.", default=parent_path+'/script/Databases/')
    
    (options, args) = parser.parse_args()
    print("Database Path:",options.database_path)
    
    #Check if required Databases are in  script/Databases directory.    
       
    # Getting the list of directories
    dir = os.listdir(options.database_path)
      
    # Checking if the list is empty or not
    if len(dir) == 0:
        print("Please download the nr and uniclust30_2017_04 databases in script/Databases directory  ")
    else:       


        print("Container Name:",options.containerName)
        print("Output Path:",options.output_path)
        #Print features options
        PrintP=False
        #Add label to features
        label=True

        # # Collect features by running the docker container
        # collectfeatures(options.containerName, options.database_path )

        # # # Merge features
        # merge_shape=mergedata(PrintP,label)

        # # Windowing 
        # startwindow_size=11
        # endwindow_size=11
        # windowing(str(merge_shape[1]-1),startwindow_size,endwindow_size)
        
        # Get prediction from the windowed features
        getprediction()



