import os
import shutil
from pathlib import Path
import sys

"""
DO NOT RUN ON ALREADY PROCESSED FOLDERS
It will delete files and mess up the naming system
"""

rootdir = '' #Unzipped, named folders, will be deleted
dest = '' #Has sorted, named images in folders by person


#deleteExcess
for root, dirs, files in os.walk(rootdir, topdown=False):
    for name in dirs:
        if(name.startswith("jtxt") or name.startswith("store") or name.startswith("_Data") or name.startswith("cdnjs") or name.startswith("fonts") or name.startswith("ssl")):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)

#moveFolders
m = 0
for roots, dirs, files in os.walk(rootdir):
    for dir in dirs:
        if(not(dir.startswith("web") or dir.startswith("www"))): #or dir.startswith("bin"))):
            if(os.path.isdir(os.path.join(dest, dir))):
                new = dir+str(m)
                os.rename(os.path.join(roots, dir), os.path.join(dest, new))
                m = m+1
            else:
                shutil.move(os.path.join(roots, dir), dest)
print("Folders moved to dest")
shutil.rmtree(rootdir, ignore_errors=True) #deletes rootdir

#autoName
for rootss, dirss, filess in os.walk(dest):
    for dir in dirss: #3 2 3 5 2 2 4 2
        i = 1
        for roots, dirs, files in os.walk(os.path.join(rootss, dir)):
            for file in files:
                if(not file.startswith(".")):
                    splt = dir.split()
                    prefix = splt[0].lower()+"_"+splt[1].lower()
                    newFileName = prefix+"-"+str(i)+".png"
                    os.rename(os.path.join(roots, file), os.path.join(roots, newFileName))
                    i = i+1
print("Files renamed")
