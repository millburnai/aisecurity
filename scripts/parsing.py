import os
import shutil
from pathlib import Path
import sys

"""
DO NOT RUN ON ALREADY PROCESSED FOLDERS
It will delete files and mess up the naming system
"""

rootdir = '' #Unzipped folders, will be deleted
dest = '' #Has named images in folders sorted by person


#deleteExcess
for root, dirs, files in os.walk(rootdir, topdown=False):
    for name in dirs:
        if(name.startswith("jtxt") or name.startswith("store") or name.startswith("_Data") or name.startswith("cdnjs") or name.startswith("fonts") or name.startswith("ssl")):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
#moveFolders
m = 0
for roots, dirs, files in os.walk(rootdir):
    for dir in dirs:
        if(not(dir.startswith("web") or dir.startswith("www"))):
            if(os.path.isdir(os.path.join(dest, dir))):
                new = dir+str(m)
                os.rename(os.path.join(roots, dir), os.path.join(dest, new))
                m = m+1
            else:
                shutil.move(os.path.join(roots, dir), dest)
print("Folders moved to dest")
shutil.rmtree(rootdir, ignore_errors=True) #deletes rootdir


#autoName
#Run after naming folders
"""
for rootss, dirss, filess in os.walk(rootdir):
    for dir in dirss:
        i = 1
        for roots, dirs, files in os.walk(os.path.join(rootss, dir)):
            for file in files:
                if(not file.startswith(".") and not file.startswith("bin")):
                    splt = dir.split()
                    length = len(splt)
                    for k in range(length):
                        if(k==0):
                            prefix = splt[k].lower()
                        else:
                            prefix = prefix+"_"+splt[k].lower()
                    newFileName = prefix+"-"+str(i)+".png"
                    os.rename(os.path.join(roots, file), os.path.join(roots, newFileName))
                    i = i+1
print("Files renamed")
"""

#Moves all files to 1 folder
"""
#dest = '' #works with previous code or diff folder
allImages = '' #Path for new folder

for roots, dirs, files in os.walk(dest):
    for file in files:
            shutil.move(os.path.join(roots, file), allImages)
"""
