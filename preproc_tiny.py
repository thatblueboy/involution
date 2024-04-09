import os
import re
def create_dir(base_path, classname):
    path = base_path + classname
    if not os.path.exists(path):
        os.mkdir(path)

def reorg(filename, base_path, wordmap):
    print(len(wordmap))
    with open('tiny-imagenet-200/val/val_annotations.txt') as vals:
        for line in vals:
            vals = line.split()
            imagename = vals[0]
            print(vals[1])
            classname = wordmap[vals[1]]
            if os.path.exists(base_path+imagename):
                print(base_path+imagename, base_path+classname+'/'+imagename)
                os.rename(base_path+imagename,  base_path+classname+'/'+imagename)


wordmap = {}
with open('tiny-imagenet-200/words.txt') as words, open('tiny-imagenet-200/wnids.txt') as wnids:
    for line in wnids:
        vals = line.split()
        wordmap[vals[0]] = ""
    for line in words:
        vals = line.split()
        if vals[0] in wordmap:
            single_words = vals[1:]
            classname =  re.sub(",", "", single_words[0])
            if len(single_words) >= 2:
                classname += '_'+re.sub(",", "", single_words[1])
            wordmap[vals[0]] = classname
            create_dir('tiny-imagenet-200/val/images/', classname)
            if os.path.exists('tiny-imagenet-200/train/'+vals[0]):
                os.rename('tiny-imagenet-200/train/'+vals[0], 'tiny-imagenet-200/train/'+classname)
            #create_dir('./test/images/', single_words[0])
            #create_dir('./train/images/', single_words[0])


reorg('tiny-imagenet-200/val/val_annotations.txt', 'tiny-imagenet-200/val/images/', wordmap)