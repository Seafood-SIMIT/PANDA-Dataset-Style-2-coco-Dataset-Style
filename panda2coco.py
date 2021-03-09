import os
import mmcv

train_annos_dir = '/Volumes/UNTITLED/Database/pANDA/images/image_annos/person_split_bbox_train.json'
root_path='/home/seafood/workdir/mmlib/mmdetection-master/data/splitimage_train/image_train'
out_file = '/home/seafood/workdir/mmlib/mmdetection-master/data/splitimage_train/image_annos/person_split_bbox_train_cocostyle.json'
def getFilesname():
    
    name_firstdir = os.listdir(root_path)
    name_images=[]
    for i in range(0,len(name_firstdir)):
        if name_firstdir[i].startswith('.'):
            continue
        name_images.append(name_firstdir[i])
    return name_images

def getHeightWeight(data_info):
    image_size = data_info["image size"]
    return image_size["height"], image_size["width"]
def getID(data_info):
    return data_info["image id"]
def getBboxUnsureP(object,height, width, person):
    if person:
        rect = object['rects']['visible body']
    else:
        rect = object['rect']
    tl = rect['tl']
    x_tl = tl['x'] * width
    y_tl = tl['y'] * height
    br = rect['br']
    x_br = br['x'] * width
    y_br = br['y'] * height
    return x_tl, y_tl, x_br, y_br
    
def getCategory(object):
    crow = 0
    cate = 0
    if object['category'] == 'person':
        cate = 1
    elif object['category'] == 'crowd':
        crow = 1
    return cate, crow
#def convert_panda_to_coco(ann_file, out_file, image_prefix):
name_images = getFilesname()
data_infos = mmcv.load(train_annos_dir)

annotations = []
images = []
obj_count = 0
for idx in range(len(name_images)):
    filename = name_images[idx]
    img_path = os.path.join(root_path, filename)
    
    height, width = getHeightWeight(data_infos[filename])
    img_id = getID(data_infos[filename])
    images.append(dict(
        id = img_id,
        file_name = filename,
        height = height,
        width = width))
    
    bboxes = []
    labels = []
    masks = []
    
    for idx , obj in enumerate(data_infos[filename]['objects list']):
        category_id,iscrowd = getCategory(obj)
        #if target is real person
        x_min, y_min, x_max, y_max = getBboxUnsureP(obj,height, width,category_id)
    
        data_anno = dict(
            image_id = img_id,
            id = obj_count,
            category_id=category_id,
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
            area = (x_max - x_min) * (y_max - y_min),
            segmentation = 0,
            iscrowd=iscrowd)
        annotations.append(data_anno)
        obj_count+=1

categories=[
    {'id':0,'name':'unsurep'},
    {'id':1,'name':'person'}
    ]
    #{'id':2,'name':'vehicle'},
    #{'id':3,'name':'unsurev'}]

coco_format_json = dict(
    images = images,
    annotations = annotations,
    categories = categories
    )
mmcv.dump(coco_format_json, out_file)