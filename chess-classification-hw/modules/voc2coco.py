import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

'''
    From RoboFlow:  https://github.com/roboflow-ai/voc2coco/tree/master/sample
    
'''


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def conv(ann_dir, ann_ids, ann_paths_list, labels, output, ext):

    
#     parser = argparse.ArgumentParser(
#         description='This script support converting voc format xmls to coco format json')
#     parser.add_argument('--ann_dir', type=str, default=None,
#                         help='path to annotation files directory. It is not need when use --ann_paths_list')
#     parser.add_argument('--ann_ids', type=str, default=None,
#                         help='path to annotation files ids list. It is not need when use --ann_paths_list')
#     parser.add_argument('--ann_paths_list', type=str, default=None,
#                         help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
#     parser.add_argument('--labels', type=str, default=None,
#                         help='path to label list.')
#     parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
#     parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
#     args = parser.parse_args()
    
#     label2id = get_label2id(labels_path=args.labels)
    
#     ann_paths = get_annpaths(
#         ann_dir_path=args.ann_dir,
#         ann_ids_path=args.ann_ids,
#         ext=args.ext,
#         annpaths_list_path=args.ann_paths_list
#     )
#     convert_xmls_to_cocojson(
#         annotation_paths=ann_paths,
#         label2id=label2id,
#         output_jsonpath=args.output,
#         extract_num_from_imgid=True
#     )
    
    label2id = get_label2id(labels_path=labels)
    
    ann_paths = get_annpaths(
        ann_dir_path=ann_dir,
        ann_ids_path=ann_ids,
        ext=ext,
        annpaths_list_path=ann_paths_list
    )

    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=output,
        # extract_num_from_imgid=True
        extract_num_from_imgid=False
    )
    
def write_annotes_fn(input_dir, output_dir):
    
    annot_ids_fn = output_dir + 'annots.txt'
    
    xmls = [f for f in os.listdir(input_dir) if f[-4:] == '.xml']
    
    with open(annot_ids_fn, 'w') as f:
        f.write('\n'.join(xmls))
        

def write_labels(output_dir):

    label_fn = output_dir + 'labels.txt'

    labels = [
    'black-bishop',
    'black-king',
    'black-knight',
    'black-pawn',
    'black-queen',
    'black-rook',
    'white-bishop',
    'white-king',
    'white-knight',
    'white-pawn',
    'white-queen',
    'white-rook']

    with open(label_fn,'w') as f:
        f.write('\n'.join(labels))

def main_conv(
        input_dir,
        output_dir,
        b_summary=False,
        ):
    
    write_annotes_fn(input_dir, output_dir)
    
    write_labels(output_dir)
    
    conv(ann_dir=input_dir, 
         ann_ids=output_dir + 'annots.txt', 
         ann_paths_list=None, 
         labels=output_dir + 'labels.txt', 
         output=output_dir + 'coco.json', 
         ext='')
    
    if b_summary:
        with open(output_dir + 'coco.json') as f:
            coco = json.load(f)
        print(f"Created COCO json with {len(coco['annotations'])} annotations and {len(coco['images'])} images")
        
