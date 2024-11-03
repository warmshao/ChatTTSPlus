# -*- coding: utf-8 -*-
# @Time    : 2024/11/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: extract_files_to_texts.py

import datetime
import os
import pdb

import fitz
from tqdm import tqdm
import os
import cv2
from paddleocr import PPStructure, save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
from copy import deepcopy


def pdf2images(pdf_file):
    image_dir = os.path.splitext(pdf_file)[0] + "_images"
    os.makedirs(image_dir, exist_ok=True)

    pdfDoc = fitz.open(pdf_file)
    totalPage = pdfDoc.page_count
    for pg in tqdm(range(totalPage)):
        page = pdfDoc[pg]
        rotate = int(0)
        zoom_x = 2
        zoom_y = 2
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(os.path.join(image_dir, f"page_{pg + 1:03d}.png"))
    print(f"save images of pdf at: {image_dir}")
    return image_dir


def extract_pdf_to_txt(pdf_file, save_txt_file=None, lang='ch'):
    # save_folder = os.path.splitext(pdf_file)[0] + "_txts"
    # os.makedirs(save_folder, exist_ok=True)
    table_engine = PPStructure(recovery=True, lang=lang, show_log=False)
    if save_txt_file is None:
        save_txt_file = os.path.splitext(pdf_file)[0] + ".txt"
    pdf_image_dir = pdf2images(pdf_file)
    text = []
    imgs = sorted(os.listdir(pdf_image_dir))
    for img_name in tqdm(imgs, total=len(imgs)):
        img = cv2.imread(os.path.join(pdf_image_dir, img_name))
        result = table_engine(img)
        # save_structure_res(result, save_folder, os.path.splitext(img_name)[0])
        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)
        # convert_info_docx(img, res, save_folder, os.path.splitext(img_name)[0])
        for line in res:
            line.pop('img')
            for pra in line['res']:
                text.append(pra['text'])
            text.append('\n')
    with open(save_txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
    print(f"save txt of pdf at: {save_txt_file}")
    return save_txt_file


def extract_image_to_txt(image_file, save_txt_file=None, lang='ch'):
    save_folder = os.path.splitext(pdf_file)[0] + "_txts"
    os.makedirs(save_folder, exist_ok=True)
    table_engine = PPStructure(recovery=True, lang=lang, show_log=False)
    if save_txt_file is None:
        save_txt_file = os.path.splitext(pdf_file)[0] + ".txt"
    if os.path.isdir(image_file):
        imgs = [os.path.join(image_file, img_) for img_ in os.listdir(image_file) if
                img_.split()[-1].lower() in ["jpg", "png", "jpeg"]]
        imgs = sorted(imgs)
    else:
        imgs = [image_file]
    text = []
    for img_path in tqdm(imgs, total=len(imgs)):
        img = cv2.imread(img_path)
        result = table_engine(img)
        # save_structure_res(result, save_folder, os.path.splitext(img_name)[0])
        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)
        # convert_info_docx(img, res, save_folder, os.path.splitext(img_name)[0])
        for line in res:
            line.pop('img')
            for pra in line['res']:
                text.append(pra['text'])
            text.append('\n')
    with open(save_txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
    print(f"save txt of pdf at: {save_txt_file}")
    return save_txt_file


if __name__ == '__main__':
    pdf_file = "../../data/pdfs/MIMO.pdf"
    # pdf2images(pdf_file)
    extract_pdf_to_txt(pdf_file, lang='en')
