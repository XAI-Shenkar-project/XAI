import sys
import ast
import numpy as np
import tensorflow as tf
import logging
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFontMetrics
from nltk.corpus import wordnet as wn
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import uic
from PyQt5.QtWidgets import QScrollArea

ResNet_hierarchical_data = {
    'directed graph': [
            ['Madagascar_cat', 'pug'], ['boxer', 'Madagascar_cat', 'pug'], ['Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['tabby', 'Egyptian_cat'], ['catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['airliner', 'warplane'], ['orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['broccoli', 'cauliflower'], ['black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['Granny_Smith', 'head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['fig', 'Granny_Smith', 'head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['zucchini', 'fig', 'Granny_Smith', 'head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['Persian_cat', 'zucchini', 'fig', 'Granny_Smith', 'head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['minivan', 'police_van'], ['limousine', 'minivan', 'police_van'], ['jeep', 'limousine', 'minivan', 'police_van'],['Persian_cat', 'zucchini', 'fig', 'Granny_Smith', 'head_cabbage', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'space_shuttle', 'flamingo', 'white_stork', 'airliner', 'warplane', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'jeep', 'limousine', 'minivan', 'police_van']
        ],
    'undirected graph': [
            ['Norwich_terrier', 'kuvasz'], ['sports_car', 'Norwich_terrier', 'kuvasz'], ['Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['tabby', 'Egyptian_cat'], ['wok', 'frying_pan'], ['teapot', 'coffeepot'], ['chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['orange', 'lemon'], ['tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['broccoli', 'cauliflower'], ['airliner', 'warplane'], ['Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['minivan', 'limousine'], ['American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['Granny_Smith', 'broccoli', 'cauliflower'], ['caldron', 'wok', 'frying_pan'], ['space_shuttle', 'airliner', 'warplane'], ['orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower'], ['pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower'], ['white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower'], ['Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower'], ['teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['container_ship', 'space_shuttle', 'airliner', 'warplane'], ['police_van', 'jeep'], ['zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['minivan', 'limousine', 'zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['container_ship', 'space_shuttle', 'airliner', 'warplane', 'minivan', 'limousine', 'zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['police_van', 'jeep', 'container_ship', 'space_shuttle', 'airliner', 'warplane', 'minivan', 'limousine', 'zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['fireboat', 'police_van', 'jeep', 'container_ship', 'space_shuttle', 'airliner', 'warplane', 'minivan', 'limousine', 'zucchini', 'head_cabbage', 'fig', 'orange', 'lemon', 'Granny_Smith', 'broccoli', 'cauliflower', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'spider_monkey', 'Madagascar_cat', 'flamingo', 'white_stork', 'boxer', 'pug', 'black_swan', 'American_coot', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'trimaran', 'catamaran', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz']
        ],
    'distances graph': [
            ['limousine', 'jeep'], ['Granny_Smith', 'zucchini'], ['white_stork', 'flamingo'], ['minivan', 'police_van'], ['Persian_cat', 'tabby'], ['airliner', 'space_shuttle'], ['broccoli', 'head_cabbage'], ['Madagascar_cat', 'spider_monkey'], ['boxer', 'kuvasz'], ['warplane', 'airliner', 'space_shuttle'], ['American_coot', 'black_swan'], ['lemon', 'fig'], ['teapot', 'coffeepot'], ['cauliflower', 'lemon', 'fig'], ['Egyptian_cat', 'Persian_cat', 'tabby'], ['gorilla', 'Madagascar_cat', 'spider_monkey'], ['container_ship', 'fireboat'], ['wok', 'frying_pan'], ['catamaran', 'trimaran'], ['container_ship', 'fireboat', 'catamaran', 'trimaran'], ['Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig'], ['chimpanzee', 'gorilla', 'Madagascar_cat', 'spider_monkey'], ['pug', 'boxer', 'kuvasz'], ['Crock_Pot', 'wok', 'frying_pan'], ['sports_car', 'limousine', 'jeep'], ['container_ship', 'fireboat', 'catamaran', 'trimaran', 'sports_car', 'limousine', 'jeep'], ['caldron', 'teapot', 'coffeepot'], ['minivan', 'police_van', 'container_ship', 'fireboat', 'catamaran', 'trimaran', 'sports_car', 'limousine', 'jeep'], ['Crock_Pot', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot'], ['Egyptian_cat', 'Persian_cat', 'tabby', 'pug', 'boxer', 'kuvasz'], ['warplane', 'airliner', 'space_shuttle', 'minivan', 'police_van', 'container_ship', 'fireboat', 'catamaran', 'trimaran', 'sports_car', 'limousine', 'jeep'], ['chimpanzee', 'gorilla', 'Madagascar_cat', 'spider_monkey', 'Egyptian_cat', 'Persian_cat', 'tabby', 'pug', 'boxer', 'kuvasz'], ['orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig'], ['broccoli', 'head_cabbage', 'orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig'], ['Norwich_terrier', 'chimpanzee', 'gorilla', 'Madagascar_cat', 'spider_monkey', 'Egyptian_cat', 'Persian_cat', 'tabby', 'pug', 'boxer', 'kuvasz'], ['white_stork', 'flamingo', 'American_coot', 'black_swan'], ['broccoli', 'head_cabbage', 'orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig', 'white_stork', 'flamingo', 'American_coot', 'black_swan'], ['Crock_Pot', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'broccoli', 'head_cabbage', 'orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig', 'white_stork', 'flamingo', 'American_coot', 'black_swan'], ['Norwich_terrier', 'chimpanzee', 'gorilla', 'Madagascar_cat', 'spider_monkey', 'Egyptian_cat', 'Persian_cat', 'tabby', 'pug', 'boxer', 'kuvasz', 'Crock_Pot', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'broccoli', 'head_cabbage', 'orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig', 'white_stork', 'flamingo', 'American_coot', 'black_swan'], ['warplane', 'airliner', 'space_shuttle', 'minivan', 'police_van', 'container_ship', 'fireboat', 'catamaran', 'trimaran', 'sports_car', 'limousine', 'jeep', 'Norwich_terrier', 'chimpanzee', 'gorilla', 'Madagascar_cat', 'spider_monkey', 'Egyptian_cat', 'Persian_cat', 'tabby', 'pug', 'boxer', 'kuvasz', 'Crock_Pot', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'broccoli', 'head_cabbage', 'orange', 'Granny_Smith', 'zucchini', 'cauliflower', 'lemon', 'fig', 'white_stork', 'flamingo', 'American_coot', 'black_swan']
        ]
}

VGG_hierarchical_data = {
    'directed graph': [
            ['Persian_cat', 'Madagascar_cat'], ['pug', 'Persian_cat', 'Madagascar_cat'], ['boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'],['trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['broccoli', 'cauliflower'], ['chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['head_cabbage', 'broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['American_coot', 'head_cabbage', 'broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['black_swan', 'American_coot', 'head_cabbage', 'broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['police_van', 'limousine'], ['jeep', 'police_van', 'limousine'], ['black_swan', 'American_coot', 'head_cabbage', 'broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'jeep', 'police_van', 'limousine'], ['minivan', 'black_swan', 'American_coot', 'head_cabbage', 'broccoli', 'cauliflower', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'warplane', 'airliner', 'lemon', 'orange', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'jeep', 'police_van', 'limousine']
        ],
    'undirected graph': [
            ['Madagascar_cat', 'Norwich_terrier'], ['kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['catamaran', 'trimaran'], ['orange', 'lemon'], ['broccoli', 'cauliflower'], ['teapot', 'coffeepot'], ['airliner', 'warplane'], ['wok', 'frying_pan'], ['sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower'], ['chimpanzee', 'spider_monkey'], ['tabby', 'Egyptian_cat'], ['sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['minivan', 'police_van'], ['jeep', 'minivan', 'police_van'], ['tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['limousine', 'jeep', 'minivan', 'police_van'], ['catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['caldron', 'wok', 'frying_pan'], ['fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey'], ['teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan'], ['Granny_Smith', 'zucchini'], ['space_shuttle', 'airliner', 'warplane'], ['fig', 'Granny_Smith', 'zucchini'], ['orange', 'lemon', 'fig', 'Granny_Smith', 'zucchini'], ['boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey', 'space_shuttle', 'airliner', 'warplane'], ['orange', 'lemon', 'fig', 'Granny_Smith', 'zucchini', 'boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey', 'space_shuttle', 'airliner', 'warplane'], ['teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'orange', 'lemon', 'fig', 'Granny_Smith', 'zucchini', 'boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey', 'space_shuttle', 'airliner', 'warplane'], ['Crock_Pot', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'orange', 'lemon', 'fig', 'Granny_Smith', 'zucchini', 'boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey', 'space_shuttle', 'airliner', 'warplane'], ['container_ship', 'Crock_Pot', 'teapot', 'coffeepot', 'caldron', 'wok', 'frying_pan', 'orange', 'lemon', 'fig', 'Granny_Smith', 'zucchini', 'boxer', 'pug', 'fireboat', 'flamingo', 'white_stork', 'Persian_cat', 'limousine', 'jeep', 'minivan', 'police_van', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'tabby', 'Egyptian_cat', 'head_cabbage', 'gorilla', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'broccoli', 'cauliflower', 'chimpanzee', 'spider_monkey', 'space_shuttle', 'airliner', 'warplane']
        ],
    'distances graph': [
            ['police_van', 'limousine'], ['sports_car', 'police_van', 'limousine'], ['American_coot', 'black_swan'], ['wok', 'frying_pan'], ['head_cabbage', 'cauliflower'], ['tabby', 'Egyptian_cat'], ['lemon', 'head_cabbage', 'cauliflower'], ['warplane', 'space_shuttle'], ['fireboat', 'warplane', 'space_shuttle'], ['catamaran', 'trimaran'], ['white_stork', 'flamingo'], ['teapot', 'coffeepot'], ['pug', 'boxer'], ['airliner', 'fireboat', 'warplane', 'space_shuttle'], ['Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['minivan', 'jeep'], ['chimpanzee', 'gorilla'], ['spider_monkey', 'chimpanzee', 'gorilla'], ['caldron', 'Crock_Pot'], ['zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['fig', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['teapot', 'coffeepot', 'caldron', 'Crock_Pot'], ['wok', 'frying_pan', 'teapot', 'coffeepot', 'caldron', 'Crock_Pot'], ['American_coot', 'black_swan', 'white_stork', 'flamingo'], ['sports_car', 'police_van', 'limousine', 'minivan', 'jeep'], ['container_ship', 'airliner', 'fireboat', 'warplane', 'space_shuttle'], ['broccoli', 'fig', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla'], ['Norwich_terrier', 'kuvasz'], ['catamaran', 'trimaran', 'container_ship', 'airliner', 'fireboat', 'warplane', 'space_shuttle'], ['pug', 'boxer', 'Norwich_terrier', 'kuvasz'], ['Persian_cat', 'tabby', 'Egyptian_cat'], ['pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['American_coot', 'black_swan', 'white_stork', 'flamingo', 'pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['wok', 'frying_pan', 'teapot', 'coffeepot', 'caldron', 'Crock_Pot', 'broccoli', 'fig', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower'], ['catamaran', 'trimaran', 'container_ship', 'airliner', 'fireboat', 'warplane', 'space_shuttle', 'Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['wok', 'frying_pan', 'teapot', 'coffeepot', 'caldron', 'Crock_Pot', 'broccoli', 'fig', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower', 'catamaran', 'trimaran', 'container_ship', 'airliner', 'fireboat', 'warplane', 'space_shuttle', 'Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['sports_car', 'police_van', 'limousine', 'minivan', 'jeep', 'wok', 'frying_pan', 'teapot', 'coffeepot', 'caldron', 'Crock_Pot', 'broccoli', 'fig', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'head_cabbage', 'cauliflower', 'catamaran', 'trimaran', 'container_ship', 'airliner', 'fireboat', 'warplane', 'space_shuttle', 'Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'pug', 'boxer', 'Norwich_terrier', 'kuvasz', 'Persian_cat', 'tabby', 'Egyptian_cat']        
        ]
}

GoogLeNet_hierarchical_data = {
    'directed graph': [
            ['Persian_cat', 'Madagascar_cat'], ['pug', 'Persian_cat', 'Madagascar_cat'], ['boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['broccoli', 'cauliflower'], ['white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['head_cabbage', 'broccoli', 'cauliflower'], ['airliner', 'warplane'], ['zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['minivan', 'airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['limousine', 'minivan', 'airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['American_coot', 'limousine', 'minivan', 'airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['black_swan', 'American_coot', 'limousine', 'minivan', 'airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['space_shuttle', 'black_swan', 'American_coot', 'limousine', 'minivan', 'airliner', 'warplane', 'flamingo', 'white_stork', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fig', 'Granny_Smith', 'spider_monkey', 'Crock_Pot', 'caldron', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower']
        ],
    'undirected graph': [
            ['Norwich_terrier', 'kuvasz'], ['sports_car', 'Norwich_terrier', 'kuvasz'], ['Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['wok', 'frying_pan'], ['catamaran', 'trimaran'], ['tabby', 'Egyptian_cat'], ['teapot', 'coffeepot'], ['chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['zucchini', 'head_cabbage'], ['caldron', 'teapot', 'coffeepot'], ['pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['broccoli', 'cauliflower'], ['zucchini', 'head_cabbage', 'broccoli', 'cauliflower'], ['minivan', 'limousine'], ['American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz'], ['airliner', 'warplane'], ['wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot'], ['space_shuttle', 'white_stork'], ['police_van', 'minivan', 'limousine'], ['airliner', 'warplane', 'space_shuttle', 'white_stork'], ['flamingo', 'airliner', 'warplane', 'space_shuttle', 'white_stork'], ['black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine'], ['catamaran', 'trimaran', 'black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine'], ['jeep', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine'], ['container_ship', 'fireboat'], ['flamingo', 'airliner', 'warplane', 'space_shuttle', 'white_stork', 'container_ship', 'fireboat'], ['Madagascar_cat', 'jeep', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine'], ['Persian_cat', 'tabby', 'Egyptian_cat'], ['flamingo', 'airliner', 'warplane', 'space_shuttle', 'white_stork', 'container_ship', 'fireboat', 'Madagascar_cat', 'jeep', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine'], ['Granny_Smith', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'Granny_Smith', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['zucchini', 'head_cabbage', 'broccoli', 'cauliflower', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'Granny_Smith', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['flamingo', 'airliner', 'warplane', 'space_shuttle', 'white_stork', 'container_ship', 'fireboat', 'Madagascar_cat', 'jeep', 'catamaran', 'trimaran', 'black_swan', 'American_coot', 'boxer', 'pug', 'lemon', 'orange', 'gorilla', 'chimpanzee', 'fig', 'spider_monkey', 'Crock_Pot', 'sports_car', 'Norwich_terrier', 'kuvasz', 'police_van', 'minivan', 'limousine', 'zucchini', 'head_cabbage', 'broccoli', 'cauliflower', 'wok', 'frying_pan', 'caldron', 'teapot', 'coffeepot', 'Granny_Smith', 'Persian_cat', 'tabby', 'Egyptian_cat']
        ],
    'distances graph': [
            ['minivan', 'limousine'], ['American_coot', 'black_swan'], ['white_stork', 'flamingo'], ['frying_pan', 'caldron'], ['orange', 'fig'], ['boxer', 'kuvasz'], ['space_shuttle', 'fireboat'], ['chimpanzee', 'spider_monkey'], ['gorilla', 'chimpanzee', 'spider_monkey'], ['warplane', 'space_shuttle', 'fireboat'], ['pug', 'boxer', 'kuvasz'], ['catamaran', 'warplane', 'space_shuttle', 'fireboat'], ['American_coot', 'black_swan', 'white_stork', 'flamingo'], ['lemon', 'orange', 'fig'], ['container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat'], ['tabby', 'Egyptian_cat'], ['wok', 'frying_pan', 'caldron'], ['police_van', 'jeep'], ['Madagascar_cat', 'trimaran'], ['teapot', 'wok', 'frying_pan', 'caldron'], ['minivan', 'limousine', 'police_van', 'jeep'], ['zucchini', 'head_cabbage'], ['coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron'], ['cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat'], ['broccoli', 'zucchini', 'head_cabbage'], ['Granny_Smith', 'lemon', 'orange', 'fig'], ['airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat'], ['Persian_cat', 'tabby', 'Egyptian_cat'], ['broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig'], ['pug', 'boxer', 'kuvasz', 'Madagascar_cat', 'trimaran'], ['sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['Norwich_terrier', 'pug', 'boxer', 'kuvasz', 'Madagascar_cat', 'trimaran'], ['airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat', 'broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig'], ['coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat', 'Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat', 'broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig', 'coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat', 'Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['Norwich_terrier', 'pug', 'boxer', 'kuvasz', 'Madagascar_cat', 'trimaran', 'airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat', 'broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig', 'coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat', 'Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['gorilla', 'chimpanzee', 'spider_monkey', 'Norwich_terrier', 'pug', 'boxer', 'kuvasz', 'Madagascar_cat', 'trimaran', 'airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat', 'broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig', 'coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat', 'Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep'], ['American_coot', 'black_swan', 'white_stork', 'flamingo', 'gorilla', 'chimpanzee', 'spider_monkey', 'Norwich_terrier', 'pug', 'boxer', 'kuvasz', 'Madagascar_cat', 'trimaran', 'airliner', 'cauliflower', 'container_ship', 'catamaran', 'warplane', 'space_shuttle', 'fireboat', 'broccoli', 'zucchini', 'head_cabbage', 'Granny_Smith', 'lemon', 'orange', 'fig', 'coffeepot', 'teapot', 'wok', 'frying_pan', 'caldron', 'Persian_cat', 'tabby', 'Egyptian_cat', 'Crock_Pot', 'sports_car', 'minivan', 'limousine', 'police_van', 'jeep']
        ]
}

EfficientNet_hierarchical_data = {
    'directed graph': [
            ['Persian_cat', 'Madagascar_cat'], ['pug', 'Persian_cat', 'Madagascar_cat'], ['boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'lemon'], ['teapot', 'coffeepot'], ['tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['airliner', 'warplane'], ['broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['black_swan', 'American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'lemon', 'black_swan', 'American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Crock_Pot', 'orange', 'lemon', 'black_swan', 'American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Granny_Smith', 'Crock_Pot', 'orange', 'lemon', 'black_swan', 'American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fig', 'Granny_Smith', 'Crock_Pot', 'orange', 'lemon', 'black_swan', 'American_coot', 'teapot', 'coffeepot', 'space_shuttle', 'limousine', 'minivan', 'airliner', 'warplane', 'cauliflower', 'head_cabbage', 'broccoli', 'gorilla', 'chimpanzee', 'Egyptian_cat', 'tabby', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'spider_monkey', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'jeep', 'sports_car', 'police_van', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat']
        ],
    'undirected graph': [
            ['Madagascar_cat', 'pug'], ['boxer', 'Madagascar_cat', 'pug'], ['Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug'], ['wok', 'frying_pan'], ['tabby', 'Egyptian_cat'], ['orange', 'lemon'], ['chimpanzee', 'gorilla'], ['teapot', 'coffeepot'], ['broccoli', 'cauliflower'], ['airliner', 'warplane'], ['trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['space_shuttle', 'airliner', 'warplane'], ['white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['Crock_Pot', 'teapot', 'coffeepot'], ['wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot'], ['American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['Granny_Smith', 'orange', 'lemon'], ['tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['minivan', 'limousine'], ['police_van', 'jeep'], ['chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower'], ['minivan', 'limousine', 'police_van', 'jeep'], ['fig', 'Granny_Smith', 'orange', 'lemon'], ['container_ship', 'fireboat'], ['caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'minivan', 'limousine', 'police_van', 'jeep'], ['fig', 'Granny_Smith', 'orange', 'lemon', 'caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'minivan', 'limousine', 'police_van', 'jeep'], ['zucchini', 'fig', 'Granny_Smith', 'orange', 'lemon', 'caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'minivan', 'limousine', 'police_van', 'jeep'], ['space_shuttle', 'airliner', 'warplane', 'zucchini', 'fig', 'Granny_Smith', 'orange', 'lemon', 'caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'minivan', 'limousine', 'police_van', 'jeep'], ['container_ship', 'fireboat', 'space_shuttle', 'airliner', 'warplane', 'zucchini', 'fig', 'Granny_Smith', 'orange', 'lemon', 'caldron', 'spider_monkey', 'Persian_cat', 'wok', 'frying_pan', 'Crock_Pot', 'teapot', 'coffeepot', 'chimpanzee', 'gorilla', 'tabby', 'Egyptian_cat', 'black_swan', 'American_coot', 'flamingo', 'white_stork', 'head_cabbage', 'trimaran', 'catamaran', 'sports_car', 'kuvasz', 'Norwich_terrier', 'boxer', 'Madagascar_cat', 'pug', 'broccoli', 'cauliflower', 'minivan', 'limousine', 'police_van', 'jeep']
        ],
    'distances graph': [
            ['coffeepot', 'Crock_Pot'], ['trimaran', 'fireboat'], ['Granny_Smith', 'lemon'], ['American_coot', 'black_swan'], ['warplane', 'space_shuttle'], ['catamaran', 'trimaran', 'fireboat'], ['chimpanzee', 'gorilla'], ['tabby', 'Egyptian_cat'], ['spider_monkey', 'chimpanzee', 'gorilla'], ['container_ship', 'warplane', 'space_shuttle'], ['airliner', 'container_ship', 'warplane', 'space_shuttle'], ['teapot', 'coffeepot', 'Crock_Pot'], ['minivan', 'limousine'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla'], ['pug', 'boxer'], ['orange', 'Granny_Smith', 'lemon'], ['fig', 'cauliflower'], ['zucchini', 'orange', 'Granny_Smith', 'lemon'], ['Persian_cat', 'kuvasz'], ['broccoli', 'head_cabbage'], ['wok', 'caldron'], ['catamaran', 'trimaran', 'fireboat', 'airliner', 'container_ship', 'warplane', 'space_shuttle'], ['frying_pan', 'wok', 'caldron'], ['white_stork', 'flamingo'], ['broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron'], ['fig', 'cauliflower', 'zucchini', 'orange', 'Granny_Smith', 'lemon'], ['pug', 'boxer', 'Persian_cat', 'kuvasz'], ['jeep', 'minivan', 'limousine'], ['tabby', 'Egyptian_cat', 'pug', 'boxer', 'Persian_cat', 'kuvasz'], ['American_coot', 'black_swan', 'white_stork', 'flamingo'], ['police_van', 'jeep', 'minivan', 'limousine'], ['teapot', 'coffeepot', 'Crock_Pot', 'broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron'], ['fig', 'cauliflower', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'teapot', 'coffeepot', 'Crock_Pot', 'broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron'], ['Norwich_terrier', 'tabby', 'Egyptian_cat', 'pug', 'boxer', 'Persian_cat', 'kuvasz'], ['sports_car', 'police_van', 'jeep', 'minivan', 'limousine'], ['catamaran', 'trimaran', 'fireboat', 'airliner', 'container_ship', 'warplane', 'space_shuttle', 'sports_car', 'police_van', 'jeep', 'minivan', 'limousine'], ['Norwich_terrier', 'tabby', 'Egyptian_cat', 'pug', 'boxer', 'Persian_cat', 'kuvasz', 'catamaran', 'trimaran', 'fireboat', 'airliner', 'container_ship', 'warplane', 'space_shuttle', 'sports_car', 'police_van', 'jeep', 'minivan', 'limousine'], ['American_coot', 'black_swan', 'white_stork', 'flamingo', 'fig', 'cauliflower', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'teapot', 'coffeepot', 'Crock_Pot', 'broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron'], ['Norwich_terrier', 'tabby', 'Egyptian_cat', 'pug', 'boxer', 'Persian_cat', 'kuvasz', 'catamaran', 'trimaran', 'fireboat', 'airliner', 'container_ship', 'warplane', 'space_shuttle', 'sports_car', 'police_van', 'jeep', 'minivan', 'limousine', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'fig', 'cauliflower', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'teapot', 'coffeepot', 'Crock_Pot', 'broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'Norwich_terrier', 'tabby', 'Egyptian_cat', 'pug', 'boxer', 'Persian_cat', 'kuvasz', 'catamaran', 'trimaran', 'fireboat', 'airliner', 'container_ship', 'warplane', 'space_shuttle', 'sports_car', 'police_van', 'jeep', 'minivan', 'limousine', 'American_coot', 'black_swan', 'white_stork', 'flamingo', 'fig', 'cauliflower', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'teapot', 'coffeepot', 'Crock_Pot', 'broccoli', 'head_cabbage', 'frying_pan', 'wok', 'caldron']
        ]
}

NASNetLarge_hierarchical_data = {
    'directed graph': [
            ['Persian_cat', 'Madagascar_cat'], ['pug', 'Persian_cat', 'Madagascar_cat'], ['boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['broccoli', 'cauliflower'], ['chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'lemon'], ['airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['orange', 'lemon', 'broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['head_cabbage', 'orange', 'lemon', 'broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['Granny_Smith', 'head_cabbage', 'orange', 'lemon', 'broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['American_coot', 'Granny_Smith', 'head_cabbage', 'orange', 'lemon', 'broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat'], ['black_swan', 'American_coot', 'Granny_Smith', 'head_cabbage', 'orange', 'lemon', 'broccoli', 'cauliflower', 'warplane', 'airliner', 'trimaran', 'catamaran', 'gorilla', 'chimpanzee', 'coffeepot', 'teapot', 'frying_pan', 'wok', 'Egyptian_cat', 'tabby', 'zucchini', 'fig', 'spider_monkey', 'Crock_Pot', 'caldron', 'flamingo', 'white_stork', 'fireboat', 'container_ship', 'space_shuttle', 'jeep', 'limousine', 'sports_car', 'police_van', 'minivan', 'kuvasz', 'Norwich_terrier', 'boxer', 'pug', 'Persian_cat', 'Madagascar_cat']        ],
    'undirected graph': [
            ['Madagascar_cat', 'Norwich_terrier'], ['kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['tabby', 'Egyptian_cat'], ['wok', 'frying_pan'], ['orange', 'lemon'], ['teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['broccoli', 'cauliflower'], ['airliner', 'warplane'], ['chimpanzee', 'gorilla'], ['head_cabbage', 'broccoli', 'cauliflower'], ['caldron', 'wok', 'frying_pan'], ['zucchini', 'caldron', 'wok', 'frying_pan'], ['pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['Granny_Smith', 'orange', 'lemon'], ['police_van', 'jeep'], ['American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['container_ship', 'airliner', 'warplane'], ['black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['fig', 'Granny_Smith', 'orange', 'lemon'], ['minivan', 'police_van', 'jeep'], ['chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan'], ['white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['flamingo', 'white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['head_cabbage', 'broccoli', 'cauliflower', 'Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan'], ['Crock_Pot', 'head_cabbage', 'broccoli', 'cauliflower', 'Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan'], ['spider_monkey', 'flamingo', 'white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['minivan', 'police_van', 'jeep', 'spider_monkey', 'flamingo', 'white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['fig', 'Granny_Smith', 'orange', 'lemon', 'Crock_Pot', 'head_cabbage', 'broccoli', 'cauliflower', 'Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan'], ['limousine', 'minivan', 'police_van', 'jeep', 'spider_monkey', 'flamingo', 'white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane'], ['tabby', 'Egyptian_cat', 'fig', 'Granny_Smith', 'orange', 'lemon', 'Crock_Pot', 'head_cabbage', 'broccoli', 'cauliflower', 'Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan'], ['limousine', 'minivan', 'police_van', 'jeep', 'spider_monkey', 'flamingo', 'white_stork', 'chimpanzee', 'gorilla', 'space_shuttle', 'black_swan', 'American_coot', 'boxer', 'pug', 'coffeepot', 'teapot', 'trimaran', 'catamaran', 'fireboat', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'container_ship', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'fig', 'Granny_Smith', 'orange', 'lemon', 'Crock_Pot', 'head_cabbage', 'broccoli', 'cauliflower', 'Persian_cat', 'zucchini', 'caldron', 'wok', 'frying_pan']        ],
    'distances graph': [
            ['broccoli', 'head_cabbage'], ['American_coot', 'black_swan'], ['limousine', 'jeep'], ['pug', 'boxer'], ['Granny_Smith', 'lemon'], ['white_stork', 'flamingo'], ['minivan', 'police_van'], ['chimpanzee', 'spider_monkey'], ['warplane', 'space_shuttle'], ['limousine', 'jeep', 'minivan', 'police_van'], ['teapot', 'coffeepot'], ['airliner', 'container_ship'], ['catamaran', 'trimaran'], ['tabby', 'Egyptian_cat'], ['frying_pan', 'zucchini'], ['warplane', 'space_shuttle', 'airliner', 'container_ship'], ['gorilla', 'chimpanzee', 'spider_monkey'], ['wok', 'Crock_Pot'], ['fig', 'cauliflower'], ['Persian_cat', 'tabby', 'Egyptian_cat'], ['caldron', 'teapot', 'coffeepot'], ['orange', 'Granny_Smith', 'lemon'], ['frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon'], ['wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot'], ['fireboat', 'limousine', 'jeep', 'minivan', 'police_van'], ['fig', 'cauliflower', 'frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon'], ['kuvasz', 'pug', 'boxer'], ['broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot'], ['American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['white_stork', 'flamingo', 'fireboat', 'limousine', 'jeep', 'minivan', 'police_van'], ['broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot', 'American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['catamaran', 'trimaran', 'white_stork', 'flamingo', 'fireboat', 'limousine', 'jeep', 'minivan', 'police_van'], ['fig', 'cauliflower', 'frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot', 'American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['warplane', 'space_shuttle', 'airliner', 'container_ship', 'fig', 'cauliflower', 'frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot', 'American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat'], ['Madagascar_cat', 'Norwich_terrier'], ['sports_car', 'Madagascar_cat', 'Norwich_terrier'], ['kuvasz', 'pug', 'boxer', 'sports_car', 'Madagascar_cat', 'Norwich_terrier'], ['gorilla', 'chimpanzee', 'spider_monkey', 'kuvasz', 'pug', 'boxer', 'sports_car', 'Madagascar_cat', 'Norwich_terrier'], ['warplane', 'space_shuttle', 'airliner', 'container_ship', 'fig', 'cauliflower', 'frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot', 'American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'spider_monkey', 'kuvasz', 'pug', 'boxer', 'sports_car', 'Madagascar_cat', 'Norwich_terrier'], ['catamaran', 'trimaran', 'white_stork', 'flamingo', 'fireboat', 'limousine', 'jeep', 'minivan', 'police_van', 'warplane', 'space_shuttle', 'airliner', 'container_ship', 'fig', 'cauliflower', 'frying_pan', 'zucchini', 'orange', 'Granny_Smith', 'lemon', 'broccoli', 'head_cabbage', 'wok', 'Crock_Pot', 'caldron', 'teapot', 'coffeepot', 'American_coot', 'black_swan', 'Persian_cat', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'spider_monkey', 'kuvasz', 'pug', 'boxer', 'sports_car', 'Madagascar_cat', 'Norwich_terrier']        ]
}

MobileNetV2_hierarchical_data = {
    'directed graph': [
            ['Madagascar_cat', 'Norwich_terrier'], ['kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['tabby', 'Egyptian_cat'], ['catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['teapot', 'coffeepot'], ['chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['airliner', 'warplane'], ['tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['broccoli', 'cauliflower'], ['airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['head_cabbage', 'Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['teapot', 'coffeepot', 'head_cabbage', 'Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['minivan', 'limousine'], ['police_van', 'jeep'], ['minivan', 'limousine', 'police_van', 'jeep'], ['white_stork', 'flamingo'], ['teapot', 'coffeepot', 'head_cabbage', 'Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'minivan', 'limousine', 'police_van', 'jeep'], ['Crock_Pot', 'teapot', 'coffeepot', 'head_cabbage', 'Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'minivan', 'limousine', 'police_van', 'jeep'], ['black_swan', 'white_stork', 'flamingo'], ['American_coot', 'black_swan', 'white_stork', 'flamingo'], ['Crock_Pot', 'teapot', 'coffeepot', 'head_cabbage', 'Persian_cat', 'space_shuttle', 'broccoli', 'cauliflower', 'boxer', 'pug', 'airliner', 'warplane', 'tabby', 'Egyptian_cat', 'gorilla', 'chimpanzee', 'lemon', 'orange', 'frying_pan', 'wok', 'trimaran', 'catamaran', 'zucchini', 'fig', 'Granny_Smith', 'spider_monkey', 'caldron', 'fireboat', 'container_ship', 'sports_car', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'minivan', 'limousine', 'police_van', 'jeep', 'American_coot', 'black_swan', 'white_stork', 'flamingo']        ],
    'undirected graph': [
            ['Madagascar_cat', 'Norwich_terrier'], ['kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['wok', 'frying_pan'], ['teapot', 'coffeepot'], ['catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier'], ['tabby', 'Egyptian_cat'], ['orange', 'lemon'], ['chimpanzee', 'gorilla'], ['head_cabbage', 'cauliflower'], ['airliner', 'warplane'], ['broccoli', 'head_cabbage', 'cauliflower'], ['Persian_cat', 'tabby', 'Egyptian_cat'], ['pug', 'boxer'], ['minivan', 'police_van'], ['pug', 'boxer', 'minivan', 'police_van'], ['Crock_Pot', 'teapot', 'coffeepot'], ['space_shuttle', 'airliner', 'warplane'], ['Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower'], ['jeep', 'pug', 'boxer', 'minivan', 'police_van'], ['limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van'], ['white_stork', 'flamingo'], ['fig', 'orange', 'lemon'], ['trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon'], ['sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van'], ['Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['wok', 'frying_pan', 'Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon'], ['sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van', 'Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['container_ship', 'fireboat'], ['black_swan', 'white_stork', 'flamingo'], ['space_shuttle', 'airliner', 'warplane', 'container_ship', 'fireboat'], ['black_swan', 'white_stork', 'flamingo', 'space_shuttle', 'airliner', 'warplane', 'container_ship', 'fireboat'], ['Persian_cat', 'tabby', 'Egyptian_cat', 'wok', 'frying_pan', 'Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon'], ['spider_monkey', 'sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van', 'Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['American_coot', 'black_swan', 'white_stork', 'flamingo', 'space_shuttle', 'airliner', 'warplane', 'container_ship', 'fireboat'], ['zucchini', 'Persian_cat', 'tabby', 'Egyptian_cat', 'wok', 'frying_pan', 'Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon'], ['caldron', 'spider_monkey', 'sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van', 'Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['zucchini', 'Persian_cat', 'tabby', 'Egyptian_cat', 'wok', 'frying_pan', 'Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon', 'caldron', 'spider_monkey', 'sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van', 'Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla'], ['American_coot', 'black_swan', 'white_stork', 'flamingo', 'space_shuttle', 'airliner', 'warplane', 'container_ship', 'fireboat', 'zucchini', 'Persian_cat', 'tabby', 'Egyptian_cat', 'wok', 'frying_pan', 'Granny_Smith', 'broccoli', 'head_cabbage', 'cauliflower', 'fig', 'orange', 'lemon', 'caldron', 'spider_monkey', 'sports_car', 'limousine', 'jeep', 'pug', 'boxer', 'minivan', 'police_van', 'Crock_Pot', 'teapot', 'coffeepot', 'trimaran', 'catamaran', 'kuvasz', 'Madagascar_cat', 'Norwich_terrier', 'chimpanzee', 'gorilla']        ],
    'distances graph': [
            ['pug', 'boxer'], ['black_swan', 'flamingo'], ['minivan', 'jeep'], ['police_van', 'limousine'], ['coffeepot', 'Crock_Pot'], ['white_stork', 'black_swan', 'flamingo'], ['warplane', 'space_shuttle'], ['Madagascar_cat', 'spider_monkey'], ['minivan', 'jeep', 'police_van', 'limousine'], ['American_coot', 'white_stork', 'black_swan', 'flamingo'], ['catamaran', 'fireboat'], ['wok', 'caldron'], ['container_ship', 'catamaran', 'fireboat'], ['airliner', 'warplane', 'space_shuttle'], ['Persian_cat', 'tabby'], ['chimpanzee', 'gorilla'], ['trimaran', 'container_ship', 'catamaran', 'fireboat'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla'], ['teapot', 'coffeepot', 'Crock_Pot'], ['frying_pan', 'wok', 'caldron'], ['broccoli', 'cauliflower'], ['frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower'], ['fig', 'zucchini'], ['Egyptian_cat', 'Persian_cat', 'tabby'], ['head_cabbage', 'frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower'], ['airliner', 'warplane', 'space_shuttle', 'trimaran', 'container_ship', 'catamaran', 'fireboat'], ['Granny_Smith', 'lemon'], ['fig', 'zucchini', 'Granny_Smith', 'lemon'], ['sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['Norwich_terrier', 'kuvasz'], ['orange', 'fig', 'zucchini', 'Granny_Smith', 'lemon'], ['head_cabbage', 'frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower', 'orange', 'fig', 'zucchini', 'Granny_Smith', 'lemon'], ['teapot', 'coffeepot', 'Crock_Pot', 'head_cabbage', 'frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower', 'orange', 'fig', 'zucchini', 'Granny_Smith', 'lemon'], ['pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['Norwich_terrier', 'kuvasz', 'pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['Egyptian_cat', 'Persian_cat', 'tabby', 'Norwich_terrier', 'kuvasz', 'pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'Egyptian_cat', 'Persian_cat', 'tabby', 'Norwich_terrier', 'kuvasz', 'pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['teapot', 'coffeepot', 'Crock_Pot', 'head_cabbage', 'frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower', 'orange', 'fig', 'zucchini', 'Granny_Smith', 'lemon', 'Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'Egyptian_cat', 'Persian_cat', 'tabby', 'Norwich_terrier', 'kuvasz', 'pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine'], ['American_coot', 'white_stork', 'black_swan', 'flamingo', 'airliner', 'warplane', 'space_shuttle', 'trimaran', 'container_ship', 'catamaran', 'fireboat'], ['teapot', 'coffeepot', 'Crock_Pot', 'head_cabbage', 'frying_pan', 'wok', 'caldron', 'broccoli', 'cauliflower', 'orange', 'fig', 'zucchini', 'Granny_Smith', 'lemon', 'Madagascar_cat', 'spider_monkey', 'chimpanzee', 'gorilla', 'Egyptian_cat', 'Persian_cat', 'tabby', 'Norwich_terrier', 'kuvasz', 'pug', 'boxer', 'sports_car', 'minivan', 'jeep', 'police_van', 'limousine', 'American_coot', 'white_stork', 'black_swan', 'flamingo', 'airliner', 'warplane', 'space_shuttle', 'trimaran', 'container_ship', 'catamaran', 'fireboat']        ]
}

class HomePage(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./HomePage.ui", self)
        self.centralWidget().setStyleSheet("background-color: white;")
        self.set_image('./Images/logo.jpg', self.labelLogo, 600, 50)
        self.toolButton.clicked.connect(self.show_xai_interface)
        self.toolButton_2.clicked.connect(self.show_xai_query)

    def set_image(self, path, label_name, x, y):
        label = label_name  # Removed redundant QLabel creation
        label.setGeometry(x, y, 450, 450)
        # Load an image from the images folder
        pixmap = QPixmap(path)
        # Set the pixmap to the label and fit it
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def show_xai_interface(self):
        global widget
        widget.setCurrentIndex(1) 

    def show_xai_query(self):
        global widget
        widget.setCurrentIndex(2)

class XAIInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./XAIInterface.ui", self)
        self.centralWidget().setStyleSheet("background-color: white;")
        self.set_image('./Images/logo.jpg', self.labelLogo, 600, 50)
        self.next_button = self.findChild(QPushButton, 'nextButton')
        self.next_button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        self.back_button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        self.next_button.clicked.connect(self.go_to_naming_page)
        self.back_button.clicked.connect(self.back_function)

    def back_function(self):
        global widget
        widget.setCurrentIndex(0)

    def set_image(self, path, label_name, x, y):
        label = label_name  # Removed redundant QLabel creation
        label.setGeometry(x, y, 450, 450)
        # Load an image from the images folder
        pixmap = QPixmap(path)
        # Set the pixmap to the label and fit it
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def go_to_naming_page(self):
        blackBox = self.findChild(QComboBox, 'blackBoxBox').currentText()
        category = self.findChild(QComboBox, 'categoryBox').currentText()
        graph = self.findChild(QComboBox, 'graphBox').currentText()

        naming_page = Naming(blackBox, category, graph)
        widget.addWidget(naming_page)
        widget.setCurrentIndex(widget.currentIndex() + 2)

class Naming(QMainWindow):
    def __init__(self, blackBox, category, graph):
        super().__init__()
        uic.loadUi("./Naming.ui", self)
        self.back_button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        self.back_button.clicked.connect(self.back_function)
        self.blackBox = blackBox
        self.category = category
        self.graph = graph
        self.image_dict =   {
                                "Persian_cat": "./ImageNet/Persian_cat_image.JPEG",
                                "tabby": "./ImageNet/tabby_image.JPEG",
                                "Madagascar_cat": "./ImageNet/Madagascar_cat_image.JPEG",
                                "Egyptian_cat": "./ImageNet/Egyptian_cat_image.JPEG",
                                "pug": "./ImageNet/pug_image.jpg",
                                "boxer": "./ImageNet/boxer_image.JPEG",
                                "Norwich_terrier": "./ImageNet/Norwich_terrier_image.JPEG",
                                "kuvasz": "./ImageNet/kuvasz_image.JPEG",
                                "minivan": "./ImageNet/minivan_image.JPEG",
                                "police_van": "./ImageNet/police_van_image.JPEG",
                                "sports_car": "./ImageNet/sports_car_image.JPEG",
                                "limousine": "./ImageNet/limousine_image.JPEG",
                                "jeep": "./ImageNet/jeep_image.JPEG",
                                "airliner": "./ImageNet/airliner_image.JPEG",
                                "warplane": "./ImageNet/warplane_image.JPEG",
                                "space_shuttle": "./ImageNet/space_shuttle_image.JPEG",
                                "catamaran": "./ImageNet/catamaran_image.JPEG",
                                "trimaran": "./ImageNet/trimaran_image.JPEG",
                                "container_ship": "./ImageNet/container_ship_image.JPEG",
                                "fireboat": "./ImageNet/fireboat_image.JPEG",
                                "American_coot": "./ImageNet/American_coot_image.JPEG",
                                "black_swan": "./ImageNet/black_swan_image.JPEG",
                                "white_stork": "./ImageNet/white_stork_image.JPEG",
                                "flamingo": "./ImageNet/flamingo_image.JPEG",
                                "teapot": "./ImageNet/teapot_image.JPEG",
                                "coffeepot": "./ImageNet/coffeepot_image.JPEG",
                                "wok": "./ImageNet/wok_image.JPEG",
                                "frying_pan": "./ImageNet/frying_pan_image.JPEG",
                                "caldron": "./ImageNet/caldron_image.JPEG",
                                "Crock_Pot": "./ImageNet/Crock_Pot_image.JPEG",
                                "chimpanzee": "./ImageNet/chimpanzee_image.JPEG",
                                "gorilla": "./ImageNet/gorilla_image.JPEG",
                                "spider_monkey": "./ImageNet/spider_monkey_image.JPEG",
                                "Granny_Smith": "./ImageNet/Granny_Smith_image.JPEG",
                                "orange": "./ImageNet/orange_image.JPEG",
                                "lemon": "./ImageNet/lemon_image.JPEG",
                                "fig": "./ImageNet/fig_image.JPEG",
                                "zucchini": "./ImageNet/zucchini_image.jpg",
                                "broccoli": "./ImageNet/broccoli_image.jpg",
                                "head_cabbage": "./ImageNet/head_cabbage_image.jpg",
                                "cauliflower": "./ImageNet/cauliflower_image.jpg"
                            }

        hierarchical_data_map = {
            'ResNet': ResNet_hierarchical_data,
            'VGG': VGG_hierarchical_data,
            'GoogLeNet': GoogLeNet_hierarchical_data,
            'EfficientNet': EfficientNet_hierarchical_data,
            'NASNetLarge': NASNetLarge_hierarchical_data,
            'MobileNetV2': MobileNetV2_hierarchical_data
        }

        # Check if the provided blackBox key exists in the map
        if blackBox in hierarchical_data_map:
            data = hierarchical_data_map[blackBox]
            self.paths = [path for path in data[graph] if category in path]
        else:
            raise ValueError(f"Unknown blackBox named {blackBox}")

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Naming Connections")  # Set window title
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: white; font-size: 16px;")
        widget = QWidget()
        vbox = QVBoxLayout()

        # Button and Header display
        button = QPushButton(f"Go Back")
        button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        button.setMinimumHeight(50)
        button.setMinimumWidth(100)
        button.setMaximumHeight(50)
        button.setMaximumWidth(100)
        button.clicked.connect(self.back_function)
        vbox.addWidget(button)

        # Add an image for the hierarchical clustering dendrogram
        image_path = "./Images/logo.jpg"
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(250, 250) 
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setGeometry(30, 30, 300, 300)
        vbox.addWidget(image_label)

        header = QLabel(f"Category: <b style='color: red; font-size: 20px;'>{self.category}</b> | Graph Type: <b>{self.graph}</b> | Model: <b>{self.blackBox}</b>")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px;")
        vbox.addWidget(header)

        # Add an image for the hierarchical clustering dendrogram
        image_paths_map = {
                'ResNet': {
                    'directed graph': "./Images/ResNet_directed_image.png",
                    'undirected graph': "./Images/ResNet_undirected_image.png",
                    'distances graph': "./Images/ResNet_distances_image.png"
                },
                'VGG': {
                    'directed graph': "./Images/VGG_directed_image.png",
                    'undirected graph': "./Images/VGG_undirected_image.png",
                    'distances graph': "./Images/VGG_distances_image.png"
                },
                'GoogLeNet': {
                    'directed graph': "./Images/GoogLeNet_directed_image.png",
                    'undirected graph': "./Images/GoogLeNet_undirected_image.png",
                    'distances graph': "./Images/GoogLeNet_distances_image.png"
                },
                'EfficientNet': {
                    'directed graph': "./Images/EfficientNet_directed_image.png",
                    'undirected graph': "./Images/EfficientNet_undirected_image.png",
                    'distances graph': "./Images/EfficientNet_distances_image.png"
                },
                'NASNetLarge': {
                    'directed graph': "./Images/NASNetLarge_directed_image.png",
                    'undirected graph': "./Images/NASNetLarge_undirected_image.png",
                    'distances graph': "./Images/NASNetLarge_distances_image.png"
                },
                'MobileNetV2': {
                    'directed graph': "./Images/MobileNetV2_directed_image.png",
                    'undirected graph': "./Images/MobileNetV2_undirected_image.png",
                    'distances graph': "./Images/MobileNetV2_distances_image.png"
                }
        }

        if self.blackBox in image_paths_map:
            if self.graph in image_paths_map[self.blackBox]:
                image_path = image_paths_map[self.blackBox][self.graph]
            else:
                raise ValueError(f"Unknown graph named {self.graph}")
        else:
            raise ValueError(f"Unknown blackBox named {self.blackBox}")

        image_label = QLabel()
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1000, 600) 
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setGeometry(30, 30, 300, 200)
        vbox.addWidget(image_label)

        if not self.paths:
            no_path_label = QLabel("The selected image category does not have any hierarchical data.")
            no_path_label.setStyleSheet("font-size: 16px; color: red;")
            no_path_label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(no_path_label)
        else:
            for path in self.paths:
                # Add the connection label to the layout
                connection_label = QLabel(f"<br><b>Connection:</b> [")
                connection_label.setTextFormat(Qt.TextFormat.RichText)
                vbox.addWidget(connection_label)

                hbox = QHBoxLayout()
                count = 0
                for item in path:
                    if item in self.image_dict:
                        if count > 15:
                            # Add the connection label to the layout
                            connection_label = QLabel(" ")
                            connection_label.setTextFormat(Qt.TextFormat.RichText)
                            hbox.addWidget(connection_label)
                            vbox.addLayout(hbox)
                            hbox = QHBoxLayout()
                            count = 0

                        count += 1
                        image_label = ImageLabel(self.image_dict[item], item)
                        image_label.setTextFormat(Qt.TextFormat.RichText)
                        formatted_item = f"<span style='color: red; font-size: 18px;'><b>{item}</b></span>" if item == self.category else item
                        image_label.setText(formatted_item)

                        # Adjust label properties
                        image_label.setContentsMargins(5, 0, 5, 0)  # Adjust margins to minimize space
                        hbox.addWidget(image_label)

                    # Adjust layout spacing and alignment
                    hbox.setContentsMargins(0, 0, 0, 0)
                    hbox.setSpacing(2)
                    hbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

                # Add the horizontal layout with image labels to the vertical layout
                vbox.addLayout(hbox)

                hbox = QHBoxLayout()
                suggested_name = self.load_saved_name(path)
                if not suggested_name:
                    suggested_name = self.common_group(path)
                suggested_name_label = QLabel(f"]<br><b>Suggested Name:</b> {suggested_name}, <b>Please suggest new name:</b> ")
                hbox.addWidget(suggested_name_label)

                # User name input
                user_name_edit = QLineEdit()
                user_name_edit.setPlaceholderText("Enter new name")
                user_name_edit.setStyleSheet("font-size: 12px;")
                user_name_edit.setFixedWidth(200)
                user_name_edit.textChanged.connect(lambda text, p=path, label=suggested_name_label: self.update_name(p, text, label))
                hbox.addWidget(user_name_edit)

                # Send Button
                send_button = QPushButton("Send")
                send_button.setStyleSheet("background-color: LightGray; font-size: 12px;")
                send_button.setFixedWidth(200)
                send_button.clicked.connect(lambda checked, p=path, edit=user_name_edit: self.save_name(p, edit.text()))
                hbox.addWidget(send_button)

                vbox.addLayout(hbox)

        widget.setLayout(vbox)
        scroll.setWidget(widget)
        self.setCentralWidget(scroll)

    def update_name(self, path, new_name, label):
        if new_name:
            label.setText(f"<b>Suggested Name:</b> {new_name}, <b>Please suggest new name:</b> ")
        else:
            suggested_name = self.common_group(path)
            label.setText(f"<b>Suggested Name:</b> {suggested_name}, <b>Please suggest new name:</b> ")

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

    def back_function(self):
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex() - 1)

    def common_group(self, groups):
        common_hypernyms = []
        hierarchy = {}
        
        # Get the synsets for each input name
        for group in groups:
            
            # Initialize an empty list for each category folder key
            hierarchy[group] = []
            
            # Extract hypernyms for each category
            synsets = wn.synsets(group)
            if synsets:
                hypernyms = synsets[0].hypernym_paths()
                for path in hypernyms:
                    hierarchy[group].extend([node.name().split('.')[0] for node in path])
                    
        # Check common hypernyms
        if len(hierarchy) == 1:
            common_hypernyms = list(hierarchy.values())[0]
        else:
            for group in groups:
                for hypernym in hierarchy[group]:
                    if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                        common_hypernyms.append(hypernym)
        
        return next(iter(common_hypernyms[::-1]), "No common hypernym found")

    def save_name(self, path, new_name):
        if new_name:
            try:
                # Read the current content of the file
                with open("./user_common_group.txt", 'r') as file:
                    lines = file.readlines()
            except FileNotFoundError:
                lines = []

            # Remove the existing entry if the path exists and add the new entry
            new_lines = [line for line in lines if f"[{path}," not in line]
            new_lines.append(f"[{path}, {new_name}]\n")

            # Write the updated content back to the file
            with open("./user_common_group.txt", 'w') as file:
                file.writelines(new_lines)
            QMessageBox.information(self, "Saved", "Your name has been saved successfully!")
        else:
            QMessageBox.warning(self, "Error", "Please enter a name before saving.")

    def load_saved_name(self, path):
        try:
            with open("./user_common_group.txt", 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if f"[{path}," in line:
                        return line.split("], ")[1].strip().replace("[", "").replace("]", "")
        except FileNotFoundError:
            return None
        return None

class XAIQuery(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./XAIQuery.ui", self)
        self.centralWidget().setStyleSheet("background-color: white;")
        self.set_image('./Images/logo.jpg', self.labelLogo, 700, 80, 250, 250)
        self.next_button = self.findChild(QPushButton, 'nextButton')
        self.next_button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        self.back_button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        self.image_button = self.findChild(QToolButton, 'selectImageButton')
        self.next_button.clicked.connect(self.go_to_graph_res_page)
        self.back_button.clicked.connect(self.back_function)
        self.image_button.clicked.connect(self.upload_image)
        self.image_path = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            self.set_image(file_name, self.labelSelectedImage, 240, 340, 300, 200)

    def go_to_graph_res_page(self):
        blackBox = self.findChild(QComboBox, 'blackBoxBox').currentText()
        graph = self.findChild(QComboBox, 'graphBox').currentText()
        if self.image_path:
            category = self.find_category_by_BlackBox(blackBox, self.image_path)
            if category is None:
                QMessageBox.warning(self, "Error", "The selected image category does not exist in the hierarchical data.")
            else:
                graph_res_page = TheGraphRes(category, graph, blackBox) # display_text
                widget.addWidget(graph_res_page)
                widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            QMessageBox.warning(self, "Error", "Please select an image first.")

    def find_category_by_BlackBox(self, blackBox, image_path):
        if blackBox == 'ResNet':
            try:
                # Get the top prediction for the image by using ResNet
                model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.resnet50.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in ResNet_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None
            
            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

        if blackBox == 'VGG':
            try:
                # Get the top prediction for the image by using VGG16
                model = tf.keras.applications.VGG16(weights='imagenet')
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Make predictions
                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.vgg16.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in VGG_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

        if blackBox == 'GoogLeNet':
            try:
                # Get the top prediction for the image by using InceptionV3
                model = tf.keras.applications.InceptionV3(weights='imagenet')
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Predict using the InceptionV3 model
                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in GoogLeNet_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

        if blackBox == 'EfficientNet':
            try:
                # Load the EfficientNetB0 model pre-trained on ImageNet
                model = tf.keras.applications.EfficientNetB0(weights='imagenet')
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Predict using the EfficientNetB0 model
                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.efficientnet.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in EfficientNet_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

        if blackBox == 'NASNetLarge':
            try:
                # Load the EfficientNetB0 model pre-trained on ImageNet
                model = tf.keras.applications.NASNetLarge(weights='imagenet')
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(331, 331))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.nasnet.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Predict using the EfficientNetB0 model
                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.nasnet.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in EfficientNet_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

        if blackBox == 'MobileNetV2':
            try:
                # Load the MobileNetV2 model pre-trained on ImageNet
                model = tf.keras.applications.MobileNetV2(weights='imagenet')
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Predict using the EfficientNetB0 model
                prediction = model.predict(img_array)
                decoded_prediction = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)
                predicted_category = decoded_prediction[0][0][1]

                # Check if the predicted category exists in the hierarchical data
                if any(predicted_category in path for paths in EfficientNet_hierarchical_data.values() for path in paths):
                    return predicted_category
                else:
                    return None

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return None

    def set_image(self, path, label_name, x, y, w, h):
        label = label_name  # Removed redundant QLabel creation
        label.setGeometry(x, y, w, h)
        # Load an image from the images folder
        pixmap = QPixmap(path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def back_function(self):
        global widget
        widget.setCurrentIndex(0)

class TheGraphRes(QMainWindow):
    def __init__(self, category, graph, blackBox):
        super().__init__()
        uic.loadUi("./TheGraphRes.ui", self)
        self.blackBox = blackBox
        self.graph = graph
        self.category = category
        self.image_dict =   {
                                "Persian_cat": "./ImageNet/Persian_cat_image.JPEG",
                                "tabby": "./ImageNet/tabby_image.JPEG",
                                "Madagascar_cat": "./ImageNet/Madagascar_cat_image.JPEG",
                                "Egyptian_cat": "./ImageNet/Egyptian_cat_image.JPEG",
                                "pug": "./ImageNet/pug_image.jpg",
                                "boxer": "./ImageNet/boxer_image.JPEG",
                                "Norwich_terrier": "./ImageNet/Norwich_terrier_image.JPEG",
                                "kuvasz": "./ImageNet/kuvasz_image.JPEG",
                                "minivan": "./ImageNet/minivan_image.JPEG",
                                "police_van": "./ImageNet/police_van_image.JPEG",
                                "sports_car": "./ImageNet/sports_car_image.JPEG",
                                "limousine": "./ImageNet/limousine_image.JPEG",
                                "jeep": "./ImageNet/jeep_image.JPEG",
                                "airliner": "./ImageNet/airliner_image.JPEG",
                                "warplane": "./ImageNet/warplane_image.JPEG",
                                "space_shuttle": "./ImageNet/space_shuttle_image.JPEG",
                                "catamaran": "./ImageNet/catamaran_image.JPEG",
                                "trimaran": "./ImageNet/trimaran_image.JPEG",
                                "container_ship": "./ImageNet/container_ship_image.JPEG",
                                "fireboat": "./ImageNet/fireboat_image.JPEG",
                                "American_coot": "./ImageNet/American_coot_image.JPEG",
                                "black_swan": "./ImageNet/black_swan_image.JPEG",
                                "white_stork": "./ImageNet/white_stork_image.JPEG",
                                "flamingo": "./ImageNet/flamingo_image.JPEG",
                                "teapot": "./ImageNet/teapot_image.JPEG",
                                "coffeepot": "./ImageNet/coffeepot_image.JPEG",
                                "wok": "./ImageNet/wok_image.JPEG",
                                "frying_pan": "./ImageNet/frying_pan_image.JPEG",
                                "caldron": "./ImageNet/caldron_image.JPEG",
                                "Crock_Pot": "./ImageNet/Crock_Pot_image.JPEG",
                                "chimpanzee": "./ImageNet/chimpanzee_image.JPEG",
                                "gorilla": "./ImageNet/gorilla_image.JPEG",
                                "spider_monkey": "./ImageNet/spider_monkey_image.JPEG",
                                "Granny_Smith": "./ImageNet/Granny_Smith_image.JPEG",
                                "orange": "./ImageNet/orange_image.JPEG",
                                "lemon": "./ImageNet/lemon_image.JPEG",
                                "fig": "./ImageNet/fig_image.JPEG",
                                "zucchini": "./ImageNet/zucchini_image.jpg",
                                "broccoli": "./ImageNet/broccoli_image.jpg",
                                "head_cabbage": "./ImageNet/head_cabbage_image.jpg",
                                "cauliflower": "./ImageNet/cauliflower_image.jpg"
                            }
        self.setupUI()

    def setupUI(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: white; font-size: 16px;")
        widget = QWidget()
        vbox = QVBoxLayout()

        # Button and Header display
        button = QPushButton(f"Go Back")
        button.setStyleSheet("background-color: LightGray; font-size: 20px;")
        button.setMinimumHeight(50)
        button.setMinimumWidth(100)
        button.setMaximumHeight(50)
        button.setMaximumWidth(100)
        button.clicked.connect(self.back_function)
        vbox.addWidget(button)

        # Add image for the hierarchical clustering dendrogram
        image_path = "./Images/logo.jpg"
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(250, 250) 
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setGeometry(30, 30, 300, 300)
        vbox.addWidget(image_label)

        header = QLabel(f"Category: <b style='color: red; font-size: 20px;'>{self.category}</b> | Graph Type: <b>{self.graph}</b> | Model: <b>{self.blackBox}</b>")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px;")
        vbox.addWidget(header)
        
        # Add an image for the hierarchical clustering dendrogram
        image_paths_map = {
                    'ResNet': {
                        'directed graph': "./Images/ResNet_directed_image.png",
                        'undirected graph': "./Images/ResNet_undirected_image.png",
                        'distances graph': "./Images/ResNet_distances_image.png"
                    },
                    'VGG': {
                        'directed graph': "./Images/VGG_directed_image.png",
                        'undirected graph': "./Images/VGG_undirected_image.png",
                        'distances graph': "./Images/VGG_distances_image.png"
                    },
                    'GoogLeNet': {
                        'directed graph': "./Images/GoogLeNet_directed_image.png",
                        'undirected graph': "./Images/GoogLeNet_undirected_image.png",
                        'distances graph': "./Images/GoogLeNet_distances_image.png"
                    },
                    'EfficientNet': {
                        'directed graph': "./Images/EfficientNet_directed_image.png",
                        'undirected graph': "./Images/EfficientNet_undirected_image.png",
                        'distances graph': "./Images/EfficientNet_distances_image.png"
                    },
                    'NASNetLarge': {
                        'directed graph': "./Images/NASNetLarge_directed_image.png",
                        'undirected graph': "./Images/NASNetLarge_undirected_image.png",
                        'distances graph': "./Images/NASNetLarge_distances_image.png"
                    },
                    'MobileNetV2': {
                        'directed graph': "./Images/MobileNetV2_directed_image.png",
                        'undirected graph': "./Images/MobileNetV2_undirected_image.png",
                        'distances graph': "./Images/MobileNetV2_distances_image.png"
                    }
        }

        if self.blackBox in image_paths_map:
            if self.graph in image_paths_map[self.blackBox]:
                image_path = image_paths_map[self.blackBox][self.graph]
            else:
                raise ValueError(f"Unknown graph named {self.graph}")
        else:
            raise ValueError(f"Unknown blackBox named {self.blackBox}")
        
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():  # Check if the image was loaded correctly
            scaled_pixmap = pixmap.scaled(1000, 600, Qt.KeepAspectRatio)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setGeometry(30, 30, 300, 200)
            vbox.addWidget(image_label)

        header = QLabel(f"<span style='font-size: 20px;'><b>Hierarchical Explanation:</b><br><br></span>")
        vbox.addWidget(header)

        # Load the user_common_group.txt file
        with open('./user_common_group.txt', 'r') as file:
            file_content = file.readlines()

        # Select paths using the hierarchical data map
        hierarchical_data_map = {
            'ResNet': ResNet_hierarchical_data,
            'VGG': VGG_hierarchical_data,
            'GoogLeNet': GoogLeNet_hierarchical_data,
            'EfficientNet': EfficientNet_hierarchical_data,
            'NASNetLarge': NASNetLarge_hierarchical_data,
            'MobileNetV2': MobileNetV2_hierarchical_data
        }

        if self.blackBox in hierarchical_data_map:
            data = hierarchical_data_map[self.blackBox]
            paths = data.get(self.graph, [])
        else:
            raise ValueError(f"Unknown blackBox named {self.blackBox}")

        if paths:
            formatted_paths = []
            for path in paths:
                if self.category in path:
                    flag = 0
                    hbox = QHBoxLayout()
                    count = 0
                    for line in file_content:
                        try:
                            # Strip whitespace and split the line into the list and group name
                            line = line.strip()
                            if line.startswith("[") and "], " in line:
                                list_part, group_name_part = line.split("], ", 1)
                                list_part += "]]"  # Add the closing bracket back to the list part
                                user_path = ast.literal_eval(list_part)
                                if np.array_equal(path, user_path[0]):
                                    flag = 1
                                    group_name = group_name_part.strip().replace(']', '')
                                    # Add the connection label to the layout
                                    connection_label = QLabel(f"A {self.category} is a part of the concept <b>'{group_name}'<b>: [")
                                    connection_label.setTextFormat(Qt.TextFormat.RichText)
                                    vbox.addWidget(connection_label)
                    
                                    for item in path:
                                        if item in self.image_dict:
                                            if count > 15:
                                                # Add the connection label to the layout
                                                vbox.addLayout(hbox)
                                                hbox = QHBoxLayout()
                                                count = 0

                                            count += 1
                                            image_label = ImageLabel(self.image_dict[item], item)
                                            image_label.setTextFormat(Qt.TextFormat.RichText)
                                            formatted_item = f"<span style='color: red; font-size: 22px;'><b>{item}</b></span>" if item == self.category else item
                                            image_label.setText(formatted_item)

                                            # Adjust label properties
                                            image_label.setContentsMargins(5, 0, 5, 0)  # Adjust margins to minimize space
                                            hbox.addWidget(image_label)

                                            # Adjust layout spacing and alignment
                                            hbox.setContentsMargins(0, 0, 0, 0)
                                            hbox.setSpacing(2)
                                            hbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

                                    vbox.addLayout(hbox)
                                    end = QLabel(f"]<br>")
                                    vbox.addWidget(end)
                                    break  # Stop searching after finding a match
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing line: {line}. Error: {e}")
                            continue  # Ignore lines that can't be parsed

                    if flag == 0:
                        # Add the connection label to the layout
                        connection_label = QLabel(f"A {self.category} is a part of the concept <b>'{self.common_group(path)}'<b>: [")
                        connection_label.setTextFormat(Qt.TextFormat.RichText)
                        vbox.addWidget(connection_label)
                        for item in path:
                            if item in self.image_dict:
                                if count > 15:
                                    # Add the connection label to the layout
                                    vbox.addLayout(hbox)
                                    hbox = QHBoxLayout()
                                    count = 0

                                count += 1
                                image_label = ImageLabel(self.image_dict[item], item)
                                image_label.setTextFormat(Qt.TextFormat.RichText)
                                formatted_item = f"<span style='color: red; font-size: 22px;'><b>{item}</b></span>" if item == self.category else item
                                image_label.setText(formatted_item)

                                # Adjust label properties
                                image_label.setContentsMargins(5, 0, 5, 0)  # Adjust margins to minimize space
                                hbox.addWidget(image_label)

                                # Adjust layout spacing and alignment
                                hbox.setContentsMargins(0, 0, 0, 0)
                                hbox.setSpacing(2)
                                hbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

                        vbox.addLayout(hbox)
                        end = QLabel(f"]<br>")
                        vbox.addWidget(end)

        widget.setLayout(vbox)
        scroll.setWidget(widget)
        self.setCentralWidget(scroll)

    def common_group(self, groups):
        common_hypernyms = []
        hierarchy = {}
        
        for group in groups:
            
            # Initialize an empty list for each category folder key
            hierarchy[group] = []
            
            # Extract hypernyms for each category
            synsets = wn.synsets(group)
            if synsets:
                hypernyms = synsets[0].hypernym_paths()
                for path in hypernyms:
                    hierarchy[group].extend([node.name().split('.')[0] for node in path])
                    
        # Check common hypernyms
        if len(hierarchy) == 1:
            common_hypernyms = list(hierarchy.values())[0]
        else:
            for group in groups:
                for hypernym in hierarchy[group]:
                    if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                        common_hypernyms.append(hypernym)
        
        return next(iter(common_hypernyms[::-1]), "No common hypernym found")

    def back_function(self):
        widget.removeWidget(widget.currentWidget())

class ImageTooltip(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent, Qt.ToolTip)
        self.setWindowFlags(Qt.ToolTip)
        self.label = QLabel(self)
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()
        self.adjustSize()

class ImageLabel(QLabel):
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.tooltip = None

    def enterEvent(self, event):
        # Show image tooltip when the mouse enters the label
        if not self.tooltip:
            self.tooltip = ImageTooltip(self.image_path, self)
        pos = self.mapToGlobal(QPoint(0, -self.tooltip.height()))
        self.tooltip.move(pos)
        self.tooltip.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Hide the tooltip when the mouse leaves the label
        if self.tooltip:
            self.tooltip.hide()
        super().leaveEvent(event)

app = QApplication(sys.argv)
widget = QStackedWidget()

home_page = HomePage()
xai_interface_page = XAIInterface()
xai_query_page = XAIQuery()

widget.addWidget(home_page)
widget.addWidget(xai_interface_page)
widget.addWidget(xai_query_page) 

widget.show()
sys.exit(app.exec_())
