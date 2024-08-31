"""
This approach involves constructing an undirected graph wherein the weights are assigned based on probabilities. 
A hierarchy of graphs is established in accordance with percentages. 
The UNION-FIND algorithm is employed to identify connections within the graph structure. 
"""

!pip install tensorflow
!pip install nltk
import os
import sys
import nltk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import logging
import random
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from datetime import datetime, timezone
from collections import Counter
from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
nltk.download('omw-1.4')
nltk.download('wordnet')

# choose the number of the top predictions and the model that will be used (ResNet, VGG, GoogLeNet, EfficientNet, NASNetLarge or MobileNetV2)
MODEL = 'ResNet'
TOP = 3
threshold = 1e-6

# Create a main folder for logging files
log_folder = "logs"

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Get the current time with timezone
current_time = datetime.now(timezone.utc)

# Format the time to include timezone information in the file name
file_name = os.path.join(log_folder, current_time.strftime("%Y-%m-%d_%H-%M-%S_%Z") + ".DEBUG")

# Configure logging
logging.basicConfig(filename=file_name, level=logging.DEBUG)

# Path to a single folder
main_folder = "Images"
if not os.path.isdir(main_folder):
    logging.error("Invalid folder path.")
    exit()
    
# Read the undirected graphs from the file based of the selected model
if MODEL == 'ResNet':
    G_undirected_path = "graphs\\ResNet_G_undirected.graphml"
elif MODEL == 'VGG':
    G_undirected_path = "graphs\\VGG_G_undirected.graphml"
elif MODEL == 'GoogLeNet':
    G_undirected_path = "graphs\\GoogLeNet_G_undirected.graphml"
elif MODEL == 'EfficientNet':
    G_undirected_path = "graphs\\EfficientNet_G_undirected.graphml"
elif MODEL == 'NASNetLarge':
    G_undirected_path = "graphs\\NASNetLarge_G_undirected.graphml"
elif MODEL == 'MobileNetV2':
    G_undirected_path = "graphs\\MobileNetV2_G_undirected.graphml"

def get_synsets(folder_name):
    synsets = wn.synset_from_pos_and_offset('n', int(folder_name[1:]))
    return synsets.lemma_names()[0]


def folder_name_to_number(folder_name):
    synsets = wn.synsets(folder_name)
    
    # Check if any synsets are found
    if synsets:
        offset = synsets[0].offset()        
        folder_number = 'n{:08d}'.format(offset)
        return folder_number

def get_top_predictions_by_ResNet(image_path, top=3):
    try:
        # Get the top prediction for the image by using ResNet_model
        ResNet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = ResNet_model.predict(img_array)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def get_top_predictions_by_VGG(image_path, top=3):
    try:
        # Get the top prediction for the image by using VGG_model
        VGG_model = tf.keras.applications.VGG16(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = VGG_model.predict(img_array)
        decoded_predictions = tf.keras.applications.vgg16.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def get_top_predictions_by_GoogLeNet(image_path, top=3):
    try:
        # Get the top prediction for the image by using GoogLeNet_model
        GoogLeNet_model = tf.keras.applications.InceptionV3(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = GoogLeNet_model.predict(img_array)
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def get_top_predictions_by_EfficientNet(image_path, top=3):
    try:
        # Get the top prediction for the image by using EfficientNet_model
        EfficientNet_model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = EfficientNet_model.predict(img_array)
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def get_top_predictions_by_NASNetLarge(image_path, top=3):
    try:
        # Get the top prediction for the image by using NASNetLarge_model
        NASNetLarge_model = tf.keras.applications.NASNetLarge(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(331, 331))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.nasnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = NASNetLarge_model.predict(img_array)
        decoded_predictions = tf.keras.applications.nasnet.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def get_top_predictions_by_MobileNetV2(image_path, top=3):
    try:
        # Get the top prediction for the image by using MobileNetV2
        MobileNetV2_model = tf.keras.applications.MobileNetV2(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = MobileNetV2_model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top)
        
        # Get the top predictions for the image
        return decoded_predictions

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []
    
def common_group(groups):
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
        for hypernym in hierarchy[groups.pop()]:
            if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                common_hypernyms.append(hypernym)
    
    return common_hypernyms[::-1]

# Create a graph
G = nx.Graph()

# Get a list of folder names in the specified directory
folder_names = os.listdir(main_folder)

# Format the time to include timezone information in the file name
file_name = os.path.join(log_folder, current_time.strftime("%Y-%m-%d_%H-%M-%S_%Z") + ".RESULT")

# Iterate over each folder name
categories = []
for folder_name in folder_names:
    if folder_name[:1] == 'n':
        synsets = wn.synset_from_pos_and_offset('n', int(folder_name[1:]))
        categories.append(synsets.lemma_names()[0])
    
# Add nodes for each category
for category in categories:
    G.add_node(category)

# Redirect output to debug log file
sys.stdout = open(f'{file_name}.log', 'w')

if os.path.isdir(G_undirected_path):
    G = nx.read_graphml(G_undirected_path)
else:
    # Dictionary to store all the predictions for each folder
    predictions = {}

    # Iterate through each category folder
    for category_folder in os.listdir(main_folder):
        category_folder_path = os.path.join(main_folder, category_folder)

        if os.path.isdir(category_folder_path):

            # Get all image files in the category folder
            image_files = [os.path.join(category_folder_path, file) for file in os.listdir(category_folder_path)]

            # Initialize an empty list for each category folder key
            predictions[category_folder] = []

            # Call the function to get top predictions
            for image_path in image_files:
                if MODEL == 'ResNet':
                    predictions[category_folder].append(get_top_predictions_by_ResNet(image_path, TOP))
                elif MODEL == 'VGG':
                    predictions[category_folder].append(get_top_predictions_by_VGG(image_path, TOP))
                elif MODEL == 'GoogLeNet':
                    predictions[category_folder].append(get_top_predictions_by_GoogLeNet(image_path, TOP))
                elif MODEL == 'EfficientNet':
                    predictions[category_folder].append(get_top_predictions_by_EfficientNet(image_path, TOP))
                elif MODEL == 'NASNetLarge':
                    predictions[category_folder].append(get_top_predictions_by_NASNetLarge(image_path, TOP))
                elif MODEL == 'MobileNetV2':
                    predictions[category_folder].append(get_top_predictions_by_MobileNetV2(image_path, TOP))
                
            print(predictions[category_folder])

            # Keep all the categories that are already as being shown
            shows_categories = []

            # Connect nodes based on hierarchy and calculate weights
            for predictions_set in predictions[category_folder]:
                try:
                    if predictions_set[0][0][0] == category_folder:
                        for prediction in predictions_set[0]:
                            if prediction[0] != category_folder and prediction[1] in categories and prediction[2] > threshold:
                                # Check if the edge already exists in graph G
                                if G.has_edge(get_synsets(category_folder), prediction[1]):
                                    G[get_synsets(category_folder)][prediction[1]]["weight"] += prediction[2]
                                else:
                                    G.add_edge(get_synsets(category_folder), prediction[1], weight = prediction[2])
                                    
                except Exception as e:
                    logging.error(f"Error processing prediction: {e}")
    
    # Save the graph to a file in GraphML format
    if MODEL == 'ResNet':
        nx.write_graphml(G, 'ResNet_G_undirected.graphml')
    elif MODEL == 'VGG':
        nx.write_graphml(G, 'VGG_G_undirected.graphml')
    elif MODEL == 'GoogLeNet':
        nx.write_graphml(G, 'GoogLeNet_G_undirected.graphml')
    elif MODEL == 'EfficientNet':
        nx.write_graphml(G, 'EfficientNet_G_undirected.graphml')
    elif MODEL == 'NASNetLarge':
        nx.write_graphml(G, 'NASNetLarge_G_undirected.graphml')
    elif MODEL == 'MobileNetV2':
        nx.write_graphml(G, 'MobileNetV2_G_undirected.graphml')

selected_categories = ["Persian_cat", "tabby", "Madagascar_cat", "Egyptian_cat", 
                      "pug", "boxer", "Norwich_terrier", "kuvasz",
                      "minivan", "police_van", "sports_car", "limousine", "jeep",
                      "airliner", "warplane", "space_shuttle",
                      "catamaran", "trimaran", "container_ship", "fireboat",
                      "American_coot", "black_swan", "white_stork", "flamingo",
                      "teapot", "coffeepot", 
                      "wok", "frying_pan", "caldron", "Crock_Pot",
                      "chimpanzee", "gorilla", "spider_monkey",
                      "Granny_Smith", "orange", "lemon", "fig",
                      "zucchini", "broccoli", "head_cabbage", "cauliflower"]

# Find categories in each cluster
def get_cluster_composition(Z, n, selected_categories):
    # Initialize each observation as its own cluster
    clusters = {i: [i] for i in range(n)}
    
    # Go through each merge step
    merge_composition = []
    for i in range(Z.shape[0]):
        cluster1 = int(Z[i, 0])
        cluster2 = int(Z[i, 1])
        
        # Merge the clusters
        new_cluster = clusters[cluster1] + clusters[cluster2]
        
        # Record the cluster category
        merge_composition.append([selected_categories[idx] for idx in new_cluster])
        clusters[n + i] = new_cluster
    
    return merge_composition

# Initialize the hierarchical graph
hierarchical_graph = nx.Graph()

# Extract the weights of edges
weights = [d['weight'] for u, v, d in G.edges(data=True) if u in selected_categories and v in selected_categories]

# Divide the weights into groups
sorted_weights = sorted(weights, reverse=True)
group_size = len(weights) // 20
remainder = len(weights) % 20
groups = []
start_index = 0

for i in range(20):
    group_length = group_size + (1 if i < remainder else 0)
    groups.append(sorted_weights[start_index:start_index + group_length])
    start_index += group_length

# Create a components percentages matrix
components_percentages_matrix = np.zeros((len(selected_categories), len(selected_categories)))

percentages = 5 #100/n

# Dictionary to map categories to their indices in the matrix
category_to_index = {category: idx for idx, category in enumerate(selected_categories)}

for group in groups:
    if percentages <= 100:
        # Add edges in the group
        for u, v, d in G.edges(data=True):
            if u in selected_categories and v in selected_categories:
                if d['weight'] in group and not hierarchical_graph.has_edge(u, v):
                    hierarchical_graph.add_edge(u, v)

        # Get the connected components
        components = list(nx.connected_components(hierarchical_graph))
        print(components)

        for component in components:
            for category in component:
                if category in selected_categories:
                    for other_category in component:
                        if other_category in selected_categories:
                            i, j = category_to_index[category], category_to_index[other_category]
                            if components_percentages_matrix[i][j] == 0 and i != j:
                                components_percentages_matrix[i][j] = percentages
                                components_percentages_matrix[j][i] = percentages

        # Increase the percentage for the next iteration
        percentages += 5

# Perform hierarchical clustering
Z = linkage(components_percentages_matrix, method='single', metric='euclidean')

# Number of observations
n = components_percentages_matrix.shape[0]

# Get the composition of each category at each step
# merge_composition = get_cluster_composition(Z, n, selected_categories)
# for i, comp in enumerate(merge_composition):
#     print(f"Step {i+1}: {comp}, group named: {common_group(comp)}")
# print(merge_composition)

# Plot the dendrogram with node names as labels
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=selected_categories, orientation='right')
plt.xlabel('percentages')
plt.ylabel('Nodes')
plt.xticks(np.arange(0, 105, 5))
plt.xlim(0, 101)
plt.show()
