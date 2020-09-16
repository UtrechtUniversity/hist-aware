import os
import gzip
import sys
import re
import shutil
import pathlib
from datetime import datetime
import xml.etree.ElementTree as et 

import numpy as np
import pandas as pd
import logging
import collections
import xmltodict
from itertools import chain

from IPython.display import display, clear_output, Markdown

def iterate_directory(root_path, dir_path, file_type):
    """Iterate over the `path_dir` and its children and
    create a dictionary of
        - name
        - path
        - dir
    names of files found
    """
    main_path=root_path+dir_path
    file_names = {}
    list_names = []

    for subdir, dirs, files in os.walk(main_path, topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(str(file_type)):
                file_names["article_name"] = file
                file_names["article_path"] = filepath
                file_names["article_dir"] = subdir
                list_names.append(file_names)
                file_names = {}

    return(list_names)

def iterate_directory_gz(root_path, dir_path, file_type):
    """Iterate over the `path_dir` and its children and
    create a dictionary of
        - name
        - path
        - dir
        - content
    of .gz files found.
    """
    main_path=root_path+dir_path
    gz_content = {}
    list_gzs = []
    
    for subdir, dirs, files in os.walk(main_path, topdown=True):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(str(file_type)):
                # Create list of dict
                with gzip.open(filepath, 'rb') as f:
                #, \
                #open(filepath+".xml", "wb") as r:
                    gz_content["metadata_name"] = file+".xml"
                    gz_content["metadata_dir"] = subdir
                    gz_content["metadata_path"] = filepath+".xml"
                    # Ungzipping and writing to .xml
                    #shutil.copyfileobj(f, r, 65536)
                    
                    list_gzs.append(gz_content)
                    gz_content = {}
    
    return(list_gzs)

def parse_XML_article(path, art_dir, title, index):
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    Takes the filepath, file title and index integer of the df
    """
    
    xtree = et.parse(path)
    xroot = xtree.getroot()
    list_articles = []
    
    # Parse the date with regex
    match = re.search(r'\d{4}[/]\d{2}[-]\d{2}', path)
    date = datetime.strptime(match.group(), '%Y/%m-%d').date()
    
    for i, node in enumerate(xroot):
        if node.tag == "title":
            article = {}
            article["type"] = "title"
            article["text"] = node.text
            article["article_name"] = str(title)
            article["date"] = str(date)
            article["index"] = index
            article["filepath"] = path
            article["dir"] = art_dir
            list_articles.append(article)
        else:
            article = {}
            article["type"] = "p"
            article["text"] = node.text
            article["article_name"] = str(title)
            article["date"] = str(date)
            article["index"] = index
            article["filepath"] = path
            article["dir"] = art_dir
            list_articles.append(article)

    # Returns list of dict of articles and titles
    return list_articles

def parse_XML_metadata(path, met_dir, title, index):
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    Takes the filepath, file title and index integer of the df
    """
    metadata = {}
    dict_metadata = {}
    
    # Parse the date with regex
    match = re.search(r'\d{4}[/]\d{2}[-]\d{2}', path)
    date = datetime.strptime(match.group(), '%Y/%m-%d').date()
    
    # Parse DIDL XML
    with open(pathlib.Path(path), 'r') as f:
        doc = xmltodict.parse(f.read())
    temp_data = doc["didl:DIDL"]["didl:Item"]["didl:Component"][0]["didl:Resource"]["srw_dc:dcx"]

    metadata["metadata_title"] = title
    metadata["date"] = date
    metadata["index"] = index
    metadata["filepath"] = path
    metadata["dir"] = met_dir
    
    # Retrieve informations about the newspaper
    metadata["newspaper_title"] = temp_data["dc:title"]
    metadata["newspaper_date"] = temp_data["dc:date"]
    metadata["newspaper_city"] = temp_data["dcterms:spatial"][1]["#text"]
    metadata["newspaper_publisher"] = temp_data["dc:publisher"]
    metadata["newspaper_source"] = temp_data["dc:source"]
    metadata["newspaper_volume"] = temp_data["dcx:volume"]
    metadata["newspaper_issuenumber"] = temp_data["dcx:issuenumber"]
    metadata["newspaper_language"] = temp_data["dc:language"]["#text"]
    
    dict_metadata[index] = metadata

    return(dict_metadata)

def iterate_files(save_path, files, restart=False, index_restart=0):
    """Iterate through files `files`, parse them and concatenate
    the result to be saved as a DataFrame in a feather object (.ftr)

    To restart the iteration, set `restart=True` and select the index from which the 
    iteration should restart with `index_restart`.
    """
    list_articles = []
    # Every 10000 articles create a ftr file
    save_each = 10000
    
    if restart == False:
        main = None
        previous_i = 0
        current_i = 0
        i = 0
        n = 0
        cnt = 0
        for index, row in files.iterrows():
            try:
                list_articles.append(parse_XML_article(
                    path = row["article_path"],
                    art_dir = row["article_dir"], 
                    title = row["article_name"],
                    index = row["index"]))
            except Exception as e:
                print(f"Index: {index}", e.args)
                continue
            # Each X, save the file in a .ftr
            if (i == save_each):
                current_i = current_i + i
                file_path = save_path+"processed_articles/processed_data_list_"+str(previous_i)+"_"+str(current_i)+".ftr"
                main = pd.DataFrame(list(chain.from_iterable(list_articles)))
                main.to_feather(file_path)
                main = None
                list_articles = []
                previous_i = current_i
                i = 0
            # Each 1000 files, print the progress
            if (i % 2000 == 0):
                clear_output(wait=True)
                display("Files parsed: "+str(2000*cnt))
                display("Current file: "+row["article_name"]+" (Index: "+str(row["index"])+")")
                cnt += 1
            i += 1
    if restart == True:
        main = None
        previous_i = index_restart
        current_i = index_restart
        i = 0
        n = 0
        cnt = index_restart/2000
        for index, row in files.iloc[index_restart:].iterrows():
            try:
                list_articles.append(parse_XML_article(
                    path = row["article_path"],
                    art_dir = row["article_dir"], 
                    title = row["article_name"],
                    index = row["index"]))
            except Exception as e:
                print(f"Index: {index}", e.args)
                continue
            # Each X, save the file in a .ftr
            if (i == save_each):
                current_i = current_i + i
                file_path = save_path+"processed_articles/processed_data_list_"+str(previous_i)+"_"+str(current_i)+".ftr"
                main = pd.DataFrame(list(chain.from_iterable(list_articles)))
                main.to_feather(file_path)
                main = None
                list_articles = []
                previous_i = current_i
                i = 0
            # Each 1000 files, print the progress
            if (i % 2000 == 0):
                clear_output(wait=True)
                display("Files parsed: "+str(2000*cnt))
                display("Current file: "+row["article_name"]+"(Index: "+str(row["index"])+")")
                cnt += 1
            i += 1

def iterate_metadata(save_path, files):
    """Iterate through files `files`, parse them and concatenate
    the result to be saved as a DataFrame in a feather object (.ftr)
    """
    main = None
    previous_i = 0
    current_i = 0
    i = 0
    n = 0
    cnt = 0
    dict_metadata = {}
    
    for index, row in files.iterrows():
        try:
            dict_metadata.update(
                parse_XML_metadata(
                    path = row["metadata_path"],
                    met_dir = row["metadata_dir"], 
                    title = row["metadata_name"],
                    index = row["index"]))
        except Exception as e:
            print(f"Index: {index}", e.args)
            continue
        # Each X, save the file in a .ftr
        if (i == 1000):
            current_i = current_i + i
            file_path = save_path+"processed_metadata/processed_metadata_"+str(previous_i)+"_"+str(current_i)+".ftr"
            main = pd.DataFrame.from_dict(dict_metadata).T.reset_index()
            main.to_feather(file_path)
            main = None
            dict_metadata = {}
            previous_i = current_i
            i = 0
        # Each 100 files, print the progress
        if (i % 100 == 0):
            clear_output(wait=True)
            display("Files parsed: "+str(2000*cnt))
            display("Current file: "+row["metadata_name"]+" (Index: "+str(row["index"])+")")
            cnt += 1
        i += 1

def iterate_ftr(df):
    """Iterate throught the .ftr files saved

    And append each of them.
    """
    result = pd.DataFrame()
    
    for index, row in df.iterrows():
        ftr = pd.read_feather(row["ftr_path"])
        result = result.append(ftr)
    
    return(result)