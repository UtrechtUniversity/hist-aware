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
import logger

HAlogger = logger.get_logger("parser")
HAlogger.debug("Test message")

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