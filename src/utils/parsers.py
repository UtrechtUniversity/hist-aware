# parsers.py
from datetime import datetime
from functools import reduce
import re
import pathlib
import xmltodict
import xml.etree.ElementTree as et


def parse_XML_article(path, art_dir, title, index):
    """Parse the input XML file and store the result in a pandas
    DataFrame with the given columns.

    Takes the filepath, file title and index integer of the df
    """

    xtree = et.parse(path)
    xroot = xtree.getroot()

    # Parse the date with regex
    match = re.search(r"\d{4}[/]\d{2}[-]\d{2}", path)
    date = datetime.strptime(match.group(), "%Y/%m-%d").date()

    list_p = []
    article = {}
    for i, node in enumerate(xroot):
        article["article_name"] = str(title)
        article["date"] = str(date)
        article["index"] = index
        article["filepath"] = path
        article["dir"] = art_dir
        if node.tag == "title":
            article["title"] = node.text
        else:
            list_p.append(node.text)
    article["p"] = list_p
    list_p = []

    # Returns list of dict of articles and titles
    return article


def parse_XML_metadata(path, met_dir, title, index):
    """Parse the input XML file and store the result in a pandas
    DataFrame with the given columns.

    Takes the filepath, file title and index integer of the df
    """
    metadata = {}
    list_articles = []

    # Parse the date with regex
    match = re.search(r"\d{4}[/]\d{2}[-]\d{2}", path)
    date = datetime.strptime(match.group(), "%Y/%m-%d").date()

    # Parse DIDL XML
    with open(pathlib.Path(path), "r") as f:
        file = f.read()
        doc = xmltodict.parse(file)
        temp_data = doc["didl:DIDL"]["didl:Item"]["didl:Component"][0]["didl:Resource"][
            "srw_dc:dcx"
        ]
        pages = doc["didl:DIDL"]["didl:Item"]

    metadata["metadata_title"] = title
    metadata["date"] = date
    metadata["index"] = index
    metadata["filepath"] = path
    metadata["dir"] = met_dir

    # Retrieve informations about the newspaper
    metadata["newspaper_title"] = deep_get(temp_data, "dc:title")
    metadata["newspaper_date"] = deep_get(temp_data, "dc:date")
    metadata["newspaper_publisher"] = deep_get(temp_data, "dc:publisher")
    metadata["newspaper_source"] = deep_get(temp_data, "dc:source")
    metadata["newspaper_volume"] = deep_get(temp_data, "dcx:volume")
    metadata["newspaper_issuenumber"] = deep_get(temp_data, "dcx:issuenumber")
    metadata["newspaper_recordIdentifier"] = deep_get(temp_data, "dcx:recordIdentifier")

    for page in range(len(pages["didl:Item"])):
        p = pages["didl:Item"][page]
        try:
            articles = p["didl:Item"]
            for article in range(len(articles)):
                article_dict = {}
                art = articles[article]
                article_dict["subject"] = art["didl:Component"][0]["didl:Resource"][
                    "srw_dc:dcx"
                ]["dc:subject"]
                article_dict["title"] = art["didl:Component"][0]["didl:Resource"][
                    "srw_dc:dcx"
                ]["dc:title"]
                article_dict["access_rights"] = art["didl:Component"][0][
                    "didl:Resource"
                ]["srw_dc:dcx"]["dcterms:accessRights"]
                article_dict["recordIdentifier"] = art["didl:Component"][0][
                    "didl:Resource"
                ]["srw_dc:dcx"]["dcx:recordIdentifier"]
                article_dict["identifier"] = art["didl:Component"][0]["didl:Resource"][
                    "srw_dc:dcx"
                ]["dc:identifier"]
                # Append newspaper-and metadata-specific data
                article_dict.update(metadata)
                list_articles.append(article_dict)
        except KeyError:
            continue

    return list_articles


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )
