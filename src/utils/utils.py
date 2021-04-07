import os
import re
from tqdm import tqdm


def clean_iter(id):
    """Find ids in metadata identifier to rebuild identifier."""

    idNum = str(re.findall(r"\d\d\d\d\d\d\d\d\d", id)[0])
    recNum = str(re.findall(r"\d\d\d\d", id)[2])
    return "DDD_" + idNum + "_" + recNum + "_articletext.xml"


def clean_article_identifier(csv):
    """Transform article identifier from metadata type to article-compatible."""

    tqdm.pandas()
    # TODO: test if tqdm works
    csv["transformedRecordIdentifier"] = csv.progress_apply(
        lambda row: clean_iter(row["recordIdentifier"]), axis=1
    )
    return csv


def make_noise():
    """Make noise after finishing executing a code"""
    duration = 4  # seconds
    freq = 440  # Hz
    os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))
