"""Build the webpage."""

import logging
import os
from argparse import ArgumentParser
from distutils.dir_util import copy_tree

# Use Django for its templating.
from django.conf import settings as django_settings
from django.template import Context
from django.template.loader import get_template


import data


_PARSER = ArgumentParser()
_PARSER.add_argument(
    "--output_dir", help="Directory to output website files.",
    default="gh-pages/")
_FLAGS = _PARSER.parse_args()


def _create_index():
    template = get_template("index.html")
    context = Context({
        'positions': data.POSITIONS,
        'publications': data.PUBLICATIONS,
    })
    with open(os.path.join(_FLAGS.output_dir, "index.html"), "w") as f:
        f.write(template.render(context))
    logging.info("Wrote index.html")


def _main():
    if not os.path.exists(_FLAGS.output_dir):
        os.makedirs(_FLAGS.output_dir)
    logging.info("Writing to %s", _FLAGS.output_dir)
    copy_tree("assets", os.path.join(_FLAGS.output_dir, "assets"))
    template_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates")
    django_settings.configure(
        TEMPLATE_DIRS=[template_dir],
        TEMPLATE_LOADERS=('django.template.loaders.filesystem.Loader',),)
    _create_index()
    logging.info(
        "Push changes by running:\n "
        "git subtree push --prefix gh-pages origin gh-pages")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _main()
