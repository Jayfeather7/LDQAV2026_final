"""
The main parser for the tagger. It takes a list of sections and returns a string of TEI markup is below:



CLI for parsing raw text files from the corpus of the Digital Dostoevsky Project
  and applying basic TEI markup.

"""
#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

from lxml import etree

from direct_speech import markup_direct_speech
from proper_names import markup_proper_names
from parse_sections import parse_sections, markup_sections


XML_PROCESSING_INSTRUCTIONS = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    """<?xml-model
  href="http://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng"
  type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>""",
    """<?xml-model
  href="http://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng"
  type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>""",
]


def create_tei_structure(text):
    """Create TEI structure for the given text."""

    return f"""\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title></title>
        <author/>
      </titleStmt>
      <editionStmt>
        <edition/>
        <respStmt>
          <resp/>
          <persName/>
        </respStmt>
      </editionStmt>
      <publicationStmt/>
      <sourceDesc/>
    </fileDesc>
  </teiHeader>
  <text>
      {text}
  </text>
</TEI>
"""


def format_tree(elem, indent="  ", level=0):
    """Pretty-prints and indents a tree beautifully."""
    i = "\n%s" % (level * indent)
    if elem.tag == "{http://www.tei-c.org/ns/1.0}p":
        return
    if elem.tag == "{http://www.tei-c.org/ns/1.0}head":
        return
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = "%s%s" % (i, indent)
        for sub_elem in elem:
            format_tree(sub_elem, indent, level + 1)
            if not sub_elem.tail or not sub_elem.tail.strip():
                sub_elem.tail = i
            if sub_elem.getnext() is not None:
                sub_elem.tail += indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i


def main():
    """Command-line entry-point."""

    parser = argparse.ArgumentParser(description="Description: {}".format(__doc__))
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Increase verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=False, help="Quiet operation"
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--rng-schema",
        action="store",
        help="RELAX NG schema to validate against (optional)",
    )
    parser.add_argument(
        "--person-names-list",
        action="store",
        help="Line-by-line list of person names (optional)",
    )
    parser.add_argument(
        "--place-names-list",
        action="store",
        help="Line-by-line list of place names (optional)",
    )
    parser.add_argument("input_text", action="store", help="Text to process")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_level = logging.CRITICAL if args.quiet else log_level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Processing input text: %s", args.input_text)

    with Path(args.input_text).open("r", encoding="utf8") as _fh:
        text = _fh.read()

    text = markup_direct_speech(text)

    if args.person_names_list:
        with Path(args.person_names_list).open("rt", encoding="utf8") as _fh:
            person_names = [line.strip() for line in _fh if line.strip()]
        text = markup_proper_names(text, person_names, "persName")

    if args.place_names_list:
        with Path(args.place_names_list).open("rt", encoding="utf8") as _fh:
            place_names = [line.strip() for line in _fh if line.strip()]
        text = markup_proper_names(text, place_names, "placeName")

    sections = parse_sections(text)
    text = markup_sections(sections)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = etree.fromstring(create_tei_structure(text).encode("utf8"))
    format_tree(doc)

    logging.info("Writing processed text to: %s", args.output or "stdout")

    with output_path.open("wt", encoding="utf8") if args.output else sys.stdout as _fh:
        _fh.write("\n".join(XML_PROCESSING_INSTRUCTIONS) + "\n")
        _fh.write(etree.tostring(doc, pretty_print=True, encoding="unicode"))

    if args.rng_schema:
        logging.info("Validating against schema: {}".format(args.rng_schema))
        relaxng_doc = etree.parse(args.rng_schema)
        relaxng = etree.RelaxNG(relaxng_doc)
        if not relaxng.validate(doc):
            for entry in relaxng.error_log:
                # Note: better validation errors are available with something like jing
                if (
                    entry.message == "Invalid attribute aloud for element said"
                    or entry.line in [15, 16]
                ):
                    # these errors are expected, given that dummy values are used
                    # -- only report if debug logging is enabled
                    logging.debug(entry)
                else:
                    logging.warning(entry)


if __name__ == "__main__":
    main()

"""
direct_speech.py: This file contains the markup_direct_speech function, which is responsible for identifying and marking up direct speech in the text. It uses regular expressions to find patterns of direct speech and wraps them in appropriate TEI tags.
"""
import regex as re

ATTRIBS = ["aloud", "direct", "who", "toWhom"]


def markup_direct_speech(text: str) -> str:
    """
    Markup direct speech in the given text with <said/> tags.
    See tests/test_direct_speech.py for examples.
    """

    # punctuation which can terminate an utterance (or part thereof)
    p = "?!).»"

    # designate two unicode "noncharacters" for use as temporary markers
    start = "\uFFFE"
    end = "\uFFFF"

    text_out = []
    for line in text.splitlines():

        # mark text inside bounding straight double quotes
        line = re.sub(r'«?"[^"\n]+"', rf"{start}\g<0>{end}", line)

        line_out = line

        # first pass for direct speech offset by an emdash
        line_out = re.sub(
            rf"""
            (?<!{start})          # onset is not already marked
            (?<!{end}\s)          # or follows immediately from a marked sequence

            (?:(?:                # emdash-offset utterance
                                  # -----------------------

              (?<![,{p}\w]\s)     # emdash does not follow terminating punctuation
              (—\s+)              # capture the emdash and any whitespace
              (.+?)               # lazily capture everything up to...
              (?=[,{p}]\s—|\n|$)  # terminating punctuation, a new emdash, or end of line

            ) | (?:               # guillemet-offset utterance
                                  # --------------------------

              («)                 # capture the guillemet
              ([^«]+?)            # lazily capture everything up to...
              (?=
                                  # an emdash if it's preceded by terminating punctuation
                                  #  that's not itself preceded by a marked sequence
                (?<!"|{end})[,{p}]\s—
                |»(?!{end})       # or a guillemet (unless it closes a marked sequence)
                |\n|$             # or end of line
              )
            ))
            ([{p}])?              # capture any trailing terminating punctuation
            """,
            lambda m: "".join(
                [
                    (m.group(1).strip() + " " if m.group(1) else m.group(3)),
                    start,
                    (m.group(2) or m.group(4)),
                    m.group(5) or "",
                    end,
                ]
            ),
            line_out,
            flags=re.VERBOSE,
        )

        # second pass for utterances offset with emdash and that resume after
        #  an inquit that terminates with a comma or a period
        if (
            # if there's already a marked utterance in this line
            f"— {start}" in line_out
            or re.search(rf"«{start}[^»]+{end}", line_out)
            or re.match(rf"«{start}[^»]+»{end}", line_out)
        ):
            line_out = re.sub(
                rf"""
                                  # onset follows a marked utterance earlier in the line
                (?<={end}[^{end}]+?) 
                (?<!{end})        # onset does not immediately follow a marked utterance

                ([,.])\s*         # capture a comma or period that terminates an inquit
                —\s+              # match the emdash and any whitespace
                (?!{start})       # unless the utterance is already marked
                (.+?)             # lazily capture everything up to...
                (?=
                  [,{p}]\s—       # an emdash preceded by terminating punctuation
                  |»(?!{end})     # or a guillemet (unless it closes a marked sequence)
                  |\n|$           # or end of line
                )
                ([{p}]+)?         # capture any trailing terminating punctuation
                """,
                rf"\g<1> — {start}\g<2>\g<3>{end}",
                line_out,
                flags=re.VERBOSE,
            )

        # post-hoc
        line_out = line_out.replace(f"«{start}", f"{start}«")
        line_out = line_out.replace(f"».{end}", f"»{end}.")

        # replace bounding guillemets with <said> tags
        line_out = re.sub(
            rf"(?<!{start})«.+?(»(?!{end})|\n|$)",
            rf"{start}\g<0>{end}",
            line_out,
        )

        # replace temporary markers with <said> tags
        line_out = line_out.replace(
            start, "<said {}>".format(" ".join(f'{k}=""' for k in ATTRIBS))
        )
        line_out = line_out.replace(end, "</said>")

        text_out.append(line_out)

    return "\n".join(text_out)


if __name__ == "__main__":
    # The following code is used for testing and development purposes only.
    import sys

    with open(sys.argv[1], "rt", encoding="utf8") as _fh:
        text = _fh.read()

    for line in text.splitlines():
        marked_up_line = markup_direct_speech(line)
        if line != marked_up_line:
            print(line)
            print(marked_up_line)
            print()

"""
proper_names.py: This file contains the markup_proper_names function, which is responsible for identifying and marking up proper names in the text. It uses regular expressions to find occurrences of specified names and wraps them in appropriate TEI tags.

"""

import re


def markup_proper_names(text, names, tag):

    re_proper_names = re.compile(
        rf"\b(?:{'|'.join(sorted(names, key=len, reverse=True))})\b"
    )
    text = re_proper_names.sub(rf"<{tag}>\g<0></{tag}>", text)

    return text

"""
parse_sections.py: This file contains the parse_sections function, which is responsible for parsing the text into sections based on specific markers and returning a structured representation of those sections. It also includes the markup_sections function, which takes the structured sections and generates TEI markup for them.

"""

import logging
import re


def is_roman_numeral(text):
    if not text:
        return False
    return all([c in "IVX" for c in text])


def roman_to_arabic(s: str) -> int:
    s = s.upper()
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    num = 0

    for i in range(len(s) - 1):
        if roman[s[i]] < roman[s[i + 1]]:
            num += roman[s[i]] * -1
            continue
        num += roman[s[i]]
    num += roman[s[-1]]

    return num


def parse_sections(text):
    titles = ["div1", "div2", "chapter_title"]
    first_section = True

    sections = []
    buffer = [""]  # ensure an empty line at the top of the document

    section_numeral = None
    for i, line in enumerate(text.splitlines()):
        line = line.strip(" \t\n")
        if is_roman_numeral(line):
            sections.append({"numeral": section_numeral, "lines": buffer})
            section_numeral = line
            buffer = []
        else:
            buffer.append(line)
    sections.append({"numeral": section_numeral, "lines": buffer})

    for i, section in enumerate(sections):
        if section["numeral"] is not None:
            prev_section = sections[i - 1]
            integer = roman_to_arabic(section["numeral"])

            if integer == 1:
                prev_titles = []

                # iterate lines from the previous section
                #  - bottom up, two at a time
                for current, previous in zip(
                    prev_section["lines"][::-1], prev_section["lines"][-2::-1]
                ):
                    # if the current line is not empty but the one above is
                    #  we have a "title" line of some sort
                    if current.strip() and not previous.strip():
                        # assign it to the next highest title level
                        prev_titles.append(current)
                        prev_section["lines"].remove(current)

                    # if we have two consecutive lines with content, we've
                    #  reached the bottom of the previous section's text
                    if current.strip() and previous.strip():
                        break

                if first_section:
                    first_section = False
                    prev_titles.reverse()
                    # must have a top-level title
                    section["title"] = prev_titles.pop(0)

                    # DUBIOUS: if there's more than one title left, take a subtitle
                    if len(prev_titles) > 1:
                        section["subtitle"], *prev_titles = prev_titles

                    prev_titles.reverse()

                    titles = titles[: len(prev_titles)]

                    titles.reverse()

                section["prev_titles"] = dict(reversed(list(zip(titles, prev_titles))))

            else:
                # sanity check -- consecutive roman numeral sections should have
                #  consecutive roman numerals (unless they are section I)
                if roman_to_arabic(prev_section["numeral"]) != integer - 1:
                    logging.warning(
                        "Section numbering is not correct "
                        f"({prev_section['numeral']=}, {section['numeral']=})"
                        f"\n\t - {section['lines'][1]}..."
                    )

            # pick up section titles for roman-numeraled sections
            # (Brothers Karamazov only)
            for j, line in enumerate(section["lines"]):
                # skip the first line, the last line, and empty lines
                if j == 0 or j == len(section["lines"]) - 1 or not line.strip():
                    continue

                # if we have two conecutive lines with content, give up and move on
                if section["lines"][j + 1].strip():
                    break

                # if we've got a line with blank lines on either side AND
                #  where all the content is upper case, take it as a section title
                if (
                    not section["lines"][j - 1].strip()
                    and not section["lines"][j + 1].strip()
                    and re.sub("<[^<]+?>", "", line).upper()
                    == re.sub("<[^<]+?>", "", line)
                ):
                    section["section_title"] = line.strip()
                    break

    return sections


def markup_sections(sections):
    first_section = True
    buffer = ["<body>"]
    section_attribs = {
        "div1": {"type": "part", "n": 0},
        "div2": {"type": "chapter", "n": 0},
    }

    def get_section_markup(section_type, heading_text):
        section_attribs[section_type]["n"] += 1
        attribs = " ".join(
            f'{k}="{v}"' for k, v in section_attribs[section_type].items()
        )
        return f"<{section_type} {attribs}><head>{heading_text}</head>"

    for section in sections:
        if section["numeral"] is None:
            front_matter = "\n".join(section["lines"]).strip()
            if front_matter:
                # <front/> section should be prepended so it goes before <body/>
                buffer = ["<front><p>", front_matter, "</p></front>", ""] + buffer
        else:
            if "title" in section:
                buffer.append(f'<head type="mainTitle">{section["title"]}</head>')
            if "subtitle" in section:
                buffer.append(f'<head type="subTitle">{section["subtitle"]}</head>')

            chapter_title = section.get("prev_titles", {}).pop("chapter_title", None)
            if "prev_titles" in section:
                if first_section:
                    has_chapters = len(section["prev_titles"]) > 1

                if not first_section:
                    if not has_chapters:
                        buffer.append("</div2>\n")
                    buffer.extend(
                        [f"</{key}>" for key in reversed(section["prev_titles"])]
                    )
                first_section = False

                if "div1" in section["prev_titles"]:
                    # restart chapter-level numbering
                    section_attribs["div2"]["n"] = 0

                buffer.extend(
                    [
                        get_section_markup(key, val)
                        for key, val in section["prev_titles"].items()
                    ]
                )

                if not has_chapters:
                    buffer.append("<div2>\n")

                if chapter_title is not None:
                    buffer.append(f'<head type="mainTitle">{chapter_title}</head>')

            integer = roman_to_arabic(section["numeral"])
            buffer.append(f'<div3 type="section" n="{integer}">')
            buffer.append(f"<head>{section['numeral']}</head>")
            if "section_title" in section:
                buffer.append(f"<head>{section['section_title']}</head>")
            buffer.extend(
                f"<p>{line.strip()}</p>" if line.strip() else ""
                for line in section["lines"]
            )
            buffer.append("</div3>\n")

    buffer.append("</div2>\n")
    buffer.append("</div1>\n")
    buffer.append("</body>")
    return "\n".join(buffer)


if __name__ == "__main__":
    # The following code is used for testing and development purposes only.
    import sys

    with open(sys.argv[1], "rt") as _fh:
        text = _fh.read()

    sections = parse_sections(text)

    print("<TEI>\n")
    print(markup_sections(sections))
    print("</TEI>")
