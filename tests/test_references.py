"""Every work the board cites has one entry in baselines/references.bib."""

import json
import sys

import pytest

from verify.common import ROOT

sys.path.insert(0, str(ROOT / "site"))
from references import Bibliography, normalize, parse  # noqa: E402

# Links that are not literature: this board's own files, and the participant
# repository the seeds were imported from.
REPOSITORY = "https://github.com/vxvware/"


def source_urls():
    urls = set()

    def walk(value):
        if isinstance(value, dict):
            if "url" in value and "title" in value:
                urls.add(value["url"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for folder in ("quadruples", "census", "certificates", "evidence", "baselines"):
        for path in (ROOT / folder).rglob("*.json"):
            walk(json.loads(path.read_text()))
    return sorted(urls)


def test_every_literature_source_has_a_reference():
    bib = Bibliography()
    missing = [
        url for url in source_urls() if not url.startswith(REPOSITORY) and not bib.resolve(url)
    ]
    assert not missing, f"sources without an entry in references.bib: {missing}"


def test_every_citation_marker_resolves():
    bib = Bibliography()
    registry = json.loads((ROOT / "baselines/contributions.json").read_text())
    for group in registry["contributions"]:
        bib.keys_in(group["summary"])
    with pytest.raises(ValueError, match="no entry"):
        bib.keys_in("an uncited claim [@nobody1900nothing]")


def test_contribution_prose_names_no_paper_without_a_marker():
    registry = json.loads((ROOT / "baselines/contributions.json").read_text())
    for group in registry["contributions"]:
        assert "arXiv:" not in group["summary"], group["id"]


def test_url_forms_of_one_work_agree():
    bib = Bibliography()
    assert (
        bib.resolve("https://arxiv.org/abs/1802.00556")
        == bib.resolve("https://arxiv.org/html/1802.00556")
        == bib.resolve("https://doi.org/10.1007/s11786-018-0381-1")
        == "djokovic2018goethals"
    )
    assert normalize("https://arxiv.org/html/0903.5053v2") == "arxiv:0903.5053"
    assert bib.resolve("https://zenodo.org/record/1195693/files/williamson-63.txt")


def test_numbers_are_a_stable_alphabetical_order():
    bib = Bibliography()
    order = sorted(bib.entries, key=bib.number.get)
    assert sorted(bib.number.values()) == list(range(1, len(order) + 1))
    # Đoković files under D, between Cati and Goethals.
    assert order.index("cati2024database") < order.index("djokovic2008skew")
    assert order.index("djokovic2018goethals") < order.index("goethals1970skew")


def test_citation_is_escaped_and_linked():
    bib = Bibliography()
    html = bib.cite("a <b> [@spence1977skew]", "../")
    assert "&lt;b&gt;" in html
    assert f'href="../references.html#ref-spence1977skew">[{bib.number["spence1977skew"]}]' in html


@pytest.mark.parametrize(
    "text",
    [
        "@article{a, title = {T}, year = {2000}}",
        "@article{a, author = {X, Y}, title = {T}, year = {2000}}\n"
        "@article{a, author = {X, Y}, title = {T}, year = {2000}}",
        "@article{a, author = {X, Y}, title = {T}, year = {2000}, colour = {red}}",
        "@article{a, author = {X, Y}, title = {T, year = {2000}}",
    ],
)
def test_malformed_bibliography_is_rejected(text):
    with pytest.raises(ValueError):
        parse(text)
