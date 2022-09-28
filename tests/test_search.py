from checker.api.search import SemanticSearch


def test_search():
    search = SemanticSearch()
    body = {
        "text": "When the New York State Senate voted to legalize abortion in 1970, 12 Republican senators voted in favor of it.",
        "num_results": 6
    }
    results = search.get_similar(body)
    for result in results:
        print(result["text"].strip())
