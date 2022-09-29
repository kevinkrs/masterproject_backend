from search import SemanticSearch


def test_search():
    search = SemanticSearch()
    body = {
        "text": "Russia “has not lost anything” in its invasion of Ukraine.",
        "statementdate": "2022-07-09",
        "source": "http://en.kremlin.ru/events/president/transcripts/69299",
        "author": "Vladimir Putin",
        "num_results": 6
    }
    results = search.get_similar(body)
    for result in results:
        print(result["title"])
        print(result["label"])
        print(result["scores"])
        print(result["url"])


test_search()