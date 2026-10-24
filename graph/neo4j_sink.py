"""OPTIONAL Neo4j mirror of detected rings (GRAPH_BACKEND=neo4j). NetworkX remains the detection engine.

Requires `pip install neo4j` and a reachable Neo4j server. This path is not exercised by the test-suite or the
reported runs (no Neo4j server was available); the default NetworkX path is fully functional without it.
"""

from __future__ import annotations


class Neo4jSink:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # optional dependency

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def write(self, rings) -> None:
        with self.driver.session() as s:
            for r in rings:
                s.run(
                    """MERGE (g:Ring {ring_id: $rid})
                       SET g.type = $type, g.score = $score, g.last_seen = $last
                       WITH g UNWIND $members AS m
                       MERGE (a:Account {id: m}) MERGE (a)-[:MEMBER_OF]->(g)""",
                    rid=r.ring_id,
                    type=r.ring_type,
                    score=float(r.score),
                    last=int(r.last_seen),
                    members=list(r.members),
                )
                s.run(
                    """UNWIND $edges AS e
                       MERGE (a:Account {id: e[0]}) MERGE (b:Account {id: e[1]})
                       MERGE (a)-[t:PAID {ts: e[3]}]->(b) SET t.amount = e[2]""",
                    edges=[list(e) for e in r.edges[:400]],
                )

    def close(self):
        self.driver.close()
