"""Cypher query constants for Neo4j operations.

All graph database queries live here, organized by operation type.
The temporal filtering pattern is consistent: every query that retrieves
data accepts an `event_order` parameter and filters nodes/edges by their
temporal validity range.

Naming convention:
- CREATE_*  : write operations
- GET_*     : single-entity reads
- FIND_*    : multi-entity reads / searches
- UPDATE_*  : modifications
- DELETE_*  : removals
"""


# ---------------------------------------------------------------------------
# Character (Person) nodes
# ---------------------------------------------------------------------------

CREATE_CHARACTER = """
MERGE (p:Person {name_lower: toLower($name)})
ON CREATE SET
    p.name        = $name,
    p.sex         = $sex,
    p.placeholder = $is_placeholder,
    p.exists_from = $exists_from,
    p.exists_until = $exists_until
RETURN p
"""

RENAME_PLACEHOLDER = """
MATCH (parent:Person {name_lower: toLower($parent_name)})
      -[:FAMILY {relation: 'parent_of'}]->(child:Person {placeholder: true})
WITH child LIMIT 1
SET child.name        = $child_name,
    child.name_lower  = toLower($child_name),
    child.placeholder = false
RETURN child
"""

SET_ATTRIBUTE = """
MATCH (p:Person {name_lower: toLower($name)})
MERGE (p)-[r:HAS_ATTRIBUTE {key: $key}]->(a:Attribute {key: $key})
ON CREATE SET
    a.value      = $value,
    a.valid_from = $valid_from,
    a.valid_until = $valid_until
ON MATCH SET
    a.value      = $value,
    a.valid_from = $valid_from,
    a.valid_until = $valid_until
RETURN a
"""


# ---------------------------------------------------------------------------
# Family relationships (FHKB base relations)
# ---------------------------------------------------------------------------

# A single canonical direction is stored per relationship. The reverse
# relation (child_of, or the mirror of a symmetric spouse_of/sibling_of)
# is never persisted — it is inferred by traversing the edge backwards at
# read time. See GraphStore._canonical_family_edge for the orientation rule.
CREATE_FAMILY_EDGE = """
MATCH (a:Person {name_lower: toLower($source)})
MATCH (b:Person {name_lower: toLower($target)})
MERGE (a)-[r:FAMILY {relation: $relation}]->(b)
ON CREATE SET
    r.valid_from  = $valid_from,
    r.valid_until = $valid_until
RETURN type(r) AS rel_type, r.relation AS relation
"""

CREATE_PLACEHOLDER_CHILDREN = """
UNWIND range(1, $count) AS i
MERGE (child:Person {name_lower: $key_prefix + '_child_' + toString(i)})
ON CREATE SET
    child.name        = $name_prefix + "'s child #" + toString(i),
    child.placeholder = true,
    child.exists_from = $exists_from
WITH child
UNWIND $parents AS parent_name
MATCH (parent:Person {name_lower: toLower(parent_name)})
MERGE (parent)-[r:FAMILY {relation: 'parent_of'}]->(child)
ON CREATE SET r.valid_from = $exists_from
RETURN DISTINCT child
"""


# ---------------------------------------------------------------------------
# Social relationships
# ---------------------------------------------------------------------------

CREATE_SOCIAL_EDGE = """
MATCH (a:Person {name_lower: toLower($source)})
MERGE (b:Concept {name_lower: toLower($target)})
ON CREATE SET b.name = $target
MERGE (a)-[r:SOCIAL {relation: $relation}]->(b)
ON CREATE SET
    r.valid_from  = $valid_from,
    r.valid_until = $valid_until
RETURN type(r) AS rel_type
"""

CREATE_SOCIAL_EDGE_TO_PERSON = """
MATCH (a:Person {name_lower: toLower($source)})
MATCH (b:Person {name_lower: toLower($target)})
MERGE (a)-[r:SOCIAL {relation: $relation}]->(b)
ON CREATE SET
    r.valid_from  = $valid_from,
    r.valid_until = $valid_until
RETURN type(r) AS rel_type
"""


# ---------------------------------------------------------------------------
# Temporal subgraph retrieval (GraphRAG core)
# ---------------------------------------------------------------------------

# Returns, per living character: outgoing family edges (the canonical
# stored direction), incoming family edges (so the reverse relation can be
# inferred at read time instead of being stored), and the attributes valid
# at this time point.
#
# Note: the three OPTIONAL MATCHes form a small cartesian product before
# collect(DISTINCT ...) dedupes each column. Fine at this scale; if the
# graph grows large, split each into its own CALL {} subquery.
FIND_SUBGRAPH_AT_TIME = """
MATCH (p:Person)
WHERE p.placeholder = false
  AND (p.exists_from IS NULL OR p.exists_from <= $event_order)
  AND (p.exists_until IS NULL OR p.exists_until >= $event_order)

OPTIONAL MATCH (p)-[ro:FAMILY]->(o:Person)
WHERE coalesce(o.placeholder, false) = false
  AND (ro.valid_from IS NULL OR ro.valid_from <= $event_order)
  AND (ro.valid_until IS NULL OR ro.valid_until >= $event_order)
  AND (o.exists_from IS NULL OR o.exists_from <= $event_order)
  AND (o.exists_until IS NULL OR o.exists_until >= $event_order)

OPTIONAL MATCH (p)<-[ri:FAMILY]-(i:Person)
WHERE coalesce(i.placeholder, false) = false
  AND (ri.valid_from IS NULL OR ri.valid_from <= $event_order)
  AND (ri.valid_until IS NULL OR ri.valid_until >= $event_order)
  AND (i.exists_from IS NULL OR i.exists_from <= $event_order)
  AND (i.exists_until IS NULL OR i.exists_until >= $event_order)

// Unnamed (placeholder) children are counted, never surfaced by name, so
// the LLM cannot mistake a placeholder label ("Kim & Jim's child #1") for a
// real character and re-ingest it as a new node on a later turn.
OPTIONAL MATCH (p)-[rc:FAMILY {relation: 'parent_of'}]->(pc:Person)
WHERE coalesce(pc.placeholder, false) = true
  AND (rc.valid_from IS NULL OR rc.valid_from <= $event_order)
  AND (pc.exists_from IS NULL OR pc.exists_from <= $event_order)
  AND (pc.exists_until IS NULL OR pc.exists_until >= $event_order)

OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:Attribute)
WHERE (a.valid_from IS NULL OR a.valid_from <= $event_order)
  AND (a.valid_until IS NULL OR a.valid_until >= $event_order)

RETURN p,
       collect(DISTINCT {relation: ro.relation, target: o.name}) AS out_relations,
       collect(DISTINCT {relation: ri.relation, source: i.name}) AS in_relations,
       count(DISTINCT pc) AS unnamed_children,
       collect(DISTINCT {key: a.key, value: a.value}) AS attributes
"""

FIND_CHARACTER_SUBGRAPH_AT_TIME = """
MATCH (center:Person {name_lower: toLower($name)})
WHERE (center.exists_from IS NULL OR center.exists_from <= $event_order)
  AND (center.exists_until IS NULL OR center.exists_until >= $event_order)
OPTIONAL MATCH (center)-[r:FAMILY|SOCIAL]-(connected)
WHERE (r.valid_from IS NULL OR r.valid_from <= $event_order)
  AND (r.valid_until IS NULL OR r.valid_until >= $event_order)
  AND (connected.exists_from IS NULL OR connected.exists_from <= $event_order)
  AND (connected.exists_until IS NULL OR connected.exists_until >= $event_order)
RETURN center, r, connected
"""


# ---------------------------------------------------------------------------
# Inferred family relations via path queries (FHKB-inspired)
# ---------------------------------------------------------------------------

FIND_GRANDPARENT = """
MATCH (gp:Person)-[:FAMILY {relation: 'parent_of'}]->()
      -[:FAMILY {relation: 'parent_of'}]->(gc:Person)
WHERE (gp.exists_from IS NULL OR gp.exists_from <= $event_order)
  AND (gc.exists_from IS NULL OR gc.exists_from <= $event_order)
RETURN gp.name AS grandparent, gc.name AS grandchild
"""

FIND_UNCLE_AUNT = """
MATCH (ua:Person)-[:FAMILY {relation: 'sibling_of'}]-(parent:Person)
      -[:FAMILY {relation: 'parent_of'}]->(niece:Person)
WHERE ua <> niece
  AND (ua.exists_from IS NULL OR ua.exists_from <= $event_order)
  AND (niece.exists_from IS NULL OR niece.exists_from <= $event_order)
RETURN ua.name AS uncle_aunt, niece.name AS niece_nephew
"""

FIND_COUSIN = """
MATCH (c1:Person)<-[:FAMILY {relation: 'parent_of'}]-(p1:Person)
      -[:FAMILY {relation: 'sibling_of'}]-(p2:Person)
      -[:FAMILY {relation: 'parent_of'}]->(c2:Person)
WHERE c1 <> c2
  AND (c1.exists_from IS NULL OR c1.exists_from <= $event_order)
  AND (c2.exists_from IS NULL OR c2.exists_from <= $event_order)
RETURN DISTINCT c1.name AS cousin1, c2.name AS cousin2
"""


# ---------------------------------------------------------------------------
# Timeline events
# ---------------------------------------------------------------------------

CREATE_TIMELINE_EVENT = """
MERGE (e:TimelineEvent {event_order: $order})
ON CREATE SET
    e.description       = $description,
    e.trigger_utterance = $trigger_utterance
RETURN e
"""

GET_ALL_TIMELINE_EVENTS = """
MATCH (e:TimelineEvent)
RETURN e
ORDER BY e.event_order ASC
"""


# ---------------------------------------------------------------------------
# Graph data for frontend visualization
# ---------------------------------------------------------------------------

GET_GRAPH_DATA = """
MATCH (p:Person)
OPTIONAL MATCH (p)-[r:FAMILY|SOCIAL]->(target)
RETURN p, collect(DISTINCT {
    relation: r.relation,
    target_name: target.name,
    target_name_lower: target.name_lower,
    target_id: elementId(target),
    rel_type: type(r)
}) AS edges
"""


# ---------------------------------------------------------------------------
# Conflict detection helpers
# ---------------------------------------------------------------------------

GET_ALL_CONTEXT = """
MATCH (p:Person {placeholder: false})
OPTIONAL MATCH (p)-[r:FAMILY]->(other:Person {placeholder: false})
OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:Attribute)
RETURN p.name AS name,
       collect(DISTINCT {relation: r.relation, target: other.name}) AS relations,
       collect(DISTINCT {key: a.key, value: a.value}) AS attributes
"""

CHECK_PERSON_EXISTS = """
MATCH (p:Person {name_lower: toLower($name), placeholder: false})
RETURN p IS NOT NULL AS exists
"""

COUNT_PLACEHOLDER_CHILDREN = """
MATCH (parent:Person {name_lower: toLower($parent_name)})
      -[:FAMILY {relation: 'parent_of'}]->(child:Person {placeholder: true})
RETURN count(child) AS placeholder_count
"""