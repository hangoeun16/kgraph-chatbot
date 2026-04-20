# Knowledge graph for storing chatbot memory

import networkx as nx
import json
from langchain.schema import HumanMessage, SystemMessage


class KnowledgeGraph:
    """
    A knowledge graph for storing and managing chatbot conversation memory.

    This class uses NetworkX MultiDiGraph to store information about people,
    their attributes, and relationships. It integrates with an LLM to extract
    structured information from natural language conversations.

    Attributes:
        graph (nx.MultiDiGraph): The underlying NetworkX graph storing nodes and edges.
        llm: The language model instance used for information extraction.
    """

    def __init__(self, llm=None):
        """
        Initialize a new KnowledgeGraph instance.

        Args:
            llm: Optional language model instance for extracting information
                 from text. If None, extraction features will be disabled.
        """
        self.graph = nx.MultiDiGraph()
        self.llm = llm

        print("Knowledge Graph created!")

    def add_message(self, message, role):
        """
        Process a message and extract information to store in the knowledge graph.

        For user messages, this method checks for conflicts with existing information,
        then extracts family relationships, personal attributes, and other relationships.

        Args:
            message (str): The message text to process.
            role (str): The role of the message sender ('user' or 'assistant').

        Returns:
            dict or None: A conflict dictionary if a conflict is detected, containing
                          'subject', 'existing', 'new', and 'explanation' keys.
                          Returns None if no conflict is found.
        """
        # process a user message and extract info
        # returns conflict if there is one

        # check for conflicts (only for user messages)
        if role == "user" and self.llm != None:
            # first check if naming unnamed people
            is_naming = self.check_if_naming_people(message)

            if is_naming == False:
                conflict = self.check_conflict(message)
                if conflict != None:
                    return conflict

        # extract info from user messages
        # extract_family first so placeholders get renamed before other extractions
        if self.llm != None and role == "user":
            self.extract_family(message)
            self.extract_attributes(message)
            self.extract_relationships(message)

        return None

    def get_relevant_context(self):
        """
        Retrieve relevant context from the knowledge graph for conversation.

        Generates a human-readable summary of all stored information about
        people and their relationships to provide context for the chatbot.

        Returns:
            str: A formatted string containing all known information about
                 people and relationships, or a message indicating the graph
                 is empty if no information has been stored.
        """
        # get context from graph
        if self.graph.number_of_nodes() == 0:
            return "No previous informative conversation yet (knowledge graph is empty)."

        graph_context = self.get_graph_context()
        if graph_context != "":
            return graph_context

        return "No previous informative conversation yet (knowledge graph is empty)."

    def get_graph_context(self):
        """
        Generate a text summary of all people and relationships in the graph.

        Iterates through all non-placeholder person nodes and compiles their
        attributes and relationships into human-readable sentences.

        Returns:
            str: A newline-separated string of facts about people and their
                 relationships, or an empty string if no named people exist.
        """
        # gather all information of people the graph

        info = []

        # get non-placeholder person nodes only
        person_nodes = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'person':
                if not self.graph.nodes[node].get('placeholder', False):
                    person_nodes.append(node)

        if len(person_nodes) == 0:
            return ""

        # for each person, save information using attribute and edge
        for node in person_nodes:
            node_data = self.graph.nodes[node]
            name = node_data['name']

            skip_attribute = ['type', 'name', 'placeholder']
            for attribute in node_data:
                if attribute not in skip_attribute:
                    info.append(name + "'s " + attribute + " is " + str(node_data[attribute]) + ".")

            # count unnamed children instead of listing placeholder names
            unnamed_child_count = 0
            for source, target, edge in self.graph.out_edges(node, data=True):
                relationship = edge.get('relation', '')
                target_data = self.graph.nodes[target]

                if target_data.get('placeholder', False):
                    if relationship == 'child':
                        unnamed_child_count += 1
                else:
                    target_name = target_data['name']
                    # convert stored relation to readable format
                    if relationship == 'child':
                        info.append(name + " is parent of " + target_name + ".")
                    elif relationship == 'parent':
                        info.append(name + " is child of " + target_name + ".")
                    else:
                        info.append(name + " is " + relationship + " of " + target_name + ".")

            if unnamed_child_count > 0:
                info.append(name + " has " + str(unnamed_child_count) + " unnamed children.")

        return "\n".join(info)

    def extract_family(self, text):
        """
        Extract family relationships from text using the language model.

        Parses the input text to identify family relationships such as
        spouse, parent, child, and sibling connections. Also handles
        naming of previously unnamed children.

        Args:
            text (str): The text to analyze for family relationships.

        Note:
            Requires an LLM to be configured. Does nothing if llm is None.
            Extracted relationships are automatically added to the graph.
        """
        # extract family relationships using prompt
        if self.llm == None:
            return

        context = self.get_relevant_context()

        # get all people node (exclude placeholders to avoid LLM confusion)
        existing_people = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'person':
                if not self.graph.nodes[node].get('placeholder', False):
                    existing_people.append(self.graph.nodes[node]['name'])

        existing_str = ", ".join(existing_people) if len(existing_people) > 0 else "None yet"

        prompt = """Extract family relationships from this text.

Context: """ + context + """

Existing people: """ + existing_str + """

Text: \"""" + text + """\"

Return a JSON array. Each item can be:

1. New relationship:
{"person1": "Name", "person2": "Name", "relation": "spouse/parent/child/sibling"}

2. Count of unnamed children:
{"person1": "Parent", "relation": "child", "count": 3}

3. Naming an unnamed child (IMPORTANT: use this when someone already has unnamed children and user gives a name):
{"action": "name_child", "parent": "Parent", "child_name": "Rick"}

Examples:
- "Kim is married to Jim" -> [{"person1": "Kim", "person2": "Jim", "relation": "spouse"}]
- "They have 3 kids" -> [{"person1": "Kim", "relation": "child", "count": 3}, {"person1": "Jim", "relation": "child", "count": 3}]
- "The first son's name is Rick" (when parent has unnamed children) -> [{"action": "name_child", "parent": "Kim", "child_name": "Rick"}]

Return ONLY JSON array. If nothing found return []"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Extract family relationships. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content.strip()
            start = response_text.find('[')
            end = response_text.rfind(']') + 1

            if start != -1 and end > start:
                relationships = json.loads(response_text[start:end])
                self.add_relationships(relationships)
        except Exception as error:
            print("Could not extract family: " + str(error))

    def add_relationships(self, relationships):
        """
        Add extracted relationship data to the knowledge graph.

        Processes a list of relationship dictionaries and creates appropriate
        nodes and edges in the graph. Handles special cases like naming
        unnamed children and creating shared children for couples.

        Args:
            relationships (list): List of relationship dictionaries, each containing:
                - For named relationships: 'person1', 'person2', 'relation'
                - For child counts: 'person1', 'relation', 'count'
                - For naming children: 'action': 'name_child', 'parent', 'child_name'
        """
        # add extracted relationships to graph
        shared_children = {}

        for relationship in relationships:
            # handle name_child action from prompt
            action = relationship.get('action', '').strip().lower()
            if action == 'name_child':
                parent = relationship.get('parent', '').strip()
                child_name = relationship.get('child_name', '').strip()
                if parent and child_name:
                    self.try_replace_placeholder(parent, child_name)
                continue

            person1_name = relationship.get('person1', '').strip()
            person2_name = relationship.get('person2', '').strip()
            relation = relationship.get('relation', '').strip().lower()
            count = relationship.get('count', 0)

            # normalize relationships
            # son/daughter means person2 is child of person1
            if relation in ['son', 'daughter']:
                relation = 'child'
            # father/mother/parent means person1 is parent of person2, so person2 is child of person1
            if relation in ['father', 'mother', 'parent']:
                relation = 'child'

            if person1_name != '' and person2_name != '' and relation != '':
                self.add_person(person1_name)
                self.add_person(person2_name)
                self.add_edge(person1_name, person2_name, relation)

            elif person1_name != '' and relation != '' and count > 0:
                children = (relation, count)
                if children not in shared_children:
                    shared_children[children] = []
                shared_children[children].append(person1_name)

        # create shared children
        for children in shared_children:
            relation, count = children
            parents = shared_children[children]
            if len(parents) > 1:
                self.add_shared_children(parents, relation, count)
            else:
                self.add_children(parents[0], relation, count)

    def add_person(self, name):
        """
        Add a person node to the knowledge graph if it doesn't already exist.

        Creates a new node with type 'person' and the given name. Checks for
        existing nodes by both ID and name to prevent duplicates.

        Args:
            name (str): The name of the person to add.
        """
        # add person node if doesnt exist
        name_lower = name.lower()
        node_id = "person_" + name_lower.replace(' ', '_')

        # check if node with this ID exists
        if self.graph.has_node(node_id):
            return

        # check if person with this name exists (e.g. renamed placeholder)
        for node in self.graph.nodes():
            if self.graph.nodes[node]['name'].lower() == name_lower:
                return  

        self.graph.add_node(node_id, type='person', name=name)

    def add_edge(self, person1, person2, relation):
        """
        Add a relationship edge between two people in the knowledge graph.

        Creates a directed edge from person1 to person2 with the specified
        relation. Also automatically creates the reverse edge for bidirectional
        relationships (spouse, parent/child, sibling).

        Args:
            person1 (str): The name of the first person (source of the edge).
            person2 (str): The name of the second person (target of the edge).
            relation (str): The type of relationship (e.g., 'spouse', 'parent',
                           'child', 'sibling').
        """
        # add edge between two people
        person1_id = "person_" + person1.lower().replace(' ', '_')
        person2_id = "person_" + person2.lower().replace(' ', '_')

        if self.graph.has_edge(person1_id, person2_id) == False:
            self.graph.add_edge(person1_id, person2_id, relation=relation)

        # add reverse edge
        reverse_relations = {'spouse': 'spouse', 'parent': 'child', 'child': 'parent', 'sibling': 'sibling'}
        if relation in reverse_relations:
            if self.graph.has_edge(person2_id, person1_id) == False:
                self.graph.add_edge(person2_id, person1_id, relation=reverse_relations[relation])

    def try_replace_placeholder(self, parent_name, child_name):
        """
        Attempt to replace a placeholder child node with a real name.

        When a parent has unnamed placeholder children and the user provides
        a name for one of them, this method renames the placeholder node
        instead of creating a duplicate.

        Args:
            parent_name (str): The name of the parent who has placeholder children.
            child_name (str): The actual name to assign to one of the placeholder children.

        Returns:
            bool: True if a placeholder was successfully renamed or child already exists,
                  False if the parent doesn't exist or has no placeholder children.
        """
        # if parent has placeholder children, rename one instead of creating new node
        parent_id = "person_" + parent_name.lower().replace(' ', '_')

        if not self.graph.has_node(parent_id):
            return False

        # check if child already exists for this parent
        for source, target, edge in self.graph.out_edges(parent_id, data=True):
            if edge.get('relation') == 'child':
                existing_name = self.graph.nodes[target].get('name', '').lower()
                if existing_name == child_name.lower():
                    return True  # already exists, skip

        # find a placeholder to replace
        for source, target, edge in self.graph.out_edges(parent_id, data=True):
            if edge.get('relation') == 'child':
                if self.graph.nodes[target].get('placeholder', False):
                    self.graph.nodes[target]['name'] = child_name
                    self.graph.nodes[target]['placeholder'] = False
                    print("Renamed placeholder to " + child_name)
                    return True

        return False

    def add_shared_children(self, parents, relation, count):
        """
        Add placeholder child nodes shared between multiple parents.

        Creates placeholder nodes for children that belong to multiple parents
        (typically a couple). Each child node is connected to all specified parents.

        Args:
            parents (list): List of parent names who share the children.
            relation (str): The type of relation (typically 'child').
            count (int): The number of placeholder children to create.
        """
        # add placeholder children for multiple parents
        parents_str = " and ".join(parents)

        for index in range(1, count + 1):
            child_name = parents_str + "'s " + relation + " #" + str(index)
            child_id = "person_" + "_".join([parent.lower().replace(' ', '_') for parent in parents]) + "_" + relation + "_" + str(index)

            if self.graph.has_node(child_id) == False:
                self.graph.add_node(child_id, type='person', name=child_name, placeholder=True)

            for parent in parents:
                self.add_person(parent)
                parent_id = "person_" + parent.lower().replace(' ', '_')

                if relation == 'child':
                    if self.graph.has_edge(parent_id, child_id) == False:
                        self.graph.add_edge(parent_id, child_id, relation='child')
                    if self.graph.has_edge(child_id, parent_id) == False:
                        self.graph.add_edge(child_id, parent_id, relation='parent')

    def add_children(self, parent, relation, count):
        """
        Add placeholder child nodes for a single parent.

        Creates placeholder nodes representing unnamed children of a parent.
        Each placeholder can later be renamed when the user provides actual names.

        Args:
            parent (str): The name of the parent.
            relation (str): The type of relation (typically 'child').
            count (int): The number of placeholder children to create.
        """
        # add placeholder children for single parent
        self.add_person(parent)
        parent_lower = parent.lower().replace(' ', '_')

        for index in range(1, count + 1):
            child_name = parent + "'s " + relation + " #" + str(index)
            child_id = "person_" + parent_lower + "_" + relation + "_" + str(index)

            if self.graph.has_node(child_id) == False:
                self.graph.add_node(child_id, type='person', name=child_name, placeholder=True)

            parent_id = "person_" + parent_lower
            if relation == 'child':
                if self.graph.has_edge(parent_id, child_id) == False:
                    self.graph.add_edge(parent_id, child_id, relation='child')
                if self.graph.has_edge(child_id, parent_id) == False:
                    self.graph.add_edge(child_id, parent_id, relation='parent')

    def extract_attributes(self, text):
        """
        Extract personal attributes from text using the language model.

        Parses the input text to identify attributes like age, occupation,
        location, personality, hobbies, education, gender, and appearance
        for people mentioned in the conversation.

        Args:
            text (str): The text to analyze for personal attributes.

        Note:
            Requires an LLM to be configured. Does nothing if llm is None.
            Extracted attributes are automatically added to the corresponding
            person nodes in the graph.
        """
        # extract age, job, location
        if self.llm == None:
            return

        context = self.get_relevant_context()

        existing_people = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'person':
                if self.graph.nodes[node].get('placeholder', False) == False:
                    existing_people.append(self.graph.nodes[node].get('name', ''))

        people_str = ", ".join(existing_people) if len(existing_people) > 0 else "None yet"

        prompt = """Extract any personal attributes from this text.

Context: """ + context + """
Existing people: """ + people_str + """
Text: \"""" + text + """\"

Extract any attribute about a person such as:
age, occupation, location, personality, hobby, education, gender, nickname, appearance, etc.

Return JSON array: [{"person": "Name", "attribute": "personality", "value": "rebellious"}]
Return ONLY JSON. If nothing found return []"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Extract attributes. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content.strip()
            start = response_text.find('[')
            end = response_text.rfind(']') + 1

            if start != -1 and end > start:
                attributes = json.loads(response_text[start:end])

                for attribute_item in attributes:
                    person = attribute_item.get('person', '').strip()
                    attribute_name = attribute_item.get('attribute', '').strip().lower()
                    value = attribute_item.get('value', '').strip()

                    if person != '' and attribute_name != '' and value != '':
                        self.add_person(person)
                        node_id = "person_" + person.lower().replace(' ', '_')
                        self.graph.nodes[node_id][attribute_name] = value
                        print("Set " + person + "." + attribute_name + " = " + value)
        except Exception as error:
            print("Could not extract attributes: " + str(error))

    def extract_relationships(self, text):
        """
        Extract non-family relationships from text using the language model.

        Parses the input text to identify social and personal relationships
        such as likes, loves, dislikes, knows, friends_with, works_with,
        owns, and wants. Can relate people to other people or concepts.

        Args:
            text (str): The text to analyze for relationships.

        Note:
            Requires an LLM to be configured. Does nothing if llm is None.
            Extracted relationships are automatically added as edges in the graph,
            with concept nodes created as needed for non-person targets.
        """
        # extract non-family relationships (likes, knows, etc)
        if self.llm == None:
            return

        context = self.get_relevant_context()

        existing_people = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'person':
                if self.graph.nodes[node].get('placeholder', False) == False:
                    existing_people.append(self.graph.nodes[node]['name'])

        people_str = ", ".join(existing_people) if len(existing_people) > 0 else "None yet"

        prompt = """Extract non-family relationships from this text.

Context: """ + context + """
Existing people: """ + people_str + """
Text: \"""" + text + """\"

Look for: likes, loves, dislikes, knows, friends_with, works_with, owns, wants

Return JSON array: [{"person": "Name", "relation": "likes", "target": "Pizza", "target_type": "concept"}]
target_type can be "person" or "concept"
Return ONLY JSON. If nothing found return []"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Extract relationships. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content.strip()
            start = response_text.find('[')
            end = response_text.rfind(']') + 1

            if start != -1 and end > start:
                relationships = json.loads(response_text[start:end])

                for relationship in relationships:
                    person = relationship.get('person', '').strip()
                    relation = relationship.get('relation', '').strip().lower()
                    target = relationship.get('target', '').strip()
                    target_type = relationship.get('target_type', 'concept').strip().lower()

                    if person != '' and relation != '' and target != '':
                        self.add_person(person)
                        person_id = "person_" + person.lower().replace(' ', '_')

                        if target_type == 'person':
                            self.add_person(target)
                            target_id = "person_" + target.lower().replace(' ', '_')
                        else:
                            target_id = "concept_" + target.lower().replace(' ', '_')
                            if self.graph.has_node(target_id) == False:
                                self.graph.add_node(target_id, type='concept', name=target)
                        if self.graph.has_edge(person_id, target_id) == False:
                            self.graph.add_edge(person_id, target_id, relation=relation)
                            print("Added: " + person + " --" + relation + "--> " + target)
        except Exception as error:
            print("Could not extract relationships: " + str(error))

    def check_conflict(self, new_message):
        """
        Check if a new message conflicts with existing information in the graph.

        Uses the language model to detect direct contradictions between the
        new message and previously stored facts. Only flags true conflicts
        (e.g., different ages for the same person), not additions or expansions.

        Args:
            new_message (str): The new user message to check for conflicts.

        Returns:
            dict or None: If a conflict is found, returns a dictionary with:
                - 'subject': Who the conflict is about
                - 'existing': The previously stored statement
                - 'new': The conflicting new statement
                - 'explanation': Why this is a conflict
            Returns None if no conflict is detected or LLM is not configured.
        """
        # check if message conflicts with existing info
        if self.llm == None:
            return None

        context = self.get_relevant_context()
        if context == "" or context == "No previous conversation history.":
            return None

        prompt = """Check if the new statement DIRECTLY CONTRADICTS existing info.

Existing conversation:
""" + context + """

New statement: \"""" + new_message + """\"

A conflict is ONLY when:
- Same person has TWO DIFFERENT ages (e.g. "Jim is 45" then "Jim is 30")
- Same person has TWO DIFFERENT spouses (e.g. "Jim married to Mary" then "Jim married to Sue")
- Direct factual contradiction of something already stated

NOT a conflict (these are fine):
- Adding new people (e.g. adding a spouse when none was mentioned)
- Adding new relationships (e.g. adding parents, children, friends)
- Adding more details about existing people
- Naming unnamed people
- Expanding on previous information

Example that is NOT a conflict:
- "Carry is Jim's daughter" then "Lydia is Jim's wife" = NOT a conflict (just adding wife)

Example that IS a conflict:
- "Jim is married to Mary" then "Jim is married to Sue" = CONFLICT (two different wives)

Return JSON:
{
  "has_conflict": true/false,
  "subject": "who its about",
  "existing_statement": "what was said before",
  "new_statement": "what was said now",
  "explanation": "why its a conflict"
}

Return ONLY JSON. Default to has_conflict: false if unsure."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Detect conflicts. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content.strip()
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
                if result.get('has_conflict', False) == True:
                    return {
                        'subject': result.get('subject', 'Unknown'),
                        'existing': result.get('existing_statement', ''),
                        'new': result.get('new_statement', new_message),
                        'explanation': result.get('explanation', 'Conflicting information')
                    }
        except Exception as error:
            print("Could not check conflict: " + str(error))

        return None

    def check_if_naming_people(self, message):
        """
        Check if the user is naming previously unnamed placeholder people.

        Uses the language model to determine if the message is providing names
        for placeholder children or other unnamed individuals in the graph.
        This is used to skip conflict detection when naming unnamed people.

        Args:
            message (str): The user message to analyze.

        Returns:
            bool: True if the message appears to be naming unnamed people,
                  False otherwise or if no LLM is configured or no placeholders exist.
        """
        # check if user is naming previously unnamed people
        if self.llm == None:
            return False

        # check if there are placeholders
        has_placeholders = any(
            self.graph.nodes[node_id].get('placeholder', False)
            for node_id in self.graph.nodes()
        )
        if has_placeholders == False:
            return False

        context = self.get_relevant_context()

        prompt = """Is this statement naming previously unnamed people?

Existing conversation:
""" + context + """

New statement: \"""" + message + """\"

Return JSON: {"is_naming_unnamed": true/false}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Detect naming. Return ONLY JSON."),
                HumanMessage(content=prompt)
            ])

            response_text = response.content.strip()
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
                return result.get('is_naming_unnamed', False)
        except:
            pass

        return False

    def get_graph_data(self):
        """
        Get the knowledge graph data formatted for frontend visualization.

        Extracts all person nodes and their relationships into a format
        suitable for rendering with a graph visualization library.

        Returns:
            dict: A dictionary containing:
                - 'nodes': List of node dictionaries with 'id', 'name', and 'attributes'
                - 'edges': List of edge dictionaries with 'from', 'to', and 'relation'
        """
        # get nodes and edges for visualisation
        nodes = []
        edges = []

        # get all person nodes
        person_nodes = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'person':
                person_nodes.append(node)

        # build node list with attributes
        for node in person_nodes:
            node_data = self.graph.nodes[node]
            name = node_data.get('name', 'Unknown')

            # get all attributes
            attributes = {}
            skip_attributes = ['type', 'name', 'placeholder']
            for attribute in node_data:
                if attribute not in skip_attributes:
                    attributes[attribute] = str(node_data[attribute])

            nodes.append({
                'id': node,
                'name': name,
                'attributes': attributes
            })

        # build edge list
        for node in person_nodes:
            for source, target, edge_data in self.graph.out_edges(node, data=True):
                # only edges to other persons
                if target in person_nodes:
                    relation = edge_data.get('relation', 'related')
                    edges.append({
                        'from': node,
                        'to': target,
                        'relation': relation
                    })

        return {'nodes': nodes, 'edges': edges}
