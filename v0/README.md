# KGraph Chatbot

A chatbot that remembers family relationships and personal information using a knowledge graph. 
I was motivated to do this because one of my hobby is to do the story telling ping-pong with ChatGPT, and one of its recognisable drawback is that GPT tends to forget the family relationship that is builded through the chat. To resolve that issue, I was thinking of creating a chatbot that shows family tree along with the chat ui. While doing the research, I realised that the graph that I saved to draw hierarchical family tree can be used as an external storage. So, that broadened the use of the graph, and eventually ended up with this project. 

### What it does

- Chat with GPT-4o in a friendly conversational style
- Stores family relationships in a knowledge graph (NetworkX)
- Extracts people, family relationships, and personal attributes from messages
- Detects conflicts with previously stored information
- Visualises unhierarchical family tree as an interactive graph
- Moderates using OpenAI's Moderation API

### Features

- **Family extraction**: Automatically extracts family relationships such as spouse, parent, child, etc.
- **Attribute extraction**: Automatically extracts attributes such as age, occupation, personality, etc. 
- **Placeholder naming**: Use placeholder when name is ungiven and later
replace it the actual name when user provides. 
- **Conflict detection**: Flags when the user provides information that conflicts with information that user earlier provided. 
- **Graph visualisation**: Click the tree button to see family connections
- **Content safety**: Filters inappropriate content before processing

### Files
Directory structure:
```
girly-chatbot-kg3/
├── app.py: Flask web app hosting the web page
├── chatbot.py: chatbot class 
├── knowledge_graph.py: stores information from the chat in a graph 
├── content_moderator.py: content safety checking
├── README.md: description of the program
└── templates/
    └── index.html: UI of the chatbot

```

### Set up 

1. Set open ai API key
```
export OPENAI_API_KEY="your-key"
```

2. Set virtual environment:
Since there can be package version compatability issue, use virtual environmnet
```
python -m venv venv_name
```

3. Install required packages 
```
pip install -r requirements.txt
```

4. Then activate it
```
python app.py
```

### Tried attempts that is not on the final version
1. heavy prompt engineering for the 
Used multiple layer of prompt engineering to do the following: detect if the user provided input should be flagged; check if user's input involves family relationship such that it can be parsed into (person 1, relationship, person2) triples format and if possible, return the triples; check if the user input gives includes the relationship between two entity but not the family relationship, if so parse into (entity1, relationship, entity2) triples format; if the user's input does not include any relational information, extract the attributes. 

Using all these prompts made the program so heavy that even getting response for "hi" took more than one minute. 

2. semantic search with knowledge graph 
When generate chat response, used semantic search along with Chromadb. While it was generating accurate response, there was unsolved Chromadb cache issue even with the code that manually creates collection. Also, storing information twice turned out to be inefficient. 

3. other ui (chat in the left, graph on the right / second page for graph)
First, I to create a multiple pages: first page for chatbot and second page for showing the family tree. However, this was not convenient. Next, I tried having chat in the leftside of the page and graph on the right side of the page. Turned out that this juxaposition is inefficient unless the graph is automatically updated whenever user gives input, which is computational expensive. Then, I realised that if I prefer the updated graph to be presented to the user, then it would be better if user can choose to do it when user wants. So, I ended up creating a button to display, rather than showing it as a default. 

4. creating a hierarchical family tree
- I tried using open-source python visualisation packages, but most of them are not able to graph top-down hierarchical family tree. Also, while there was a package that allows vertical hierarchical family tree, it's input required a specific format of txt file. To run that package within my program, I had to reformat all the graph information to the txt format that the program wants. While reformatting the graph based information itself was difficult, dealing with placeholders was an additional issue: while my program do not manually make place holders unless mentioned by the user, the package required complete family generation information (for instance, all people who have a parent must have both father and mother). 
- Additionally, I tried to use an attribute call generation. The first people initially added to the graph will have 0 as the generation attribute. All the following people node will have generation attribute by the relative relationship with that first people node. However, there were several issues: first people in the node might not actually have family, while rest of the people are family; it puts aunt/uncle and parent in a same generation, which requires additional job; there could be more than one family in the chat; a distinct family can later be merged. While I try to manually hardcode these edge cases, it was difficult to do so. 
- Lastly, I tried to use chatbot generate family tree by reformatting the chatbot genereated family tree when user asks "can you give me a family tree of [people name]." However, in order to properly reformat, it needs to have a background information, which could be given through complex prompt. So, for this project, at this point, I ended up creating a non-hierarchical family tree, more of a relationship graph. 


### Remaining issues and future plan
1. efficient size of prompt
Ideal size of the prompt was a continuous issue. While I do not wanted to hardcode all the possible family relationships and attributes, I did not want to write long prompts as well due to the computational cost and speed. My initial attempt when generating response was whenever user asks question, it uses NER to find the relevant node and then look at the edges. By gathering relevant node and edges, it creates a context and then the chat tries to generate response from that. That apparently did not work because the accuracy was so low and the computation was also low. Balancing prompt and using the graph's structure itself is still the remaining issue.

2. random shutdown issue
When the session is opened for more than 7 to 8 minutes, the program slightly breaks (it looks like pixel is brken at the edge of the chat ui) and performance decreases. While it could be a problem with my laptop or some other code construction issue, I am planning to further investigate.

3. placeholder issue
While it efficiently works, sometimes the place holder does not work when we use it along with the conflict detection. Since this occurs randomly, I would like to further delve into this issue. 

4. visualising nodes and edges that are used for chat response
This idea stemmed from arxiv paper "Agentigraph: An interactive knowledge graph platform for llm-based chatbots utilizing private data." While using graph itself is helpful for the generating response, creating interactive visualisation graph provided more information and credibility for the user. Based on that idea, I thought that showing edges and nodes that are used to generate chat response along with the response could add credibility to the answer and also make user more engaged. 

5. chat-inside-chat
I plan to create an option that user can directly chat with the people that was mentioned and built by the user. For instance, if the user mentioned "Kim is married to Jim" and "Kim likes chocolate," the user can have a chat with Kim who is married to Jim and likes chocoalte. While creating that side chat, the information generated through that chat will be also updated to the knowledge graph. 

6. creating a hierarchical family tree
While all the attempts that I tried fail, I still want to solve this issue. What I have in mind is training the model by giving example hierarchical family tree and corresponding family relations. Then, I will try to apply that separate program onto this project. 

 
