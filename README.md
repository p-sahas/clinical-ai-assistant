# Clinical AI Assistant

A comprehensive AI-powered clinical assistant that demonstrates advanced AI engineering patterns including Reflection, Tool Use, Planning, and Multi-Agent systems. Built with Retrieval-Augmented Generation (RAG) for medical knowledge, integrated with booking management and web search capabilities.

## Features

### AI Engineering Patterns

- **Reflection**: Self-critique and iterative improvement (Draft → Critique → Revise)
- **Tool Use**: External tool integration for real-world actions
- **Planning & ReAct**: Goal-directed reasoning and action planning
- **Multi-Agent Systems**: Collaborative agent architectures

### Clinical Capabilities

- **Medical Knowledge RAG**: Internal retrieval system for healthcare documents (PDF, TXT, DOCX)
- **Appointment Booking**: Patient booking lookup by name, phone, or email
- **Web Search**: Real-time healthcare information retrieval using Tavily API
- **Medical Guides**: Pre-loaded knowledge base for common conditions and health information

### Technical Features

- **Multi-Provider LLM Support**: OpenRouter, OpenAI, Anthropic, Google, Groq, DeepSeek
- **Vector Storage**: Qdrant-based vector database for efficient retrieval
- **Configurable Embeddings**: Multiple embedding models with batch processing
- **Modular Architecture**: Clean separation of concerns with tools, RAG, and utilities

## Installation

### Prerequisites

- Python 3.8+
- API keys for your chosen LLM provider (OpenRouter recommended for unified access)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/p-sahas/clinical-ai-assistant.git
cd clinical-ai-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file in the root directory:

```env
# ========================================
# Required Secrets
# ========================================

# Required: Supabase credentials
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here  # Optional, for admin operations
SUPABASE_DB_URL=your_supabase_db_url_here

# Required: Choose your LLM provider
OPENROUTER_API_KEY=your_openrouter_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# ========================================
# Optional Provider Secrets
# ========================================
# For web search functionality
TAVILY_API_KEY=your_tavily_api_key_here

GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

4. Configure your settings:
   Edit `config/params.yaml` to customize:

- LLM provider and model tiers
- Embedding settings
- Retrieval parameters
- Web search preferences

## Usage

### Running Scratch Demos

Explore individual AI patterns through Jupyter notebooks:

```bash
jupyter notebook scratch_demos/
```

Available demos:

- `01_reflection_scratch.ipynb`: Self-critique and iterative improvement
- `02_tool_use_scratch.ipynb`: External tool integration
- `03_planning_react_scratch.ipynb`: Goal-directed reasoning
- `04_multi_agent_scratch.ipynb`: Multi-agent collaboration

### Running LangGraph Demos

Explore agent orchestration using LangGraph through Jupyter notebooks:

```bash
jupyter notebook langgraph_demos/
```

Available demos:

- `00_langgraph_intro.ipynb`: Introduction to LangGraph fundamentals
- `01_reflection_langgraph.ipynb`: Self-critique and iterative improvement with LangGraph
- `02_tool_use_langgraph.ipynb`: External tool integration with LangGraph
- `03_planning_react_langgraph.ipynb`: Goal-directed reasoning with LangGraph
- `04_multi_agent_langgraph.ipynb`: Multi-agent collaboration with LangGraph

### Basic Usage Example

```python
from utils import create_llm_provider, get_config
from rag_internal.retriever import HealthcareRAG
from tools.booking import BookingLookupTool
from tools.web_search import WebSearchTool

# Initialize components
config = get_config()
llm = create_llm_provider()
rag = HealthcareRAG()
booking_tool = BookingLookupTool()
web_tool = WebSearchTool()

# Example: Medical knowledge retrieval
results = rag.search("common symptoms of diabetes")
print(results)

# Example: Patient booking lookup
bookings = booking_tool.lookup(patient_name="John Doe")
print(bookings)

# Example: Web search for healthcare info
search_results = web_tool.search("hospital emergency hours")
print(search_results)
```

## Project Structure

```
clinical-ai-assistant/
├── config/                 # Configuration files
│   ├── models.yaml        # LLM model configurations
│   └── params.yaml        # System parameters
├── data/                  # Data files
│   ├── bookings.json      # Patient booking data
│   └── medical_guides/    # Medical knowledge documents
├── rag_internal/          # Internal RAG system
│   ├── retriever.py       # Healthcare document retrieval
│   └── __init__.py
├── langgraph_demos/       # LangGraph orchestration notebooks
│   ├── 00_langgraph_intro.ipynb
│   ├── 01_reflection_langgraph.ipynb
│   ├── 02_tool_use_langgraph.ipynb
│   ├── 03_planning_react_langgraph.ipynb
│   └── 04_multi_agent_langgraph.ipynb
├── scratch_demos/         # Educational notebooks
│   ├── 01_reflection_scratch.ipynb
│   ├── 02_tool_use_scratch.ipynb
│   ├── 03_planning_react_scratch.ipynb
│   └── 04_multi_agent_scratch.ipynb
├── store/                 # Vector database storage
├── tools/                 # External tools
│   ├── booking.py         # Appointment lookup
│   ├── web_search.py      # Web search functionality
│   └── __init__.py
├── utils/                 # Utility modules
│   ├── config.py          # Configuration management
│   ├── llm_services.py    # LLM provider abstraction
│   └── __init__.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

### LLM Providers

The system supports multiple LLM providers. Configure in `config/params.yaml`:

```yaml
provider:
  default: openrouter # openrouter | openai | anthropic | google | groq | deepseek
  tier: general # general | strong | reason
```

### Embedding Models

Choose from various embedding models for document retrieval:

```yaml
embedding:
  tier: small # default | small
```

### Retrieval Settings

Customize RAG behavior:

```yaml
retrieval:
  top_k: 4
  similarity_threshold: 0.7
```

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest
```

### Adding New Tools

1. Create a new tool class in `tools/`
2. Implement the required interface
3. Register the tool in the multi-agent system

### Extending Medical Knowledge

Add new documents to `data/medical_guides/` and rebuild the vector store:

```python
from rag_internal.retriever import HealthcareRAG
rag = HealthcareRAG()
rag.build_index()  # Rebuild from documents
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is an educational demonstration of AI engineering patterns. The medical information provided is for illustrative purposes only and should not be used for actual medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## Acknowledgments

- Built with [OpenAI](https://openai.com/), [Qdrant](https://qdrant.tech/), and [Tavily](https://tavily.com/)
- Inspired by modern AI engineering practices and healthcare technology trends
- Educational content based on general medical knowledge resources

