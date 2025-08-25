# Collaborative AI Storytelling System

A sophisticated multi-agent storytelling system that uses specialized AI agents to collaboratively generate high-quality fantasy stories. The system employs a controlled workflow with configurable limits to prevent runaway costs while ensuring coherent, engaging narratives.

## 🌟 Features

### Multi-Agent Collaboration
- **Writer Agent**: Generates engaging narrative scenes with rich descriptions
- **Editor Agent**: Polishes prose, improves pacing, and ensures stylistic consistency
- **Continuity Checker**: Maintains story consistency and identifies plot contradictions
- **Illustrator Agent**: Creates detailed visual descriptions for artwork generation


### Advanced Workflow Control
- **Controlled Loop System**: Explicit loops instead of recursion to prevent infinite cycles
- **Configurable LLM Limits**: Hard caps on API calls per scene to control costs
- **Circuit Breaker Pattern**: Graceful degradation when limits are reached
- **Anti-Repetition System**: Prevents duplicate content and ensures story progression
- **Content Deduplication**: Real-time detection of repeated sentences and paragraphs
- **Story Arc Tracking**: Ensures proper narrative progression across scenes
- **Comprehensive Logging**: Detailed diagnostics for debugging and monitoring

### Flexible Configuration
- **Environment-Based Settings**: Customize behavior via environment variables
- **YAML-Based Lore Database**: Externalized world-building knowledge
- **Multiple Output Formats**: Story text and art prompts for visualization
- **Fallback Mechanisms**: Robust error handling ensures story completion

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install langchain langchain-openai crewai langgraph python-dotenv pyyaml
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional Configuration for ANY OpenAI compatible model/api. I use Inception for it's incredible speed. Here is an example
OPENAI_BASE_URL=https://api.inceptionlabs.ai/v1  # Custom endpoint
OPENAI_MODEL=mercury-coder                       # Model to use
SCENE_COUNT=3                                    # Number of scenes to generate
MAX_CONTINUITY_ATTEMPTS=3                        # Max continuity fix attempts
MAX_LLM_CALLS_PER_SCENE=8                        # Hard limit on API calls per scene
CONTENT_SIMILARITY_THRESHOLD=0.4                 # Similarity threshold for duplicate detection
LORE_FILE_PATH=lore_database.yaml                # Path to lore database

```

### Basic Usage

```bash
python story.py
```

The system will:
1. Generate 3 fantasy story scenes by default
2. Create detailed art prompts for each scene
3. Maintain story continuity across scenes
4. Prevent repetitive content and ensure story progression
5. Log all activities to both console and file
6. Respect LLM call limits to control costs

## 📊 System Architecture

### Workflow Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Writer    │ -> │   Editor    │ -> │ Continuity  │ -> │Illustrator  │ -> │   Story     │
│   Agent     │    │   Agent     │    │  Checker    │    │   Agent     │    │Integration  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ↓
                                      ┌─────────────┐
                                      │ Regenerate? │
                                      │ (Max 2x)    │
                                      └─────────────┘
```

### LLM Call Budget Per Scene

| Step | Calls | Purpose |
|------|-------|---------|
| Scene Generation | 2 | Content + metadata extraction |
| Editing | 1 | Polish and improve prose |
| Continuity Check | 1-4 | Analyze consistency + repetition (max 2 attempts) |
| Art Prompt | 1 | Generate visual descriptions |
| **Total** | **5-8** | **Hard limit enforced** |

### Anti-Repetition System

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Scene        │ -> │Duplicate    │ -> │Continuity   │ -> │Story Arc    │
│Generation   │    │Detection    │    │Check        │    │Progression  │
│(Stage-aware)│    │(Real-time)  │    │(Enhanced)   │    │(Tracking)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Features:**
- **Content Deduplication**: Detects exact sentence/paragraph repetition
- **Story Arc Tracking**: Ensures proper narrative progression (opening → development → climax → resolution)
- **Character Continuity**: Prevents re-introducing the same characters as if new
- **Enhanced Continuity Checking**: Specifically looks for repetitive events, dialogue, and descriptions
- **Stage-Aware Prompts**: Different generation guidance for each story phase

### State Management

The system maintains a comprehensive story state:

```python
{
    "story_so_far": str,           # Accumulated narrative
    "characters": dict,            # Character profiles
    "settings": dict,              # World-building details
    "current_scene": dict,         # Active scene data
    "art_prompts": list,           # Visual descriptions
    "genre": str,                  # Story genre
    "tone": str,                   # Narrative tone
    "pov": str,                    # Point of view
    "continuity_issues": list      # Detected inconsistencies
}
```

## ⚙️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_BASE_URL` | `None` | Custom OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `gpt-4` | Language model to use |
| `SCENE_COUNT` | `3` | Number of scenes to generate |
| `MAX_CONTINUITY_ATTEMPTS` | `3` | Maximum regeneration attempts |
| `MAX_LLM_CALLS_PER_SCENE` | `8` | Hard limit on API calls per scene |
| `CONTENT_SIMILARITY_THRESHOLD` | `0.4` | Similarity threshold for duplicate detection (0.0-1.0) |
| `LORE_FILE_PATH` | `lore_database.yaml` | Path to lore database |

### Anti-Repetition Configuration

| Setting | Range | Effect |
|---------|-------|--------|
| `CONTENT_SIMILARITY_THRESHOLD=0.2` | Lower | Stricter duplicate detection |
| `CONTENT_SIMILARITY_THRESHOLD=0.8` | Higher | More lenient similarity checking |
| `MAX_CONTINUITY_ATTEMPTS=1` | Lower | Less repetition checking, faster generation |
| `MAX_CONTINUITY_ATTEMPTS=5` | Higher | Even more thorough repetition prevention |

**Recommended Settings:**
- **Default (Balanced)**: `CONTENT_SIMILARITY_THRESHOLD=0.4`, `MAX_CONTINUITY_ATTEMPTS=3`
- **High Quality**: `CONTENT_SIMILARITY_THRESHOLD=0.2`, `MAX_CONTINUITY_ATTEMPTS=5`
- **Fast Generation**: `CONTENT_SIMILARITY_THRESHOLD=0.8`, `MAX_CONTINUITY_ATTEMPTS=1`

### Lore Database (`lore_database.yaml`)

Customize the world-building knowledge:

```yaml
lore:
  fantasy: "Fantasy stories often involve magic, mythical creatures, and medieval settings."
  knight: "Knights are warriors who follow a code of chivalry."
  dragon: "Dragons are powerful, fire-breathing creatures that hoard treasure."
  magic: "Magic in fantasy often follows specific rules and systems."
  # Add your own lore entries...
```

## 📝 Output Examples

### Story Generation

The system generates complete fantasy narratives with rich descriptions:

```
The moon hung low in the sky, its pale light casting eerie shadows through 
the dense canopy of the Dark Forest. The air was thick with the scent of 
decay and the distant hum of unseen creatures...
```

### Art Prompts

Each scene includes detailed visual descriptions:

```
A dark forest scene with moonlight filtering through ancient trees, 
mysterious fog at ground level, a cloaked figure holding an enchanted 
lantern, dramatic lighting with deep shadows and ethereal glow...
```

## 🔍 Monitoring and Debugging

### Comprehensive Logging

The system creates detailed log files with timestamps:

```
2025-08-24 17:19:03,246 - INFO - Starting LLM call #1 for scene generation
2025-08-24 17:19:09,127 - INFO - LLM call #1 completed in 5.88s, response length: 1247 chars
2025-08-24 17:19:10,537 - INFO - Starting LLM call #2 for metadata extraction
```

### Performance Metrics

Each scene completion shows:
- Total execution time
- Number of LLM calls used
- Continuity check results
- Duplicate content warnings
- Story progression stage
- Cost efficiency warnings

```
Scene 1 completed successfully (used 5 LLM calls)
2025-08-24 17:19:03,677 - WARNING - Found 2 duplicate sentences in generated scene
Scene 2 completed successfully (used 7 LLM calls)
2025-08-24 17:19:13,774 - INFO - Stage: DEVELOPMENT - Develop the conflict
Scene 3 completed successfully (used 4 LLM calls)
```

## 🛡️ Cost Control Features

### Circuit Breaker Pattern

The system implements multiple safety mechanisms:

1. **Hard Limits**: Absolute maximum on LLM calls per scene
2. **Smart Allocation**: Reserves calls for essential steps
3. **Graceful Degradation**: Uses fallbacks when limits are hit
4. **Early Termination**: Stops processing if budget exceeded

### Fallback Mechanisms

When limits are reached, the system:
- Uses pre-written fallback content
- Skips optional processing steps
- Provides coherent story completion
- Logs cost-saving decisions

## 🔧 Development

### Project Structure

```
StoryAgents/
├── story.py                 # Main application
├── lore_database.yaml      # World-building knowledge
├── .env                    # Environment configuration
├── README.md              # This file
└── logs/                  # Generated log files
    └── story_generation_*.log
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `generate_scene_with_limits()` | Main controlled scene generation |
| `invoke_llm_safely()` | LLM calls with logging and limits |
| `check_exact_duplicates()` | Detect repeated sentences/paragraphs |
| `check_content_similarity()` | Measure word-based content similarity |
| `get_story_progression_stage()` | Determine current story phase |
| `build_*_prompt()` | Specialized prompt builders |
| `parse_continuity_report()` | Continuity analysis parsing |

### Story Progression Stages

The system automatically determines story phase based on scene number:

| Stage | Scenes | Purpose | Prompts Focus On |
|-------|--------|---------|-------------------|
| **Opening** | Scene 1 | Introduce protagonist, setting, conflict | World establishment, character introduction |
| **Development** | Scenes 2-60% | Develop conflict, introduce challenges | NEW encounters, plot advancement |
| **Climax Build** | Scenes 60-80% | Escalate tensions, approach resolution | Tension escalation, final challenges |
| **Resolution** | Final scenes | Resolve conflict, provide closure | Conflict resolution, character growth |

### Adding New Agents

To add a new specialized agent:

1. Create agent using `create_agent()` function
2. Define tools if needed (inherit from `BaseTool`)
3. Add corresponding workflow node function
4. Update the controlled generation loop
5. Add appropriate logging and limits

## 🐛 Troubleshooting

### Common Issues

**Problem**: Too many LLM calls
**Solution**: Adjust `MAX_LLM_CALLS_PER_SCENE` environment variable to a lower value.

**Problem**: Poor story quality
**Solution**: Increase `MAX_CONTINUITY_ATTEMPTS` or improve the lore database.

**Problem**: Repetitive content
**Solution**: Lower `CONTENT_SIMILARITY_THRESHOLD` to 0.2 for even stricter duplicate detection, or increase `MAX_CONTINUITY_ATTEMPTS` to 5.

**Problem**: Generic protagonist
**Solution**: Set `PROTAGONIST_ARCHETYPE=cursed_wanderer` or another specific archetype instead of `random`.

**Problem**: Predictable stories
**Solution**: Increase `TWIST_PROBABILITY=0.8` for more plot twists, or try different archetype combinations.

**Problem**: Twist reveals too late
**Solution**: Set `TWIST_REVEAL_SCENE=2` for earlier reveals in 3-scene stories.

### Debug Mode

For detailed debugging, check the generated log files:

```bash
# View latest log file
ls -la story_generation_*.log | tail -1
tail -f story_generation_$(date +%Y%m%d)_*.log
```

## 🤝 Contributing

### Code Quality Standards

The project follows these principles:
- **Single Responsibility**: Each function has one clear purpose
- **Error Handling**: Comprehensive fallback mechanisms
- **Documentation**: Detailed comments and docstrings
- **Logging**: Extensive diagnostic information
- **Configuration**: Environment-based settings

### Security Considerations

- API keys are not logged or printed
- All sensitive data is handled via environment variables
- Input validation prevents injection attacks
- Error messages don't expose internal details

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - LLM framework
- [CrewAI](https://crewai.io/) - Multi-agent orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Workflow management
- [OpenAI API](https://openai.com/) - Language model API

