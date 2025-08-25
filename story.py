import os
import time
import json
import yaml
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from crewai import Agent, Task, Crew, Process
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from typing import Dict, List, TypedDict, Annotated
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Set up environment
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "gpt-4")
scene_count = int(os.getenv("SCENE_COUNT", "3"))
lore_file_path = os.getenv("LORE_FILE_PATH", "lore_database.yaml")
max_continuity_attempts = int(os.getenv("MAX_CONTINUITY_ATTEMPTS", "3"))
max_llm_calls_per_scene = int(os.getenv("MAX_LLM_CALLS_PER_SCENE", "8"))
content_similarity_threshold = float(os.getenv("CONTENT_SIMILARITY_THRESHOLD", "0.4"))

# Verify environment variables are loaded
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
print(f"Model: {model_name}")
print("Environment variables loaded successfully")

# Logging Configuration
# ====================
# Set up comprehensive logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(f'story_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info("=== Story Generation Session Started ===")
logger.info(f"Model: {model_name}")
logger.info(f"Scene count: {scene_count}")
logger.info(f"Lore file: {lore_file_path}")
logger.info(f"Max continuity attempts: {max_continuity_attempts}")
logger.info(f"Max LLM calls per scene: {max_llm_calls_per_scene}")
logger.info(f"Content similarity threshold: {content_similarity_threshold}")

# Initialize LLM with configurable base URL and model
def create_llm():
    """Create a configured LLM instance with retry logic"""
    llm_config = {
        "model": model_name,
        "openai_api_key": api_key,
        "temperature": 0.7,
        "max_retries": 3,
        "request_timeout": 60,
    }
    
    if base_url:
        llm_config["openai_api_base"] = base_url
    
    return ChatOpenAI(**llm_config)

# Define the state structure
class StoryState(TypedDict):
    messages: Annotated[List, add_messages]
    story_so_far: str
    characters: Dict
    settings: Dict
    plot_points: List
    current_scene: Dict
    art_prompts: List
    genre: str
    tone: str
    pov: str
    continuity_issues: List

# Define functions that will be used as tools
def retrieve_lore(query: str) -> str:
    """Retrieve relevant lore information using keyword matching."""
    def _retrieve():
        query_lower = query.lower()
        relevant_info = []
        
        # Simple keyword matching
        for keyword, info in lore_database.items():
            if keyword in query_lower:
                relevant_info.append(info)
        
        # Add some general fantasy knowledge if no specific matches
        if not relevant_info:
            relevant_info.append("Fantasy stories typically involve heroes, magical elements, and mythical creatures.")
        
        return "\n".join(relevant_info)
    
    return safe_execute(_retrieve, "retrieve_lore", "Fantasy stories typically involve heroes, magical elements, and mythical creatures.")

def check_continuity(character_name: str, attribute: str, current_value: str) -> str:
    """Check if a character attribute is consistent with established facts."""
    def _check():
        # Simple in-memory storage for character details
        known_facts = {
            "Sir Galen": {
                "weapon": "broadsword",
                "armor": "silver plate mail",
                "companion": "faithful hawk"
            }
        }
        
        if character_name in known_facts and attribute in known_facts[character_name]:
            if known_facts[character_name][attribute] != current_value:
                return f"Inconsistency found: {character_name}'s {attribute} was previously {known_facts[character_name][attribute]}, but is now {current_value}"
        
        return "No continuity issues found"
    
    return safe_execute(_check, "check_continuity", "No continuity issues found")


# For CrewAI, we need to create tools using their expected format
class RetrieveLoreTool(BaseTool):
    name: str = "Retrieve Lore"
    description: str = "Retrieve relevant lore information using keyword matching."
    
    def _run(self, query: str) -> str:
        return safe_execute(
            lambda: retrieve_lore(query),
            "RetrieveLoreTool",
            "Fantasy stories typically involve heroes, magical elements, and mythical creatures."
        )

class CheckContinuityTool(BaseTool):
    name: str = "Check Continuity"
    description: str = "Check if a character attribute is consistent with established facts."
    
    def _run(self, character_name: str, attribute: str, current_value: str) -> str:
        return safe_execute(
            lambda: check_continuity(character_name, attribute, current_value),
            "CheckContinuityTool",
            "No continuity issues found"
        )


# Create the LLM instance
llm = create_llm()

# Load lore database from YAML file
def load_lore_database(file_path: str) -> dict:
    """Load lore database from YAML file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data.get('lore', {})
    except FileNotFoundError:
        print(f"Warning: Lore file {file_path} not found. Using minimal fallback data.")
        return {"fantasy": "Fantasy stories often involve magic, mythical creatures, and medieval settings."}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {file_path}: {e}. Using minimal fallback data.")
        return {"fantasy": "Fantasy stories often involve magic, mythical creatures, and medieval settings."}
    except Exception as e:
        print(f"Warning: Unexpected error loading lore database: {e}. Using minimal fallback data.")
        return {"fantasy": "Fantasy stories often involve magic, mythical creatures, and medieval settings."}

lore_database = load_lore_database(lore_file_path)

# Utility functions to reduce code duplication
def safe_execute(func, error_context: str, fallback_result=None):
    """Execute a function safely with standardized error handling"""
    try:
        return func()
    except Exception as e:
        print(f"Error in {error_context}: {e}")
        return fallback_result

def invoke_llm_safely(llm, prompt: str, context: str) -> str:
    """Safely invoke LLM with standardized error handling"""
    def _invoke():
        global llm_call_count
        llm_call_count += 1
        
        logger.info(f"Starting LLM call #{llm_call_count} for {context}")
        start_time = time.time()
        
        # Log prompt details (truncated for readability)
        if isinstance(prompt, str):
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            logger.debug(f"Prompt for {context}: {prompt_preview}")
            response = llm.invoke(prompt)
        else:
            logger.debug(f"Message list for {context}: {len(prompt)} messages")
            response = llm.invoke(prompt)
        
        elapsed_time = time.time() - start_time
        
        # Log response details
        content = response.content
        logger.info(f"LLM call #{llm_call_count} for {context} completed in {elapsed_time:.2f}s, response length: {len(content)} chars")
        logger.debug(f"Response preview for {context}: {content[:200]}...")
        
        return content
    
    return safe_execute(_invoke, f"LLM invocation in {context}", "LLM invocation failed")

# Global LLM call counter
llm_call_count = 0

def reset_llm_counter():
    """Reset LLM call counter for new scene"""
    global llm_call_count
    llm_call_count = 0
    logger.info("LLM call counter reset")

def get_llm_call_count():
    """Get current LLM call count"""
    return llm_call_count

def check_content_similarity(new_content: str, existing_story: str, threshold: float = 0.7) -> bool:
    """Check if new content is too similar to existing story content.
    
    Args:
        new_content (str): New scene content to check
        existing_story (str): Existing story content  
        threshold (float): Similarity threshold (0.0-1.0)
        
    Returns:
        bool: True if content is too similar (should be rejected)
    """
    if not existing_story.strip():
        return False  # First scene, no comparison needed
    
    # Simple word-based similarity check
    new_words = set(new_content.lower().split())
    existing_words = set(existing_story.lower().split())
    
    if not new_words or not existing_words:
        return False
    
    intersection = new_words.intersection(existing_words)
    similarity = len(intersection) / len(new_words.union(existing_words))
    
    logger.debug(f"Content similarity: {similarity:.3f} (threshold: {threshold})")
    return similarity > threshold

def check_exact_duplicates(new_content: str, existing_story: str) -> list:
    """Find exact duplicate sentences/paragraphs in new content.
    
    Args:
        new_content (str): New scene content
        existing_story (str): Existing story content
        
    Returns:
        list: List of duplicate sentences found
    """
    if not existing_story.strip():
        return []
    
    # Split into sentences (simple approach)
    new_sentences = [s.strip() for s in new_content.split('.') if s.strip()]
    existing_sentences = [s.strip() for s in existing_story.split('.') if s.strip()]
    
    duplicates = []
    for new_sent in new_sentences:
        if len(new_sent) > 20:  # Only check substantial sentences
            for existing_sent in existing_sentences:
                if new_sent.lower() == existing_sent.lower():
                    duplicates.append(new_sent)
                    break
    
    return duplicates

def generate_scene_with_limits(state: StoryState) -> dict:
    """Generate a scene with controlled LLM usage and explicit loop instead of recursion.
    
    This replaces the recursive LangGraph workflow with an explicit loop that has
    configurable limits to prevent runaway LLM costs.
    
    Args:
        state (StoryState): Current story state
        
    Returns:
        dict: Updated state with new scene
    """
    logger.info("=== CONTROLLED SCENE GENERATION STARTED ===")
    scene_start_calls = get_llm_call_count()
    
    # Step 1: Generate initial scene
    logger.info("Step 1: Generating initial scene content...")
    scene_number = len(state.get("art_prompts", [])) + 1  # Determine scene number
    scene_text = generate_scene_content(state, llm, scene_number)
    
    # Check for content duplication
    duplicates = check_exact_duplicates(scene_text, state['story_so_far'])
    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate sentences in generated scene")
        for dup in duplicates[:3]:  # Show first 3 duplicates
            logger.warning(f"Duplicate: {dup[:100]}...")
    
    scene_metadata = extract_scene_metadata(scene_text, llm)
    
    current_scene = {
        "text": scene_text,
        "metadata": scene_metadata
    }
    
    # Step 2: Edit the scene
    logger.info("Step 2: Editing scene...")
    prompt = build_editing_prompt(scene_text, state['tone'], state['pov'])
    edited_scene = invoke_llm_safely(llm, prompt, "controlled_editor")
    
    current_scene["text"] = edited_scene
    
    # Step 3: Continuity checking with explicit loop
    logger.info("Step 3: Checking continuity with loop control...")
    continuity_attempts = 0
    
    while continuity_attempts < max_continuity_attempts:
        # Check if we've hit the LLM call limit
        current_calls = get_llm_call_count()
        calls_used_this_scene = current_calls - scene_start_calls
        
        if calls_used_this_scene >= max_llm_calls_per_scene:
            logger.warning(f"Hit LLM call limit ({max_llm_calls_per_scene}) for this scene, skipping further continuity checks")
            break
        
        logger.info(f"Continuity check attempt {continuity_attempts + 1}/{max_continuity_attempts}")
        
        # Build continuity prompt and check
        prompt = build_continuity_prompt(state, current_scene["text"])
        continuity_report = invoke_llm_safely(llm, prompt, f"controlled_continuity_check_{continuity_attempts + 1}")
        issues = parse_continuity_report(continuity_report)
        
        if not issues:
            logger.info("No continuity issues found, proceeding to illustration")
            break
        
        logger.warning(f"Found {len(issues)} continuity issues: {issues}")
        continuity_attempts += 1
        
        if continuity_attempts < max_continuity_attempts:
            # Check LLM limit again before regenerating
            current_calls = get_llm_call_count()
            calls_used_this_scene = current_calls - scene_start_calls
            
            if calls_used_this_scene >= max_llm_calls_per_scene - 1:  # Reserve 1 call for illustration
                logger.warning("Near LLM call limit, accepting scene with continuity issues")
                break
            
            logger.info("Regenerating scene to fix continuity/repetition issues...")
            scene_text = generate_scene_content(state, llm, scene_number)
            
            # Check if regeneration helped with duplicates
            new_duplicates = check_exact_duplicates(scene_text, state['story_so_far'])
            if new_duplicates:
                logger.warning(f"Regenerated scene still has {len(new_duplicates)} duplicates")
            
            current_scene["text"] = scene_text
            # Skip re-editing and metadata extraction for regenerated scenes to save LLM calls
        else:
            logger.warning("Max continuity attempts reached, accepting scene as-is")
    
    # Step 4: Generate art prompt
    logger.info("Step 4: Generating art prompt...")
    current_calls = get_llm_call_count()
    calls_used_this_scene = current_calls - scene_start_calls
    
    if calls_used_this_scene < max_llm_calls_per_scene:
        art_prompt_text = build_art_prompt(current_scene["text"], state['tone'])
        art_prompt = invoke_llm_safely(llm, art_prompt_text, "controlled_art_prompt")
    else:
        logger.warning("Hit LLM call limit, using fallback art prompt")
        art_prompt = "A fantasy scene with dramatic lighting"
    
    # Update scene metadata with art prompt
    current_scene["metadata"]["art_prompt"] = art_prompt
    
    # Step 5: Update story
    logger.info("Step 5: Updating story...")
    new_story = state["story_so_far"] + "\n\n" + current_scene["text"]
    
    final_calls = get_llm_call_count()
    total_calls_used = final_calls - scene_start_calls
    logger.info(f"=== CONTROLLED SCENE GENERATION COMPLETED ===")
    logger.info(f"Total LLM calls used for this scene: {total_calls_used}")
    
    # Return updated state
    return {
        **state,
        "current_scene": current_scene,
        "story_so_far": new_story,
        "art_prompts": state.get("art_prompts", []) + [art_prompt],
        "messages": state.get("messages", []) + [HumanMessage(content=f"Added scene: {current_scene['text']}")]
    }

def create_fallback_scene(scene_type: str = "default") -> dict:
    """Generate standardized fallback scene data"""
    fallback_scenes = {
        "default": {
            "text": """The moon hung low in the sky, its pale light casting eerie shadows through the dense canopy of the Dark Forest. 
The air was thick with the scent of decay and the distant hum of unseen creatures. I had been wandering these 
woods for hours, my path illuminated only by the flickering glow of my enchanted lantern. The forest seemed to 
whisper secrets to me, each step a reminder of the dangers that lurked within.""",
            "metadata": {
                "characters_involved": ["protagonist"],
                "setting": "Dark Forest",
                "key_events": ["entering the forest"],
                "art_prompt": "A dark forest with moonlight filtering through trees"
            }
        },
        "editing_failed": {
            "text": "Scene editing failed.",
            "metadata": {}
        },
        "illustration_failed": {
            "text": "Scene illustration failed.",
            "metadata": {"art_prompt": "A fantasy scene with dramatic lighting"}
        }
    }
    return fallback_scenes.get(scene_type, fallback_scenes["default"])

def create_fallback_metadata() -> dict:
    """Generate standardized fallback metadata for scene extraction failures.
    
    When JSON parsing or metadata extraction fails, this provides a consistent
    default structure that won't break downstream processing.
    
    Returns:
        dict: Default scene metadata structure
    """
    return {
        "characters_involved": [],
        "setting": "unknown",
        "key_events": [],
        "art_prompt": "A fantasy scene with dramatic lighting"
    }

# Specialized Helper Functions for Scene Processing
# ================================================
# These functions handle specific aspects of scene generation and processing

def build_scene_prompt(state: StoryState, lore: str) -> str:
    """Build the comprehensive prompt for scene generation.
    
    Args:
        state (StoryState): Current story state with context
        lore (str): Relevant lore information
        
    Returns:
        str: Complete prompt for scene generation
    """
    characters_text = ', '.join(state['characters'].keys()) if state['characters'] else 'Introduce new characters if needed'
    
    return f"""
    You are writing a {state['genre']} story with a {state['tone']} tone, told from {state['pov']} perspective.
    
    Story so far:
    {state['story_so_far']}
    
    Relevant lore:
    {lore}
    
    Write the next scene. Include:
    1. Engaging narrative that moves the plot forward
    2. Vivid descriptions of settings and characters
    3. Meaningful dialogue if appropriate
    4. A hook that makes the reader want to continue
    
    Focus on these characters: {characters_text}
    
    Write in a {state['tone']} tone.
    """

def build_metadata_extraction_prompt() -> str:
    """Build the prompt for extracting structured metadata from scenes.
    
    Returns:
        str: JSON extraction prompt
    """
    return """
    Extract the following information from the scene as JSON:
    {
      "characters_involved": [],
      "setting": "",
      "key_events": [],
      "art_prompt": "A detailed description for an illustrator"
    }
    """

def extract_scene_metadata(scene_text: str, llm) -> dict:
    """Extract structured metadata from generated scene text.
    
    Args:
        scene_text (str): The generated scene content
        llm: Language model instance
        
    Returns:
        dict: Extracted metadata or fallback structure
    """
    extract_prompt = build_metadata_extraction_prompt()
    
    extraction_result = invoke_llm_safely(
        llm, 
        [HumanMessage(content=f"Scene: {scene_text}\n\n{extract_prompt}")],
        "metadata extraction"
    )
    
    
    def _parse_json():
        # Try to extract JSON from the response using regex
        import re
        
        # First try to parse the whole response
        try:
            return json.loads(extraction_result.strip())
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to find JSON within the response
        json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, raise an exception to trigger fallback
        raise json.JSONDecodeError("No valid JSON found", extraction_result, 0)

    def _parse_json2():
        return json.loads(extraction_result)
    
    return safe_execute(_parse_json, "JSON parsing", create_fallback_metadata())

def get_story_progression_stage(scene_number: int, total_scenes: int) -> str:
    """Determine what stage of story progression we're in."""
    if scene_number == 1:
        return "opening"
    elif scene_number <= total_scenes * 0.6:
        return "development"
    elif scene_number <= total_scenes * 0.8:
        return "climax_build"
    else:
        return "resolution"

def generate_scene_content(state: StoryState, llm, scene_number: int = 1) -> str:
    """Generate the main scene content using LLM with anti-repetition logic.
    
    Args:
        state (StoryState): Current story state
        llm: Language model instance
        scene_number (int): Current scene number
        
    Returns:
        str: Generated scene text
    """
    # Determine story progression stage
    stage = get_story_progression_stage(scene_number, scene_count)
    
    # Stage-specific guidance
    stage_guidance = {
        "opening": "Introduce the protagonist, setting, and initial conflict.",
        "development": "Develop the conflict, introduce NEW challenges. Avoid repeating previous encounters.",
        "climax_build": "Escalate tensions, approach main conflict resolution.",
        "resolution": "Resolve the main conflict, provide closure."
    }
    
    # Anti-repetition instructions
    repetition_warning = ""
    if state['story_so_far'].strip():
        repetition_warning = """
        CRITICAL: Do NOT repeat previous events, dialogue, or descriptions. 
        Move the story forward with NEW events and NEW locations.
        If you mention the guardian, woman, or armored man - they should NOT repeat previous dialogue.
        """
    
    # Build anti-repetition prompt
    if not state['story_so_far'].strip():
        prompt = f"Write the opening scene of a {state['genre']} story with a {state['tone']} tone in {state['pov']} perspective."
    else:
        prompt = f"""
        Continue this {state['genre']} story with a {state['tone']} tone in {state['pov']} perspective.
        
        Previous story excerpt:
        ...{state['story_so_far'][-400:]}
        
        Stage: {stage.upper()} - {stage_guidance[stage]}
        
        {repetition_warning}
        
        Write the next scene that introduces NEW story elements and progresses the plot.
        """
    
    return invoke_llm_safely(llm, prompt, f"scene_gen_{stage}_{scene_number}")

def build_editing_prompt(scene_text: str, tone: str, pov: str) -> str:
    """Build prompt for editing and polishing scene content.
    
    Args:
        scene_text (str): Raw scene text to edit
        tone (str): Story tone for consistency
        pov (str): Point of view for consistency
        
    Returns:
        str: Complete editing prompt
    """
    return f"""
    Edit the following scene to improve:
    1. Grammar and spelling
    2. Pacing and flow
    3. Show don't tell
    4. Consistency with {tone} tone
    5. Strengthen the {pov} perspective
    
    Scene to edit:
    {scene_text}
    
    Return only the edited scene without any additional commentary.
    """

def build_continuity_prompt(state: StoryState, scene_text: str) -> str:
    """Build prompt for continuity analysis.
    
    Args:
        state (StoryState): Complete story state
        scene_text (str): Scene to analyze
        
    Returns:
        str: Complete continuity analysis prompt
    """
    # Simplified for performance and repetition detection
    if not state['story_so_far'].strip():
        return "This is the first scene. Return 'No issues found'."
    
    return f"""
    Analyze this scene for problems with the previous story:
    
    Previous story (last 500 chars):
    {state['story_so_far'][-500:]}
    
    New scene:
    {scene_text}
    
    Check for REPETITION ISSUES:
    1. Repeated events or encounters (same meetings, conversations)
    2. Identical dialogue being repeated verbatim  
    3. Same locations being "discovered" again
    4. Characters being re-introduced as if meeting for first time
    5. Repetitive descriptions or phrases
    
    Return 'No issues found' if the scene properly advances the story,
    OR list specific repetition problems found.
    """

def parse_continuity_report(continuity_report: str) -> list:
    """Parse continuity analysis report into issue list.
    
    Args:
        continuity_report (str): Raw continuity analysis from LLM
        
    Returns:
        list: List of continuity issues (empty if none found)
    """
    issues = []
    if "No issues" not in continuity_report and "No inconsistencies" not in continuity_report:
        issues = continuity_report.split("\n")
    return issues

def build_art_prompt(scene_text: str, tone: str) -> str:
    """Build prompt for generating visual art descriptions.
    
    Args:
        scene_text (str): Scene to visualize
        tone (str): Story tone for visual style
        
    Returns:
        str: Complete art generation prompt
    """
    return f"""
    Create a detailed art prompt for this scene:
    {scene_text}
    
    The prompt should be rich with visual details about:
    1. Characters and their appearance
    2. Setting and atmosphere
    3. Lighting and mood
    4. Key action or moment to depict
    
    Style should match {tone} tone.
    
    Return only the art prompt without any additional commentary.
    """

# Create tool instances
retrieve_lore_tool = RetrieveLoreTool()
check_continuity_tool = CheckContinuityTool()

# Define CrewAI agents with consistent LLM configuration
def create_agent(role, goal, backstory, tools=None):
    """Create a CrewAI agent with consistent LLM configuration"""
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        llm=llm  # Ensure all agents use the configured LLM
    )

writer = create_agent(
    role='Creative Writer',
    goal='Craft engaging narrative scenes with rich descriptions and dialogue',
    backstory='An award-winning fantasy author with decades of experience in world-building and character development.'
)

editor = create_agent(
    role='Story Editor',
    goal='Polish writing, improve pacing, and ensure stylistic consistency',
    backstory='A seasoned editor from a major publishing house, known for sharpening prose without losing the author\'s voice.'
)

continuity_checker = create_agent(
    role='Continuity Checker',
    goal='Identify and resolve inconsistencies in characters, settings, and plot',
    backstory='A meticulous archivist who remembers every detail of the story universe.',
    tools=[retrieve_lore_tool, check_continuity_tool]
)

illustrator = create_agent(
    role='Illustrator',
    goal='Create vivid visual descriptions that can be translated into artwork',
    backstory='A concept artist for major fantasy films and games, skilled at visualizing scenes and characters.'
)

# Define LangGraph nodes with error handling
def writer_node(state: StoryState):
    """WORKFLOW NODE 1: Generate a new story scene.
    
    This node orchestrates scene generation by coordinating content creation and
    metadata extraction through specialized helper functions.
    
    Workflow Role:
        - Entry point for scene generation
        - Integrates existing story context with new content
        - Provides structured output for downstream nodes
        
    Args:
        state (StoryState): Current workflow state containing story context
        
    Returns:
        dict: Updated state with new scene in 'current_scene' key
    """
    def _generate_scene():
        # Generate the main scene content
        scene_text = generate_scene_content(state, llm)
        
        # Extract structured metadata from the generated scene
        scene_metadata = extract_scene_metadata(scene_text, llm)
        
        # Return structured scene data for workflow continuation
        return {
            "current_scene": {
                "text": scene_text,
                "metadata": scene_metadata
            }
        }
    
    # Execute scene generation with comprehensive error handling
    result = safe_execute(_generate_scene, "writer_node", None)
    
    # Provide coherent fallback if generation completely fails
    if result is None:
        fallback_scene_data = create_fallback_scene("default")
        return {
            "current_scene": fallback_scene_data
        }
    return result

def editor_node(state: StoryState):
    """WORKFLOW NODE 2: Edit and polish the generated scene.
    
    This node refines content from the writer node using a specialized
    editing prompt builder and maintains scene metadata consistency.
    
    Workflow Role:
        - Receives raw scene from writer_node
        - Applies editorial improvements without changing core content
        - Maintains scene metadata for downstream processing
        
    Args:
        state (StoryState): State containing current_scene from writer_node
        
    Returns:
        dict: Updated state with polished scene text
    """
    def _edit_scene():
        scene_text = state["current_scene"]["text"]
        
        # Build editing prompt using helper function
        prompt = build_editing_prompt(scene_text, state['tone'], state['pov'])
        
        # Apply editorial improvements
        edited_scene = invoke_llm_safely(llm, prompt, "editor_node")
        
        # Preserve metadata while updating text
        return {
            "current_scene": {
                "text": edited_scene, 
                "metadata": state["current_scene"]["metadata"]
            }
        }
    
    # Execute editing with error protection
    result = safe_execute(_edit_scene, "editor_node", None)
    
    # Fallback to original scene if editing fails
    if result is None:
        fallback_scene_data = create_fallback_scene("editing_failed")
        return {
            "current_scene": state.get("current_scene", fallback_scene_data)
        }
    return result

def continuity_node(state: StoryState):
    """WORKFLOW NODE 3: Check for story continuity errors.
    
    This is a CRITICAL CONTROL POINT in the workflow. It uses specialized
    helper functions to analyze consistency and determine workflow direction.
    
    Workflow Role:
        - Quality gate for story consistency
        - Can trigger workflow loops if issues are found
        - Prevents accumulation of plot holes and contradictions
        
    Conditional Logic:
        - If issues found: Workflow loops back to writer_node
        - If no issues: Workflow continues to illustrator_node
        
    Args:
        state (StoryState): Complete story state for consistency analysis
        
    Returns:
        dict: State with continuity_issues list (empty if no problems)
    """
    def _check_continuity():
        scene_text = state["current_scene"]["text"]
        
        # Build comprehensive continuity analysis prompt
        prompt = build_continuity_prompt(state, scene_text)
        
        # Analyze scene for consistency problems
        continuity_report = invoke_llm_safely(llm, prompt, "continuity_node")
        
        # Parse report into structured issue list
        issues = parse_continuity_report(continuity_report)
        
        return {"continuity_issues": issues}
    
    return safe_execute(_check_continuity, "continuity_node", {"continuity_issues": []})

def illustrator_node(state: StoryState):
    """WORKFLOW NODE 4: Generate visual art prompts for the scene.
    
    This node uses a specialized art prompt builder to create visual
    descriptions and manages the art prompt collection.
    
    Workflow Role:
        - Final creative enhancement step
        - Translates narrative text into visual descriptions
        - Accumulates art prompts for entire story
        
    Args:
        state (StoryState): State with approved scene from continuity check
        
    Returns:
        dict: State with new art prompt added to art_prompts list
    """
    def _create_art_prompt():
        scene = state["current_scene"]
        scene_text = scene["text"]
        
        # Build art generation prompt using helper function
        prompt = build_art_prompt(scene_text, state['tone'])
        
        # Generate visual description
        art_prompt = invoke_llm_safely(llm, prompt, "illustrator_node")
        
        # Add art prompt to collection and update scene metadata
        return {
            "art_prompts": state["art_prompts"] + [art_prompt],
            "current_scene": {
                "text": scene_text,
                "metadata": {
                    **scene["metadata"],
                    "art_prompt": art_prompt
                }
            }
        }
    
    # Execute art prompt generation with error protection
    result = safe_execute(_create_art_prompt, "illustrator_node", None)
    
    # Provide fallback art prompt if generation fails
    if result is None:
        fallback_prompt = "A fantasy scene with dramatic lighting"
        fallback_scene_data = create_fallback_scene("illustration_failed")
        return {
            "art_prompts": state.get("art_prompts", []) + [fallback_prompt],
            "current_scene": state.get("current_scene", fallback_scene_data)
        }
    return result

def update_story_node(state: StoryState):
    """WORKFLOW NODE 5: Integrate completed scene into the main story.
    
    This is the final node in the workflow cycle. It commits the processed
    scene to the main story and prepares the state for the next iteration.
    This node always leads to END, completing one cycle of the workflow.
    
    Workflow Role:
        - Final integration step
        - Accumulates story content across iterations
        - Updates message history for LangGraph
        - Prepares state for next workflow cycle
        
    Args:
        state (StoryState): Complete state with processed scene
        
    Returns:
        dict: Updated story_so_far and messages for next iteration
    """
    def _update_story():
        new_scene = state["current_scene"]["text"]
        
        # INTEGRATION: Add new scene to accumulated story
        # Each scene is separated by double newlines for readability
        updated_story = state["story_so_far"] + "\n\n" + new_scene
        
        # TODO: Extract and update characters and settings from metadata
        # In a production system, this would:
        # 1. Parse scene metadata for new character information
        # 2. Update character profiles and attributes
        # 3. Add new setting details to world-building database
        # 4. Track plot points and story progression
        
        # Update LangGraph message history
        return {
            "story_so_far": updated_story,
            "messages": state["messages"] + [HumanMessage(content=f"Added scene: {new_scene}")]
        }
    
    return safe_execute(
        _update_story, 
        "update_story_node", 
        {
            "story_so_far": state.get("story_so_far", ""),
            "messages": state.get("messages", [])
        }
    )


# Build the graph
workflow = StateGraph(StoryState)

# Add nodes
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.add_node("continuity_checker", continuity_node)
workflow.add_node("illustrator", illustrator_node)
workflow.add_node("update_story", update_story_node)

# Add edges
workflow.set_entry_point("writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", "continuity_checker")
# Add loop counter to prevent infinite recursion
def continuity_decision(state: StoryState) -> str:
    """Decide whether to loop back or continue based on continuity issues.
    
    Prevents infinite loops by limiting regeneration attempts.
    """
    # Check if we have loop counter in state
    loop_count = state.get("continuity_loop_count", 0)
    
    # If we've looped too many times, force continuation
    if loop_count >= 2:  # Allow max 2 regeneration attempts
        print(f"Warning: Maximum continuity loop attempts reached, continuing anyway")
        return "continue"
    
    # Normal decision logic
    return "loop_back" if state.get("continuity_issues") else "continue"

workflow.add_conditional_edges(
    "continuity_checker",
    continuity_decision,
    {
        "loop_back": "writer",
        "continue": "illustrator"
    }
)
workflow.add_edge("illustrator", "update_story")
workflow.add_edge("update_story", END)

# Compile the graph
app = workflow.compile()

# Initialize state
initial_state = {
    "messages": [],
    "story_so_far": "",
    "characters": {},
    "settings": {},
    "plot_points": [],
    "current_scene": {},
    "art_prompts": [],
    "genre": "fantasy",
    "tone": "dark",
    "pov": "first-person",
    "continuity_issues": [],
    "continuity_loop_count": 0
}

# Run the graph for configured number of scenes
print(f"Starting collaborative storytelling for {scene_count} scenes...")
logger.info(f"Starting main execution loop for {scene_count} scenes")

for scene_num in range(scene_count):
    print(f"\n=== Generating Scene {scene_num + 1} ===")
    logger.info(f"\n{'='*50}")
    logger.info(f"STARTING SCENE {scene_num + 1} OF {scene_count}")
    logger.info(f"{'='*50}")
    
    # Reset LLM counter for this scene
    reset_llm_counter()
    scene_start_time = time.time()
    
    try:
        logger.info("Using controlled scene generation (no recursion)...")
        logger.debug(f"Initial state keys: {list(initial_state.keys())}")
        logger.debug(f"Story length before: {len(initial_state.get('story_so_far', ''))} chars")
        
        # Use controlled scene generation instead of recursive workflow
        result = generate_scene_with_limits(initial_state)
        
        scene_elapsed_time = time.time() - scene_start_time
        total_calls = get_llm_call_count()
        
        initial_state = result  # Update state for next iteration
        
        logger.info(f"Scene {scene_num + 1} completed in {scene_elapsed_time:.2f}s using {total_calls} LLM calls")
        logger.debug(f"Result keys: {list(result.keys())}")
        logger.debug(f"Story length after: {len(result.get('story_so_far', ''))} chars")
        
        print(f"Scene {scene_num + 1} completed successfully (used {total_calls} LLM calls)")
        
        # Display efficiency info
        if total_calls > max_llm_calls_per_scene:
            logger.warning(f"Scene {scene_num + 1} exceeded recommended LLM call limit")
            
    except Exception as e:
        logger.error(f"Error generating scene {scene_num + 1}: {e}", exc_info=True)
        print(f"Error generating scene {scene_num + 1}: {e}")
        # Add a small delay before continuing to prevent rapid failures
        logger.info("Adding 2-second delay before continuing")
        time.sleep(2)

logger.info("Main execution loop completed")

# Output the final story
print("\n" + "="*50)
print("FINAL STORY")
print("="*50)
print(initial_state["story_so_far"])
print("\n" + "="*50)
print("ART PROMPTS")
print("="*50)
for i, prompt in enumerate(initial_state["art_prompts"]):
    print(f"\nScene {i+1}: {prompt}")

