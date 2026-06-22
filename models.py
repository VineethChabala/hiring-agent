from typing import List, Optional, Dict, Tuple, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelProvider(Enum):
    """Enum for supported model providers."""

    OLLAMA = "ollama"
    GEMINI = "gemini"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to the LLM provider."""
        ...


class Location(BaseModel):
    """Location information for JSON Resume format."""

    address: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None
    countryCode: Optional[str] = None
    region: Optional[str] = None


class Profile(BaseModel):
    """Social profile information for JSON Resume format."""

    network: Optional[str] = None
    username: Optional[str] = None
    url: str


class Basics(BaseModel):
    """Basic information for JSON Resume format."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = None


class Work(BaseModel):
    """Work experience for JSON Resume format."""

    name: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Volunteer(BaseModel):
    """Volunteer experience for JSON Resume format."""

    organization: Optional[str] = None
    position: Optional[str] = None
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[List[str]] = None


class Education(BaseModel):
    """Education information for JSON Resume format."""

    institution: Optional[str] = None
    url: Optional[str] = None
    area: Optional[str] = None
    studyType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    score: Optional[str] = None
    courses: Optional[List[str]] = None


class Award(BaseModel):
    """Award information for JSON Resume format."""

    title: Optional[str] = None
    date: Optional[str] = None
    awarder: Optional[str] = None
    summary: Optional[str] = None


class Certificate(BaseModel):
    """Certificate information for JSON Resume format."""

    name: Optional[str] = None
    date: Optional[str] = None
    issuer: Optional[str] = None
    url: Optional[str] = None


class Publication(BaseModel):
    """Publication information for JSON Resume format."""

    name: Optional[str] = None
    publisher: Optional[str] = None
    releaseDate: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class Skill(BaseModel):
    """Skill information for JSON Resume format."""

    name: Optional[str] = None
    level: Optional[str] = None
    keywords: Optional[List[str]] = None


class Language(BaseModel):
    """Language information for JSON Resume format."""

    language: Optional[str] = None
    fluency: Optional[str] = None


class Interest(BaseModel):
    """Interest information for JSON Resume format."""

    name: Optional[str] = None
    keywords: Optional[List[str]] = None


class Reference(BaseModel):
    """Reference information for JSON Resume format."""

    name: Optional[str] = None
    reference: Optional[str] = None


class Project(BaseModel):
    """Project information for JSON Resume format."""

    name: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[List[str]] = None
    url: Optional[str] = None
    technologies: Optional[List[str]] = None
    skills: Optional[List[str]] = None


class BasicsSection(BaseModel):
    """Basics section containing basic information."""

    basics: Optional[Basics] = None


class WorkSection(BaseModel):
    """Work section containing a list of work experiences."""

    work: Optional[List[Work]] = None


class EducationSection(BaseModel):
    """Education section containing a list of education entries."""

    education: Optional[List[Education]] = None


class SkillsSection(BaseModel):
    """Skills section containing a list of skill categories."""

    skills: Optional[List[Skill]] = None


class ProjectsSection(BaseModel):
    """Projects section containing a list of projects."""

    projects: Optional[List[Project]] = None


class AwardsSection(BaseModel):
    """Awards section containing a list of awards."""

    awards: Optional[List[Award]] = None


class JSONResume(BaseModel):
    """Complete JSON Resume format model."""

    basics: Optional[Basics] = None
    work: Optional[List[Work]] = None
    volunteer: Optional[List[Volunteer]] = None
    education: Optional[List[Education]] = None
    awards: Optional[List[Award]] = None
    certificates: Optional[List[Certificate]] = None
    publications: Optional[List[Publication]] = None
    skills: Optional[List[Skill]] = None
    languages: Optional[List[Language]] = None
    interests: Optional[List[Interest]] = None
    references: Optional[List[Reference]] = None
    projects: Optional[List[Project]] = None


class CategoryScore(BaseModel):
    score: float = Field(ge=0, description="Score achieved in this category")
    max: int = Field(gt=0, description="Maximum possible score")
    evidence: str = Field(min_length=1, description="Evidence supporting the score")


class Scores(BaseModel):
    open_source: CategoryScore
    self_projects: CategoryScore
    production: CategoryScore
    technical_skills: CategoryScore


class BonusPoints(BaseModel):
    total: float = Field(ge=0, le=20, description="Total bonus points")
    breakdown: str = Field(description="Breakdown of bonus points")


class Deductions(BaseModel):
    total: float = Field(
        ge=0,
        description="Total deduction points (stored as positive, applied as negative)",
    )
    reasons: str = Field(description="Reasons for deductions")


class EvaluationData(BaseModel):
    scores: Scores
    bonus_points: BonusPoints
    deductions: Deductions
    key_strengths: List[str] = Field(min_items=1, max_items=5)
    areas_for_improvement: List[str] = Field(min_items=1, max_items=5)


class GitHubProfile(BaseModel):
    """Pydantic model for GitHub profile data."""

    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    avatar_url: Optional[str] = None
    blog: Optional[str] = None
    twitter_username: Optional[str] = None
    hireable: Optional[bool] = None


class OllamaProvider:
    """Ollama LLM provider implementation."""

    def __init__(self):
        import ollama

        self.client = ollama

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama."""
        # Fallback if a Gemini model is passed to Ollama
        from prompt import MODEL_PROVIDER_MAPPING, ModelProvider
        if MODEL_PROVIDER_MAPPING.get(model) == ModelProvider.GEMINI:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Model '{model}' is explicitly mapped to Gemini but Ollama provider is active. "
                f"Falling back to 'gemma3:4b' for Ollama provider."
            )
            model = "gemma3:4b"

        ollama_options = options.copy() if options else {}

        # remove steam from ollama options
        ollama_options.pop("stream", None)

        # Add num_ctx 32K context window to options
        ollama_options["num_ctx"] = 32768

        # convert to chat params
        chat_params = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }

        # add it to top level
        if "stream" in kwargs:
            chat_params["stream"] = kwargs["stream"]

        if "response_model" in kwargs:
            chat_params["format"] = kwargs["response_model"].model_json_schema()
        elif "format" in kwargs:
            chat_params["format"] = kwargs["format"]

        return self.client.chat(**chat_params)


def pydantic_to_gemini_schema(model_class_or_schema) -> dict:
    """Convert a Pydantic model or schema dict into a Google Gemini API compatible schema."""
    if hasattr(model_class_or_schema, "model_json_schema"):
        raw_schema = model_class_or_schema.model_json_schema()
    else:
        raw_schema = model_class_or_schema

    if not isinstance(raw_schema, dict):
        return raw_schema

    defs = raw_schema.get("$defs", {})

    def resolve_refs(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                def_name = ref_path.split("/")[-1]
                return resolve_refs(defs[def_name])
            return {k: resolve_refs(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [resolve_refs(x) for x in node]
        return node

    # Resolve all references first
    resolved_schema = resolve_refs(raw_schema)

    def flatten_schema(node):
        if not isinstance(node, dict):
            return node

        for union_key in ["anyOf", "oneOf"]:
            if union_key in node:
                sub_schemas = node[union_key]
                # Filter out null schema
                non_null = [s for s in sub_schemas if not (isinstance(s, dict) and s.get("type") == "null")]
                if non_null:
                    # Pick the first non-null schema and flatten it
                    flat = flatten_schema(non_null[0])
                    if isinstance(flat, dict):
                        # Merge other keys from node (like description) but keep the flattened schema properties
                        result = flat.copy()
                        if "description" in node:
                            result["description"] = node["description"]
                        result["nullable"] = True
                        return result
                    return flat
                break

        # Handle allOf
        if "allOf" in node:
            merged = {}
            for sub in node["allOf"]:
                flat = flatten_schema(sub)
                if isinstance(flat, dict):
                    merged.update(flat)
            # Add other keys
            for k, v in node.items():
                if k != "allOf" and k not in merged:
                    merged[k] = v
            return flatten_schema(merged)

        return {k: flatten_schema(v) for k, v in node.items()}

    flat_schema = flatten_schema(resolved_schema)

    def clean_schema(node):
        if not isinstance(node, dict):
            return node

        cleaned = {}
        
        # Determine type
        if "type" in node:
            t = node["type"]
            if isinstance(t, list):
                non_null_types = [x for x in t if x != "null"]
                if non_null_types:
                    cleaned["type"] = non_null_types[0]
                if "null" in t:
                    cleaned["nullable"] = True
            else:
                cleaned["type"] = t
        
        # Standard schema fields
        for field in ["description", "properties", "required", "items", "enum", "nullable"]:
            if field in node:
                if field == "properties":
                    cleaned["properties"] = {k: clean_schema(v) for k, v in node["properties"].items()}
                elif field == "items":
                    cleaned["items"] = clean_schema(node["items"])
                elif field == "nullable":
                    cleaned["nullable"] = bool(node["nullable"])
                else:
                    cleaned[field] = node[field]

        # For array type, items is required by Gemini
        if cleaned.get("type") == "array" and "items" not in cleaned:
            cleaned["items"] = {"type": "string"}

        # If no type, determine if object or array or string
        if "type" not in cleaned:
            if "properties" in cleaned:
                cleaned["type"] = "object"
            elif "items" in cleaned:
                cleaned["type"] = "array"
            else:
                cleaned["type"] = "string"

        return cleaned

    return clean_schema(flat_schema)


def normalize_model_name_for_gemini(model_name: str) -> str:
    """Normalize Ollama-style model names to Gemini-style API names."""
    if not model_name:
        return model_name
        
    model_lower = model_name.lower().strip()
    
    # Direct mappings
    mappings = {
        "gemma4:31b": "gemma-4-31b-it",
        "gemma4:26b": "gemma-4-26b-a4b-it",
        "gemma-4-31b": "gemma-4-31b-it",
        "gemma-4-26b": "gemma-4-26b-a4b-it",
    }
    if model_lower in mappings:
        return mappings[model_lower]
        
    # Regex-based normalization (e.g., gemma4:31b -> gemma-4-31b-it)
    import re
    if ":" in model_name:
        parts = model_name.split(":")
        base = parts[0]
        tag = parts[1]
        
        # Add hyphen between name and version (e.g. gemma4 -> gemma-4)
        base = re.sub(r"([a-zA-Z]+)(\d+)", r"\1-\2", base)
        # Combine
        normalized = f"{base}-{tag}"
        # Append -it if it's a gemma model and doesn't end with -it
        if "gemma" in normalized.lower() and not normalized.lower().endswith("-it"):
            normalized = f"{normalized}-it"
        return normalized
        
    return model_name


class GeminiProvider:
    """Google Gemini API provider implementation."""

    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Google Gemini API."""
        # Normalize model name first to resolve Gemini-supported names
        model = normalize_model_name_for_gemini(model)

        # Fallback ONLY if the model is explicitly mapped to Ollama
        from prompt import MODEL_PROVIDER_MAPPING, ModelProvider
        if MODEL_PROVIDER_MAPPING.get(model) == ModelProvider.OLLAMA:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Model '{model}' is explicitly mapped to Ollama but Gemini provider is active. "
                f"Falling back to 'gemini-3.5-flash' for Gemini provider."
            )
            model = "gemini-3.5-flash"

        # Map options to Gemini parameters
        generation_config = {}
        if options:
            if "temperature" in options:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options:
                generation_config["top_p"] = options["top_p"]

        # Support structured JSON output schema via format or response_model
        if "response_model" in kwargs:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = pydantic_to_gemini_schema(kwargs["response_model"])
        elif "format" in kwargs:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = pydantic_to_gemini_schema(kwargs["format"])

        # Extract system instruction and filter chat history
        system_instruction = None
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})

        # Create a Gemini model with system instruction
        gemini_model = self.client.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )

        # Send the chat request with adaptive retry logic for rate limits/quota/server errors
        import time
        import re
        import logging
        from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable

        logger = logging.getLogger(__name__)

        max_retries = 5
        base_delay = 5.0  # seconds

        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content(gemini_messages)
                # Convert Gemini response to Ollama-like format for compatibility
                return {"message": {"role": "assistant", "content": response.text}}
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = (
                    isinstance(e, ResourceExhausted) or
                    "429" in error_msg or
                    "quota" in error_msg or
                    "exhausted" in error_msg or
                    "rate limit" in error_msg
                )
                is_server_error = (
                    isinstance(e, (InternalServerError, ServiceUnavailable)) or
                    "500" in error_msg or
                    "503" in error_msg or
                    "internal error" in error_msg or
                    "service unavailable" in error_msg or
                    "temporarily unavailable" in error_msg
                )
                if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                    # Attempt to extract explicit retry delay from error message
                    match = re.search(r"retry in ([\d\.]+)s", str(e))
                    if not match:
                        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
                    
                    if match:
                        delay = float(match.group(1)) + 1.5  # Add a 1.5s buffer
                    else:
                        delay = base_delay * (2 ** attempt)
                    
                    error_type = "rate limit" if is_rate_limit else "transient server error"
                    logger.warning(
                        f"⚠️ Gemini API {error_type} hit. Retrying attempt {attempt + 1}/{max_retries} "
                        f"in {delay:.2f} seconds... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Gemini API call failed: {e}")
                    raise
