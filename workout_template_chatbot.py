from __future__ import annotations
import os, orjson, uuid, re, secrets, traceback
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import enhanced_edit_template
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import SmartWorkoutEditor
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import extract_bulk_operation_info
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import AIConversationManager
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_structured import (
    StructurizeAndSaveRequest,
    _gather_ids,
    _fetch_qr_rows,
    _build_day_payload,
    _persist_payload
)

from app.models.deps import get_mem, get_oai
from app.models.database import get_db
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
   sse_json, OPENAI_MODEL, is_yes as _is_yes_base, is_no as _is_no_base
)
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import (
   is_workout_template_intent,
   render_markdown_from_template,
   llm_generate_template_from_profile,
   llm_edit_template,
   explain_template_with_llm,
   DAYS6,
   build_id_only_structure,
)
from app.models.fittbot_models import Client, WeightJourney, WorkoutTemplate, ClientTarget
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.exercise_catalog_db import load_catalog, id_for_name


def _generate_unique_day_key(template_name: str, existing_keys: set) -> str:
    """Generate a unique day key from template name, handling duplicates"""
    base_key = template_name.lower().replace(' ', '_').replace('-', '_')
    # Remove any non-alphanumeric characters except underscores
    base_key = re.sub(r'[^a-z0-9_]', '', base_key)

    if base_key not in existing_keys:
        return base_key

    # If duplicate, add suffix numbers until unique
    counter = 2
    while f"{base_key}_{counter}" in existing_keys:
        counter += 1

    return f"{base_key}_{counter}"


async def _ensure_template_has_ids(template: dict, db: Session) -> dict:
    """Ensure all exercises in template have valid IDs from the database"""
    if not template or not template.get('days'):
        return None

    try:
        # Load exercise catalog
        catalog = load_catalog(db)
        if not catalog:
            print("⚠️ Could not load exercise catalog")
            return None

        modified = False
        template_copy = template.copy()
        days_copy = template_copy.get('days', {}).copy()

        for day_key, day_data in days_copy.items():
            if not isinstance(day_data, dict):
                continue

            exercises = day_data.get('exercises', [])
            if not isinstance(exercises, list):
                continue

            updated_exercises = []
            for exercise in exercises:
                if not isinstance(exercise, dict):
                    continue

                # Check if exercise already has a valid ID
                existing_id = exercise.get('id')
                if isinstance(existing_id, int) and existing_id > 0:
                    updated_exercises.append(exercise)
                    continue

                # Try to find ID by exercise name
                exercise_name = exercise.get('name', '')
                if exercise_name:
                    found_id = id_for_name(exercise_name, catalog)
                    if found_id:
                        exercise_copy = exercise.copy()
                        exercise_copy['id'] = found_id
                        updated_exercises.append(exercise_copy)
                        modified = True
                        print(f"✅ Assigned ID {found_id} to exercise '{exercise_name}'")
                    else:
                        print(f"⚠️ Could not find ID for exercise '{exercise_name}'")
                        # Still add the exercise without ID - it will be filtered out later
                        updated_exercises.append(exercise)
                else:
                    print(f"⚠️ Exercise missing name: {exercise}")
                    updated_exercises.append(exercise)

            # Update the day's exercises
            day_data_copy = day_data.copy()
            day_data_copy['exercises'] = updated_exercises
            days_copy[day_key] = day_data_copy

        template_copy['days'] = days_copy

        if modified:
            print(f"✅ Template updated with exercise IDs")

        return template_copy

    except Exception as e:
        print(f"❌ Error ensuring template has IDs: {e}")
        return None


def _ensure_unique_exercise_ids(template: dict) -> dict:
    """Ensure all exercise IDs are unique across the entire template"""
    if not template or not template.get('days'):
        return template

    used_ids = set()
    next_id = 1

    # First pass: collect all existing IDs and find the next available ID
    for day_data in template['days'].values():
        if isinstance(day_data, dict) and 'exercises' in day_data:
            for exercise in day_data.get('exercises', []):
                if isinstance(exercise, dict) and 'id' in exercise:
                    exercise_id = exercise['id']
                    if isinstance(exercise_id, int):
                        used_ids.add(exercise_id)
                        next_id = max(next_id, exercise_id + 1)

    # Second pass: assign unique IDs to any exercises without IDs or with duplicate IDs
    for day_data in template['days'].values():
        if isinstance(day_data, dict) and 'exercises' in day_data:
            exercises = day_data.get('exercises', [])
            for exercise in exercises:
                if isinstance(exercise, dict):
                    if 'id' not in exercise or exercise['id'] in used_ids:
                        # Find next available ID
                        while next_id in used_ids:
                            next_id += 1
                        exercise['id'] = next_id
                        used_ids.add(next_id)
                        next_id += 1

    return template


def _format_template_for_display(template: dict) -> str:
    """Format template for frontend display with enhanced styling and emojis"""
    if not template or not template.get('days'):
        return "❌ No workout data available"

    formatted_lines = []
    day_count = 1

    # Add attractive header
    formatted_lines.append("💪 YOUR WORKOUT TEMPLATE 💪")
    formatted_lines.append("═" * 40)
    formatted_lines.append("")

    for day_key, day_data in template['days'].items():
        if not isinstance(day_data, dict):
            continue

        # Get the title from the template data
        title = day_data.get('title', '')

        # Create comprehensive day header with emojis
        day_emoji = _get_day_emoji(day_count)

        if title:
            # Use the custom title which may include custom day names
            if "—" in title or ":" in title:
                combined_title = f"{day_emoji} Day {day_count}: {title} {day_emoji}"
            else:
                combined_title = f"{day_emoji} Day {day_count}: {title} {day_emoji}"
        else:
            # Fallback to cleaned up key name
            clean_title = day_key.replace('_', ' ').title()
            combined_title = f"{day_emoji} Day {day_count}: {clean_title} {day_emoji}"

        formatted_lines.append(combined_title)
        formatted_lines.append("─" * (len(combined_title) - 4))  # Adjust for emoji length
        formatted_lines.append("")

        # Add exercises with better formatting
        exercises = day_data.get('exercises', [])
        if exercises:
            for i, exercise in enumerate(exercises, 1):
                if isinstance(exercise, dict):
                    name = exercise.get('name', 'Unknown Exercise')
                    sets = exercise.get('sets', 0)
                    reps = exercise.get('reps', 0)
                    exercise_emoji = _get_exercise_emoji(name)
                    formatted_lines.append(f"   {exercise_emoji} {i}. {name}")
                    formatted_lines.append(f"      📊 {sets} sets × {reps} reps")
                    formatted_lines.append("")
        else:
            formatted_lines.append("   ⚠️ No exercises added yet")
            formatted_lines.append("")

        formatted_lines.append("") # Extra space between days
        day_count += 1

    formatted_lines.append("═" * 40)
    formatted_lines.append("🎯 Ready to crush your goals! 🎯")

    return "\n".join(formatted_lines)

def _get_day_emoji(day_num: int) -> str:
    """Get emoji based on day number"""
    day_emojis = {
        1: "🔥", 2: "💥", 3: "⚡", 4: "🚀",
        5: "💪", 6: "🎯", 7: "🌟"
    }
    return day_emojis.get(day_num, "💫")

def _get_exercise_emoji(exercise_name: str) -> str:
    """Get relevant emoji based on exercise type"""
    exercise_name_lower = exercise_name.lower()

    if any(word in exercise_name_lower for word in ['squat', 'leg', 'deadlift', 'lunge']):
        return "🦵"
    elif any(word in exercise_name_lower for word in ['bench', 'press', 'chest', 'push']):
        return "💪"
    elif any(word in exercise_name_lower for word in ['pull', 'row', 'lat', 'back']):
        return "🎣"
    elif any(word in exercise_name_lower for word in ['shoulder', 'overhead', 'lateral']):
        return "🤲"
    elif any(word in exercise_name_lower for word in ['curl', 'bicep', 'arm']):
        return "💪"
    elif any(word in exercise_name_lower for word in ['tricep', 'dip', 'extension']):
        return "💥"
    elif any(word in exercise_name_lower for word in ['core', 'plank', 'abs', 'crunch']):
        return "🔥"
    elif any(word in exercise_name_lower for word in ['cardio', 'run', 'bike', 'treadmill']):
        return "🏃"
    else:
        return "🏋️"
router = APIRouter(prefix="/workout_template", tags=["workout_template"])
# ═══════════════════════════════════════════════════════════════
# ENHANCED FLEXIBLE NATURAL LANGUAGE PROCESSING
# ═══════════════════════════════════════════════════════════════
class UltraFlexibleParser:
   """Ultra-flexible natural language parser with typo tolerance and context awareness"""
  
   # Intent detection with fuzzy matching
   CREATE_INTENTS = {
       'patterns': [
           r'(?:create|make|build|generate|new|start|design|craft|setup|construct)',
           r'(?:workout|template|plan|routine|program|schedule|regimen)',
           r'(?:want|need|like|prefer).*(?:workout|plan|routine)',
           r'(?:give|show).*(?:me|us).*(?:workout|plan)',
           r'(?:i|we).*(?:want|need|would like).*(?:to|a).*(?:workout|exercise)',
           r'(?:let\'s|lets).*(?:create|make|start|begin)',
       ],
       'keywords': ['create', 'make', 'build', 'new', 'workout', 'plan', 'routine', 'template'],
       'confidence_threshold': 0.3
   }
  
   SHOW_INTENTS = {
       'patterns': [
           r'(?:show|view|see|display|look|check).*(?:my|current|existing|saved)',
           r'(?:what|which).*(?:template|plan|routine|workout).*(?:have|got|saved)',
           r'(?:current|existing|saved|my).*(?:template|plan|routine|workout)',
           r'(?:see|view|show|display).*(?:template|plan|routine|workout)',
       ],
       'keywords': ['show', 'view', 'see', 'current', 'existing', 'my', 'saved'],
       'confidence_threshold': 0.25
   }
  
   EDIT_INTENTS = {
       'patterns': [
           r'(?:change|edit|modify|alter|update|adjust|tweak|fix|improve)',
           r'(?:replace|swap|substitute|switch|exchange)',
           r'(?:add|include|insert|put in|bring in).*(?:more|some|extra)',
           r'(?:remove|delete|take out|exclude|drop)',
           r'(?:increase|decrease|more|less|heavier|lighter|harder|easier)',
           r'(?:different|another|other|alternative)',
           r'(?:i|we).*(?:want|need|would like).*(?:to|different|other)',
       ],
       'keywords': ['change', 'edit', 'modify', 'different', 'more', 'less', 'add', 'remove'],
       'confidence_threshold': 0.2
   }
  
   # Ultra-flexible day patterns with common typos and abbreviations
   DAY_PATTERNS = {
       'monday': [
           r'mon(?:day)?', r'm[ou]n\w*', r'mnd?y?', r'mndy', r'mond?', r'monda?y?'
       ],
       'tuesday': [
           r'tue(?:s(?:day)?)?', r't[ue]\w*', r'tues?', r'tusd?y?', r'tusday'
       ],
       'wednesday': [
           r'wed(?:nesday)?', r'w[ed]\w*', r'wedn?', r'wedns?day', r'wensd?y?'
       ],
       'thursday': [
           r'thu(?:rs?day)?', r'th[ur]\w*', r'thrs?', r'thursd?y?', r'thrsdy'
       ],
       'friday': [
           r'fri(?:day)?', r'f[ri]\w*', r'frid?y?', r'fridy'
       ],
       'saturday': [
           r'sat(?:urday)?', r's[at]\w*', r'satd?y?', r'saturdy', r'satrdy'
       ],
       'sunday': [
           r'sun(?:day)?', r's[un]\w*', r'sund?y?', r'sundy'
       ]
   }
  
   # Flexible number extraction patterns
   NUMBER_PATTERNS = [
       r'\b(\d+)\s*(?:days?|day)\b',           # "5 days", "3day"
       r'\b(\d+)\s*(?:times?|time)?\s*(?:per|a)?\s*week\b',  # "5 times a week"
       r'\b(\d+)\s*(?:workout|session)s?\b',   # "5 workouts"
       r'(?:for|about|around)\s*(\d+)\b',      # "for 5"
       r'\b(\d+)\s*(?:of|out of)\s*7\b',       # "5 of 7"
       r'(\d+)',                               # any standalone number
   ]
  
   # Flexible yes/no patterns with context awareness
   POSITIVE_PATTERNS = [
       r'^(?:y|yes|yep|yeah|yup|ya|sure|ok|okay|alright|right)$',
       r'^(?:go|do)(?:\s*(?:ahead|it|that))?$',
       r'^(?:proceed|continue|next|forward)$',
       r'^(?:please|absolutely|definitely|certainly|of course)$',
       r'^(?:sounds?\s*(?:good|great|fine|perfect))$',
       r'^(?:that(?:\'s|s)?\s*(?:good|great|fine|perfect|right))$',
       r'^(?:let(?:\'s|s)?\s*(?:go|do it))$',
       r'^(?:i(?:\'m|m)?\s*(?:ready|good))$',
       r'^perfect$', r'^good$', r'^great$', r'^fine$',
       r'^save(?:\s*it)?$', r'^confirm$', r'^approved?$'
   ]
  
   NEGATIVE_PATTERNS = [
       r'^(?:n|no|nope|nah|not?)$',
       r'^(?:cancel|stop|quit|exit|abort)$',
       r'^(?:not\s*(?:now|yet|today|ready))$',
       r'^(?:skip|pass|later|maybe\s*later)$',
       r'^(?:don\'?t|do\s*not|not\s*(?:really|quite))$',
       r'^(?:i\s*(?:don\'?t|do\s*not)\s*(?:want|like|think))$',
       r'^(?:that\'?s\s*(?:not|wrong))$',
       r'^(?:need\s*(?:changes?|edit|different))$'
   ]

   ALL_DAYS_PATTERNS = [
    r'(?:all|every|each)\s*days?',
    r'(?:all|every|each)\s*(?:of\s*the\s*)?(?:workout\s*)?days?',
    r'(?:for\s*)?(?:all|every|each)\s*(?:day|days)',
    r'(?:on\s*)?(?:all|every|each)\s*(?:day|days)',
]

   SPECIFIC_COUNT_PATTERNS = [
    r'(?:for|on)\s*(\d+)\s*days?',
    r'(\d+)\s*days?',
    r'(?:for|on)\s*(?:the\s*)?(?:first|last)\s*(\d+)\s*days?',
]

   MUSCLE_CHANGE_PATTERNS = {
    'legs': [r'leg\s*(?:exercise|workout|training)', r'lower\s*body', r'quadriceps?', r'hamstrings?', r'glutes?'],
    'upper': [r'upper\s*body', r'upper\s*(?:exercise|workout)', r'chest\s*and\s*arms?', r'arms?\s*and\s*chest'],
    'core': [r'core\s*(?:exercise|workout)', r'ab\s*(?:exercise|workout)', r'abdominal'],
    'chest': [r'chest\s*(?:exercise|workout)', r'pec\s*(?:exercise|workout)'],
    'back': [r'back\s*(?:exercise|workout)', r'lat\s*(?:exercise|workout)', r'pull\s*(?:exercise|workout)'],
    'biceps': [r'bicep\s*(?:exercise|workout)', r'arm\s*curl', r'bicep\s*curl'],
    'triceps': [r'tricep\s*(?:exercise|workout)', r'tri\s*(?:exercise|workout)'],
    'shoulders': [r'shoulder\s*(?:exercise|workout)', r'delt\s*(?:exercise|workout)'],
    'cardio': [r'cardio\s*(?:exercise|workout)', r'aerobic', r'running', r'cycling']
}
  
   @classmethod
   def calculate_intent_confidence(cls, text: str, intent_config: Dict) -> float:
       """Calculate confidence score for intent detection"""
       text_lower = text.lower().strip()
       confidence = 0.0
      
       # Pattern matching
       pattern_matches = sum(1 for pattern in intent_config['patterns']
                           if re.search(pattern, text_lower, re.I))
       if pattern_matches > 0:
           confidence += (pattern_matches / len(intent_config['patterns'])) * 0.6
      
       # Keyword matching with fuzzy tolerance
       keyword_matches = sum(1 for keyword in intent_config['keywords']
                           if keyword in text_lower or
                           any(cls._fuzzy_match(keyword, word) for word in text_lower.split()))
       if keyword_matches > 0:
           confidence += (keyword_matches / len(intent_config['keywords'])) * 0.4
      
       return min(confidence, 1.0)
  
   @classmethod
   def _fuzzy_match(cls, target: str, word: str, threshold: float = 0.8) -> bool:
       """Simple fuzzy string matching for typo tolerance"""
       if len(word) < 3 or len(target) < 3:
           return word == target
      
       # Simple character overlap ratio
       common_chars = set(target) & set(word)
       similarity = len(common_chars) / max(len(set(target)), len(set(word)))
       return similarity >= threshold
  
   @classmethod
   def extract_intent(cls, text: str, context: Optional[Dict] = None) -> Tuple[str, float]:
       """Extract primary intent with confidence score"""
       text = text.strip()
      
       # Calculate confidence for each intent
       create_conf = cls.calculate_intent_confidence(text, cls.CREATE_INTENTS)
       show_conf = cls.calculate_intent_confidence(text, cls.SHOW_INTENTS)
       edit_conf = cls.calculate_intent_confidence(text, cls.EDIT_INTENTS)
      
       # Context-aware adjustments
       if context:
           current_state = context.get('state', '')
           if current_state in ['EDIT_DECISION', 'CONFIRM_SAVE']:
               edit_conf += 0.2  # Boost edit confidence in edit contexts
      
       # Determine best intent
       confidences = [
           ('create', create_conf),
           ('show', show_conf),
           ('edit', edit_conf)
       ]
      
       best_intent, best_conf = max(confidences, key=lambda x: x[1])
      
       if best_conf < 0.15:  # Very low confidence threshold
           return "unknown", best_conf
          
       return best_intent, best_conf
  
   





   @classmethod
   def extract_days_count(cls, text: str) -> Optional[int]:
        """Ultra-flexible day count extraction - returns None if no days found"""
        if not text or not text.strip():
            return None
            
        text = text.lower().strip()
        
        # Handle special phrases first
        special_phrases = {
            'usual': 6, 'normal': 6, 'default': 6, 'standard': 6, 'typical': 6,
            'full week': 7, 'whole week': 7, 'all days': 7, 'every day': 7, 'daily': 7,
            'weekdays': 5, 'work days': 5, 'monday to friday': 5, 'mon-fri': 5,
            'weekend': 2, 'weekends': 2,
            'monday to saturday': 6, 'mon-sat': 6,
            'as usual': 6, 'like usual': 6, 'same as usual': 6,
            '1week': 7, '1 week': 7, 'one week': 7,
            '2week': 14, '2 week': 14, 'two week': 14,
            'week': 7, 'weekly': 7
        }
        
        for phrase, count in special_phrases.items():
            if phrase in text:
                return count
        
        # Enhanced number extraction patterns
        enhanced_patterns = [
            r'^\s*(\d+)\s*$',  # ADD THIS LINE - matches standalone numbers like "5"
            r'\b(\d+)\s*(?:days?|day)\b',
            r'\b(\d+)\s*(?:times?|time)?\s*(?:per|a)?\s*week\b',
            r'\b(\d+)\s*(?:workout|session)s?\b',
            r'(?:for|about|around)\s*(\d+)\b',
            r'\b(\d+)\s*(?:of|out of)\s*7\b',
            r'(?:build|create|make)\s*(\d+)',
            r'(\d+)\s*(?:days?|day)?\s*(?:workout|plan|routine)',
            r'(?:create|make|build)\s*(\d+)\s*(?:template|plan)s?',
            r'(\d+)\s*(?:template|plan)s?',
            r'(\d+)\s*weeks?\s*(?:of|worth)',
            r'(\d+)\s*(?:week|weekly)',
        ]
        
        for pattern in enhanced_patterns:
            matches = re.findall(pattern, text, re.I)
            if matches:
                try:
                    count = int(matches[0])
                    # Special handling for week requests
                    if 'week' in text and count <= 4:
                        return count * 7
                    elif 1 <= count <= 7:
                        return count
                except ValueError:
                    continue
        
        # Count explicit day mentions with fuzzy matching
        mentioned_days = set()
        for day, patterns in cls.DAY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    mentioned_days.add(day)
                    break
        
        if mentioned_days:
            return len(mentioned_days)
        
        # Return None if no days information found
        return None
  
   @classmethod
   def extract_template_names(cls, text: str, count: int) -> List[str]:
       """Ultra-flexible template name extraction"""
       text = text.lower().strip()

       # Handle empty input or "nothing" keywords - return proper day names immediately
       nothing_keywords = ['nothing', 'no', 'skip', 'default', 'defaults', 'normal', 'standard', 'none', 'nope', 'nah']
       if not text or len(text) < 2 or text in nothing_keywords:
           default_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
           return default_days[:count] if count <= 7 else [f"Day {i+1}" for i in range(count)]

       if ',' in text:
        custom_names = [name.strip().title() for name in text.split(',') if name.strip()]
        if len(custom_names) >= count:
            return custom_names[:count]
        elif len(custom_names) > 0:
            # Pad with proper day names if needed
            default_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            while len(custom_names) < count:
                if len(custom_names) < 7:
                    custom_names.append(default_days[len(custom_names)])
                else:
                    custom_names.append(f"Day {len(custom_names)+1}")
            return custom_names[:count]
    
       # Handle default requests
       default_triggers = ['default', 'normal', 'standard', 'usual', 'typical', 'regular']
       if any(trigger in text for trigger in default_triggers):
           defaults = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
           return defaults[:count]
      
       # Handle day-based requests
       if any(re.search('|'.join(patterns), text, re.I)
              for patterns in cls.DAY_PATTERNS.values()):
           found_days = []
           for day, patterns in cls.DAY_PATTERNS.items():
               for pattern in patterns:
                   if re.search(pattern, text, re.I):
                       found_days.append(day.capitalize())
                       break
          
           if found_days:
               # Fill remaining with sequential defaults
               all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
               while len(found_days) < count:
                   for day in all_days:
                       if day not in found_days:
                           found_days.append(day)
                           break
                   if len(found_days) >= count:
                       break
               return found_days[:count]
      
       # Handle muscle group patterns
       muscle_groups = ['push', 'pull', 'legs', 'upper', 'lower', 'full body', 'cardio', 'arms', 'chest', 'back']
       found_groups = [group.title() for group in muscle_groups if group in text]
       if len(found_groups) >= count:
           return found_groups[:count]
      
       # Extract custom names (comma/newline separated)
       separators = [',', '\n', '|', ';', '/', '\\']
       for sep in separators:
           if sep in text:
               names = [name.strip().title() for name in text.split(sep) if name.strip()]
               if len(names) >= count:
                   return names[:count]
      
       # Try to extract quoted or numbered items
       quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
       if len(quoted) >= count:
           return [name.strip().title() for name in quoted[:count]]

       # ENHANCED: Try to extract space-separated custom names like "monster day crunch day"
       # Look for patterns like "word day" repeated
       day_pattern = r'(\w+\s+day)'
       day_matches = re.findall(day_pattern, text, re.I)
       if len(day_matches) >= count:
           return [match.strip().title() for match in day_matches[:count]]

       # Try to extract any meaningful words that could be day names
       # Skip common words that aren't likely to be custom day names
       skip_words = {
           'workout', 'template', 'plan', 'routine', 'exercise', 'training', 'fitness',
           'create', 'make', 'build', 'generate', 'want', 'need', 'like', 'prefer',
           'days', 'day', 'times', 'week', 'monday', 'tuesday', 'wednesday', 'thursday',
           'friday', 'saturday', 'sunday', 'the', 'and', 'or', 'but', 'for', 'with'
       }

       words = [word.strip() for word in text.split() if word.strip()]
       potential_names = []

       for word in words:
           if (len(word) > 2 and
               word.lower() not in skip_words and
               not word.isdigit() and
               len(potential_names) < count):
               potential_names.append(word.title())

       if len(potential_names) >= count:
           return potential_names[:count]
       elif len(potential_names) > 0:
           # Pad with proper day names if we found some custom names
           default_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
           while len(potential_names) < count:
               if len(potential_names) < 7:
                   potential_names.append(default_days[len(potential_names)])
               else:
                   potential_names.append(f"Day {len(potential_names) + 1}")
           return potential_names[:count]

       # Fallback to proper day names
       default_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
       return default_days[:count] if count <= 7 else [f"Day {i+1}" for i in range(count)]
   
   @classmethod
   def extract_comprehensive_workout_info(cls, text: str) -> Dict[str, Any]:
        """Extract all workout-related info from a single input"""
        result = {
            'has_days_info': False,
            'days_count': None,  # Changed from 6 to None
            'has_names_info': False,
            'template_names': [],
            'has_complete_request': False,
            'muscle_focus': None,
            'is_muscle_specific_template': False
        }
        
        # Check for day information - IMPROVED LOGIC
        days_count = cls.extract_days_count(text)

        # CRITICAL FIX: Only set has_days_info if we actually found day information
        if days_count is not None:
            result['has_days_info'] = True
            result['days_count'] = days_count
            print(f"🎯 Detected {days_count} days from: '{text}'")

        # Template number patterns - only if we found a number
        template_number_patterns = [
            r'(?:create|make|build)\s*(\d+)\s*(?:template|plan)s?',
            r'(\d+)\s*(?:template|plan)s?',
            r'(\d+)\s*(?:day|days)',
            r'(\d+)\s*(?:workout|routine)s?'
        ]

        found_number = None
        for pattern in template_number_patterns:
            match = re.search(pattern, text.lower())
            if match:
                found_number = int(match.group(1))
                break

        if found_number and not result['has_days_info']:
            result['has_days_info'] = True
            result['days_count'] = found_number
            print(f"🎯 Detected {found_number} days from template pattern: '{text}'")
        
        # Rest of the method remains the same...
        # NEW: Check for muscle-specific template creation
        muscle_template_patterns = [
            r'create\s+\d+\s*days?\s+(\w+)\s*(?:body|workout|template)',
            r'make\s+\d+\s*days?\s+(\w+)\s*(?:body|workout|template)',  
            r'(\w+)\s*(?:body|workout)\s+template',
            r'create\s+(\w+)\s*(?:body|workout)\s+for\s+\d+\s*days?',
            r'\d+\s*days?\s+(\w+)\s*(?:body|workout|template)'
        ]
        
        text_lower = text.lower()
        for pattern in muscle_template_patterns:
            match = re.search(pattern, text_lower)
            if match:
                potential_muscle = match.group(1).lower()
                muscle_mapping = {
                    'upper': 'upper', 'upperbody': 'upper', 'upper_body': 'upper',
                    'lower': 'legs', 'lowerbody': 'legs', 'lower_body': 'legs', 'leg': 'legs',
                    'core': 'core', 'ab': 'core', 'abs': 'core',
                    'chest': 'chest', 'back': 'back', 'arm': 'upper', 'arms': 'upper'
                }
                
                if potential_muscle in muscle_mapping:
                    result['muscle_focus'] = muscle_mapping[potential_muscle]
                    result['is_muscle_specific_template'] = True
                    result['has_complete_request'] = True
                    print(f"🎯 Detected muscle-specific template request: {result['muscle_focus']}")
                    break
        
        # Check for template name patterns (existing logic)
        if result['days_count']:
            template_names = cls.extract_template_names(text, result['days_count'])
            day_mentions = sum(1 for patterns in cls.DAY_PATTERNS.values() 
                            for pattern in patterns if re.search(pattern, text_lower))
            muscle_mentions = sum(1 for muscle in ['push', 'pull', 'legs', 'upper', 'lower', 'chest', 'back', 'arms'] 
                                if muscle in text_lower)
            
            if day_mentions > 0 or muscle_mentions > 0:
                result['has_names_info'] = True
                result['template_names'] = template_names
        
        # Check if this is a complete request - IMPROVED LOGIC
        create_patterns = [
            r'(?:create|make|build|generate).*(?:\d+.*)?(?:day|workout|plan|routine|template)',
            r'(?:\d+.*day).*(?:workout|plan|routine|template)',
            r'(?:workout|plan|routine|template).*(?:\d+.*day)',
        ]
        
        if any(re.search(pattern, text_lower) for pattern in create_patterns):
            result['has_complete_request'] = True
        
        return result
   

   
  
   @classmethod
   def is_positive_response(cls, text: str) -> bool:
        """Ultra-flexible positive response detection"""
        text = text.lower().strip()
        
        # Explicit save commands should be treated as positive for saving context
        save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'finalize', 'done']
        if text in save_commands:
            return True
            
        return any(re.search(pattern, text, re.I) for pattern in cls.POSITIVE_PATTERNS)
  
   @classmethod
   def is_negative_response(cls, text: str) -> bool:
    """Ultra-flexible negative response detection"""
    text = text.lower().strip()
    
    # Don't treat edit requests as negative
    edit_keywords = ['change', 'edit', 'modify', 'replace', 'alternative', 'different']
    if any(keyword in text for keyword in edit_keywords):
        return False
        
    return any(re.search(pattern, text, re.I) for pattern in cls.NEGATIVE_PATTERNS)
   

   @classmethod
   def extract_bulk_operation_info(cls, text: str) -> Dict[str, Any]:
        """Extract information for bulk operations like 'add biceps to all days'"""
        text_lower = text.lower()
        result = {
            'is_bulk_operation': False,
            'operation': None,  # 'add', 'replace', 'change'
            'target_muscle': None,
            'target_days': 'all',  # 'all', 'specific_count', 'specific_days'
            'specific_count': None,
            'specific_days': [],
            'is_complete_change': False  # Change entire template focus
        }
        
        # Check for bulk operations
        bulk_indicators = ['all days', 'every day', 'each day', 'for all', 'on all']
        if any(indicator in text_lower for indicator in bulk_indicators):
            result['is_bulk_operation'] = True
        
        # Check for specific day counts
        for pattern in cls.SPECIFIC_COUNT_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                result['is_bulk_operation'] = True
                result['target_days'] = 'specific_count'
                result['specific_count'] = int(match.group(1))
                break
        
        # Determine operation type
        if any(word in text_lower for word in ['change', 'replace', 'swap', 'make']):
            result['operation'] = 'replace'
            # Check if it's a complete template change
            if any(phrase in text_lower for phrase in ['change all', 'make all', 'create all']):
                result['is_complete_change'] = True
        elif any(word in text_lower for word in ['add', 'include', 'give', 'put']):
            result['operation'] = 'add'
        
        # Extract target muscle
        for muscle, patterns in cls.MUSCLE_CHANGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    result['target_muscle'] = muscle
                    break
            if result['target_muscle']:
                break
        
        return result
#--------------------------------------------------------------------------------------
    
#-----------------------------------------------------------------------------------------------------
# ═══════════════════════════════════════════════════════════════
# ENHANCED STATE MANAGEMENT WITH ULTRA FLEXIBILITY
# ═══════════════════════════════════════════════════════════════
class FlexibleConversationState:
   """Manages ultra-flexible conversation state with free-form transitions"""
  
   STATES = {
       "START": "start",
       "FETCH_PROFILE": "fetch_profile",
       "PROFILE_CONFIRMATION": "profile_confirmation",
       "ASK_DAYS": "ask_days",
       "ASK_NAMES": "ask_names",
       "DRAFT_GENERATION": "draft_generation",
       "EDIT_DECISION": "edit_decision",
       "APPLY_EDIT": "apply_edit",
       "CONFIRM_SAVE": "confirm_save",
       "DONE": "done"
   }
  
   @staticmethod
   def determine_next_state(
       current_state: str,
       user_input: str,
       user_intent: str,
       intent_confidence: float,
       context: Optional[Dict] = None
   ) -> str:
       """Ultra-flexible state determination with context awareness"""
      
       # Global overrides - users can jump to any state anytime
       if user_intent == "create" and intent_confidence > 0.3:
           return FlexibleConversationState.STATES["FETCH_PROFILE"]
       elif user_intent == "show" and intent_confidence > 0.25:
           return "SHOW_TEMPLATE"  # Special handling
       elif user_intent == "edit" and intent_confidence > 0.2:
           return FlexibleConversationState.STATES["APPLY_EDIT"]
      
       # Context-aware state progression
       if current_state == FlexibleConversationState.STATES["START"]:
           return FlexibleConversationState.STATES["FETCH_PROFILE"]
          
       elif current_state == FlexibleConversationState.STATES["FETCH_PROFILE"]:
           return FlexibleConversationState.STATES["ASK_DAYS"]
          
       elif current_state == FlexibleConversationState.STATES["ASK_DAYS"]:
            # Check if user provided day information OR if we already have it from initial input
            extracted_days = UltraFlexibleParser.extract_days_count(user_input)
            context_days = context.get('profile', {}).get('days_count') if context else None
            
            if (extracted_days is not None and extracted_days > 0) or context_days:
                return FlexibleConversationState.STATES["ASK_NAMES"]
            return current_state  # Stay and re-message # Stay and re-message
          
       elif current_state == FlexibleConversationState.STATES["ASK_NAMES"]:
           # Only proceed to generation if user provided actual names (not just days)
           days_keywords = ['day', 'days', 'workout', 'week', 'time']
           is_days_input = any(keyword in user_input.lower() for keyword in days_keywords)

           # Check for "nothing" type responses that should use defaults
           nothing_keywords = ['nothing', 'no', 'skip', 'default', 'defaults', 'normal', 'standard', 'none', 'nope', 'nah']
           is_nothing_response = user_input.strip().lower() in nothing_keywords

           if not is_days_input and (user_input.strip() == "" or len(user_input.strip()) > 2 or is_nothing_response):
               return FlexibleConversationState.STATES["DRAFT_GENERATION"]
           return current_state  # Stay and wait for proper workout names
          
       elif current_state == FlexibleConversationState.STATES["DRAFT_GENERATION"]:
           # Draft generation should complete and wait for user feedback
           return FlexibleConversationState.STATES["EDIT_DECISION"]
          
       elif current_state == FlexibleConversationState.STATES["EDIT_DECISION"]:
        # Check for explicit save commands first
        save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'perfect', 'looks good', 'good to go',
                        'finalize', 'finalize it', 'done', 'ready', 'confirm', 'approved', 'accept', 'yes save',
                        'save template', 'save plan', 'save workout', 'this is good', 'looks great', 'all set']
        if any(cmd in user_input.lower() for cmd in save_commands):
            return FlexibleConversationState.STATES["CONFIRM_SAVE"]
        elif UltraFlexibleParser.is_positive_response(user_input) or user_intent == "edit":
            return FlexibleConversationState.STATES["APPLY_EDIT"]
        elif UltraFlexibleParser.is_negative_response(user_input):
            return FlexibleConversationState.STATES["CONFIRM_SAVE"]
        else:
            # Treat unclear responses as edit requests
            return FlexibleConversationState.STATES["APPLY_EDIT"]
              
       elif current_state == FlexibleConversationState.STATES["APPLY_EDIT"]:
           return FlexibleConversationState.STATES["EDIT_DECISION"]
          
       elif current_state == FlexibleConversationState.STATES["CONFIRM_SAVE"]:
           if UltraFlexibleParser.is_positive_response(user_input):
               return FlexibleConversationState.STATES["DONE"]
           elif UltraFlexibleParser.is_negative_response(user_input):
               return FlexibleConversationState.STATES["EDIT_DECISION"]
           else:
               # Unclear response - treat as edit request
               return FlexibleConversationState.STATES["APPLY_EDIT"]
      
       return current_state
# ═══════════════════════════════════════════════════════════════
# ENHANCED RESPONSE GENERATORS WITH MORE NATURAL LANGUAGE
# ═══════════════════════════════════════════════════════════════
class SmartResponseGenerator:
   """Generates contextual, natural responses for each state"""
  
   PROMPTS = {
       "FETCH_PROFILE": [
           "Let me check your profile to create the perfect workout plan...",
           "Analyzing your fitness goals and experience level...",
           "Getting your profile ready for a personalized workout..."
       ],
      
       "ASK_DAYS": [
            "How many days do you want to work out? You can say '5 days', 'Monday to Friday', or just give your preference — or say nothing to use a default plan.",
            "What's your workout schedule? For example, '6 days a week', 'weekdays only', or any routine you like — or say nothing to use a default plan.",
            "How often do you want to work out? You can say 'Mon-Sat', 'most days', or whatever works for you — or say nothing to use a default plan.",
            "Tell me your workout days! Like '5 times a week', 'daily except Sunday', or 'normal routine' — or say nothing to use a default plan."
            ],
      
       "ASK_NAMES": [
            "What do you want to name your workout days? You can say 'Monday, Tuesday', use muscle groups like 'Push, Pull, Legs', or make your own names — or say 'nothing' or 'default' to use standard day names.",
            "How should I name your workouts? Use normal day names, muscle groups, or any names you like — or say 'skip' or 'default' to use standard day names.",
            "Time to name your workouts! You can keep it simple with days of the week, use body parts like 'Chest, Back, Legs', or make fun custom names — or say 'nothing' to use standard day names.",
            "What names do you prefer for your workouts? Days of the week, muscle groups, or your own creative names are all fine — or say 'default' to use standard day names."
            ],
      
       "EDIT_DECISION": [
           "How does this look? Say 'perfect' or 'looks good' to save it, or just tell me what you'd like to change - I understand natural language!",
           "What do you think of this plan? If it's good to go, just say so! Otherwise, describe any changes you want - like 'more chest work' or 'easier on Monday'.",
           "Ready to save this template? Or would you prefer some adjustments? Just chat naturally about what you'd like different!",
           "This is your personalized plan! Say 'save it' if you're happy, or tell me modifications like 'add cardio' or 'any other change you want' - whatever you need!"
       ],
      
       "CONFIRM_SAVE": [
           "All set to save your workout template? Just say 'yes' or 'save it' to finalize, or 'no' if you want more changes!",
           "Ready to store this plan? Confirm with 'yes', 'go ahead', or 'save' - or let me know if something still needs tweaking!",
           "Should I save this as your workout template? Say anything positive to confirm, or mention if you need more adjustments!",
           "Final check - save this workout plan? A simple 'yes' works, or tell me if there's anything else to modify!"
       ],
      
       "APPLY_EDIT": [
           "What would you like to change? Describe it however feels natural - 'make Monday harder', 'swap bench press for dumbbell press', 'add more leg exercises', etc.",
           "Tell me your modifications! I understand requests like 'more cardio on Friday', 'easier warm-up', 'different exercises for shoulders' - just say it naturally!",
           "What needs adjusting? Whether it's 'increase reps', 'change the order', 'add rest day', or 'make it more challenging' - describe it your way!",
           "How should I modify this? You can request anything - 'less volume', 'different muscle focus', 'swap exercises', or 'make specific days different' - I'll understand!"
       ]
   }
  
   @classmethod
   def get_contextual_prompt(cls, state: str, context: Optional[Dict] = None) -> str:
       """Get a contextual prompt based on state and context"""
       base_prompts = cls.PROMPTS.get(state, ["What would you like to do next?"])
       prompt = secrets.choice(base_prompts)
      
       # Add contextual information
       if context:
           if state == "ASK_DAYS" and context.get('profile'):
               prof = context['profile']
               goal = prof.get('client_goal', 'fitness')
               experience = prof.get('experience', 'beginner')
               context_info = f"Based on your {experience} level and {goal} goal"
               if prof.get('weight_delta_text'):
                   context_info += f" (Target: {prof['weight_delta_text']})"
               prompt = f"{context_info}, {prompt.lower()}"
      
       return prompt
# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (Enhanced)
# ═══════════════════════════════════════════════════════════════
def _evt(payload: Dict[str, Any]) -> str:
   """Enhanced SSE event wrapper with debugging"""
   payload = {
       "msg_id": str(uuid.uuid4()),
       "id": str(uuid.uuid4()),
       "prompt": "",
       "timestamp": str(uuid.uuid4())[:8],
       **payload
   }
   print(f"🚀 Backend event: {payload.get('type', 'unknown')} - {payload.get('status', 'no-status')}")
   return sse_json(payload)
def _fetch_profile(db: Session, client_id: int):
   """Fetch complete client profile including weight journey and calorie targets"""
   try:
       # Get latest weight journey
       w = (
           db.query(WeightJourney)
           .where(WeightJourney.client_id == client_id)
           .order_by(WeightJourney.id.desc())
           .first()
       )

       current_weight = float(w.actual_weight) if w and w.actual_weight is not None else 70.0
       target_weight = float(w.target_weight) if w and w.target_weight is not None else 65.0

       weight_delta_text = None
       goal_type = "maintain"

       if current_weight is not None and target_weight is not None:
           diff = round(target_weight - current_weight, 1)
           if diff > 0:
               weight_delta_text = f"Gain {abs(diff)} kg (from {current_weight} → {target_weight})"
               goal_type = "weight_gain"
           elif diff < 0:
               weight_delta_text = f"Lose {abs(diff)} kg (from {current_weight} → {target_weight})"
               goal_type = "weight_loss"
           else:
               weight_delta_text = f"Maintain {current_weight} kg"
               goal_type = "maintain"

       # Get client details
       c = db.query(Client).where(Client.client_id == client_id).first()
       client_goal = (getattr(c, "goals", None) or getattr(c, "goal", None) or "muscle gain") if c else "muscle gain"
       lifestyle= c.lifestyle if c else "moderate"

       # Get calorie target
       ct = db.query(ClientTarget).where(ClientTarget.client_id == client_id).first()
       target_calories = float(ct.calories) if ct and ct.calories else 2000.0

       return {
           "client_id": client_id,
           "current_weight": current_weight,
           "target_weight": target_weight,
           "weight_delta_text": weight_delta_text,
           "client_goal": client_goal,
           "goal_type": goal_type,
           "target_calories": target_calories,
           "lifestyle": lifestyle,
           "days_per_week": 6,  # Mon–Sat
           "experience": "beginner",  # Default for compatibility
           "profile_complete": True
       }

   except Exception as e:
       print(f"Error fetching profile for client {client_id}: {e}")
       print(f"Profile fetch traceback: {traceback.format_exc()}")
       # Return default profile for testing
       return {
           "client_id": client_id,
           "current_weight": 70.0,
           "target_weight": 65.0,
           "weight_delta_text": "Lose 5.0 kg (from 70.0 → 65.0)",
           "client_goal": "weight loss",
           "goal_type": "weight_loss",
           "target_calories": 1800.0,
           "lifestyle": "moderate",
           "days_per_week": 6,
           "experience": "beginner",
           "profile_complete": False
       }
async def _store_template(mem, db: Session, client_id: int, template: dict, name: str) -> bool:
   """Enhanced template storage with error handling"""
   try:
       # Ensure unique exercise IDs before storage
       template = _ensure_unique_exercise_ids(template)
       id_only = build_id_only_structure(template)
       await mem.r.set(
           f"workout_template:{client_id}",
           orjson.dumps({
               "name": name,
               "template": template,
               "template_ids": id_only,
               "created_at": str(uuid.uuid4())[:8]
           })
       )
       return True
   except Exception as e:
       print(f"Template storage error: {e}")
       return False
   
   ##########################################################
def _validate_template_integrity(template: dict) -> bool:
    """Validate that template has proper structure and isn't empty"""
    if not template or not isinstance(template, dict):
        return False
    
    days = template.get('days', {})
    if not days:
        return False
    
    # Check if at least one day has exercises
    has_exercises = any(
        day.get('exercises') and len(day['exercises']) > 0 
        for day in days.values() 
        if isinstance(day, dict)
    )
    
    return has_exercises
   ##########################################################
async def _get_saved_template(mem, db: Session, client_id: int) -> Optional[Dict[str, Any]]:
   """Enhanced template retrieval with multiple fallbacks"""
   # Try cache first
   try:
       raw = await mem.r.get(f"workout_template:{client_id}")
       if raw:
           obj = orjson.loads(raw)
           if "template" in obj and "template_ids" not in obj:
               # Ensure unique exercise IDs before building structure
               obj["template"] = _ensure_unique_exercise_ids(obj["template"])
               obj["template_ids"] = build_id_only_structure(obj["template"])
           return obj
   except Exception as e:
       print(f"Cache retrieval error: {e}")
   # Try database fallback
   try:
       rec = (
           db.query(WorkoutTemplate)
           .where(WorkoutTemplate.client_id == client_id)
           .order_by(WorkoutTemplate.id.desc())
           .first()
       )
       if rec and getattr(rec, "json", None):
           tpl = orjson.loads(rec.json)
           # Ensure unique exercise IDs before building structure
           tpl = _ensure_unique_exercise_ids(tpl)
           return {
               "name": rec.name,
               "template": tpl,
               "template_ids": build_id_only_structure(tpl)
           }
   except Exception as e:
       print(f"Database retrieval error: {e}")
      
   return None
# ═══════════════════════════════════════════════════════════════
# MAIN ULTRA-FLEXIBLE STREAMING ENDPOINT
# ═══════════════════════════════════════════════════════════════
@router.get("/workout_stream")
async def ultra_flexible_workout_stream(
   user_id: int,
   text: str = Query(...),
   mem = Depends(get_mem),
   oai = Depends(get_oai),
   db: Session = Depends(get_db),
):
   """Ultra-flexible conversational workout template handler"""
  
   if not user_id or not text.strip():
       raise HTTPException(400, "user_id and text required")
   user_input = text.strip()
  
   # Get current context
   pend = (await mem.get_pending(user_id)) or {}
   current_state = pend.get("state", FlexibleConversationState.STATES["START"])


   # Parse user intent with context using AI
   ai_analysis = AIConversationManager.analyze_user_intent(oai, OPENAI_MODEL, user_input, pend)
   user_intent = ai_analysis["intent"]
   intent_confidence = ai_analysis["confidence"]

   # Use AI to determine conversation flow
   flow_decision = AIConversationManager.determine_conversation_flow(
       oai, OPENAI_MODEL, user_input, current_state, pend
   )
   next_state = flow_decision["next_state"]
  
   print(f"🤖 Ultra-flexible transition: {current_state} → {next_state} (intent: {user_intent}, conf: {intent_confidence:.2f})")
   print(f"🔍 DEBUG - User input: '{user_input}', Current State: '{current_state}', Next State: '{next_state}'")
   print(f"🔍 DEBUG - Checking state conditions:")
   print(f"  - current_state == DRAFT_GENERATION: {current_state == FlexibleConversationState.STATES['DRAFT_GENERATION']}")
   print(f"  - next_state == 'DRAFT_GENERATION': {next_state == 'DRAFT_GENERATION'}")
   print(f"  - next_state == 'draft_generation': {next_state == 'draft_generation'}")
   print(f"  - FlexibleConversationState.STATES['DRAFT_GENERATION']: '{FlexibleConversationState.STATES['DRAFT_GENERATION']}'")
   print(f"  - Actual next_state value: '{next_state}'")

   # Skip processing if no real state change (avoid duplicate processing)
   if current_state == next_state and current_state != FlexibleConversationState.STATES["START"]:
       async def _no_change():
           yield "event: done\ndata: [DONE]\n\n"
       return StreamingResponse(_no_change(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # SHOW TEMPLATE - Can be accessed from anywhere
   if next_state == "SHOW_TEMPLATE":
       saved = await _get_saved_template(mem, db, user_id)

       if saved and saved.get("template", {}).get("days"):
           tpl = saved["template"]
           md = render_markdown_from_template(tpl)
           tpl_ids = saved.get("template_ids") or build_id_only_structure(tpl)

           async def _show_saved():
               yield _evt({
                   "type": "workout_template",
                   "status": "fetched",
                   "template_markdown": md,
                   "template_json": tpl,
                   "template_ids": tpl_ids,
                   "message": "🎉 Here's your saved workout template! 🎉"
               })
               yield _evt({
                   "type": "workout_template",
                   "status": "edit_decision",
                   "message": "What would you like to do with this template? You can edit it, create a new one, or save changes."
               })
               yield "event: done\ndata: [DONE]\n\n"
           return StreamingResponse(_show_saved(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
       else:
           async def _no_template():
               yield _evt({
                   "type": "workout_template",
                   "status": "hint",
                   "message": "🎯 Ready to create your first workout template?\n\n💪 Say 'make me a workout plan' or 'create template'\n🚀 Let's build something amazing together!\n\n✨ I'll guide you through every step!"
               })
               yield "event: done\ndata: [DONE]\n\n"
           return StreamingResponse(_no_template(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # START STATE - Show profile for ANY first message
   if current_state == FlexibleConversationState.STATES["START"]:
        async def _start_with_profile():
            prof = _fetch_profile(db, user_id)
            print(f"🔍 DEBUG - Fetched profile for START state client {user_id}: {prof}")

            # Format profile information for display
            profile_info = []
            profile_info.append(f"💪 Goal: {prof.get('client_goal', 'muscle gain')}")
            profile_info.append(f"📈 Experience: {prof.get('experience', 'beginner')}")
            profile_info.append(f"🏋️ Current Weight: {prof.get('current_weight', 70.0)} kg")
            profile_info.append(f"🎯 Target Weight: {prof.get('target_weight', 65.0)} kg")

            if prof.get("weight_delta_text"):
                profile_info.append(f"📊 Progress Goal: {prof['weight_delta_text']}")
            if prof.get("lifestyle"):
                profile_info.append(f"🏃 Lifestyle: {prof['lifestyle']}")
            if prof.get("target_calories"):
                profile_info.append(f"🔥 Daily Calorie Target: {prof['target_calories']} kcal")

            profile_display = "\n".join(profile_info)

            # Set state to profile confirmation since we're showing profile
            await mem.set_pending(user_id, {
                "state": "PROFILE_CONFIRMATION",
                "profile": prof
            })

            message = f"Hi! I'm your workout template assistant. Here's your current profile:\n\n{profile_display}\n\nWould you like me to create a workout plan based on this profile, or do you have any specific preferences or changes you'd like to make first?"
            print(f"🔍 DEBUG - START state message: {message}")

            yield _evt({
                "type": "workout_template",
                "status": "profile_shown",
                "message": message,
                "profile_data": prof
            })
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(_start_with_profile(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # FETCH_PROFILE STATE - Show existing profile and ask for confirmation
   elif next_state == FlexibleConversationState.STATES["FETCH_PROFILE"]:
        async def _fetch_and_show_profile():
            prof = _fetch_profile(db, user_id)
            print(f"🔍 DEBUG - Fetched profile for client {user_id}: {prof}")

            # Format profile information for display
            profile_info = []

            # Always show goal and experience (they have defaults)
            profile_info.append(f"💪 Goal: {prof.get('client_goal', 'muscle gain')}")
            profile_info.append(f"📈 Experience: {prof.get('experience', 'beginner')}")

            # Add weight info - always available now with defaults
            profile_info.append(f"🏋️ Current Weight: {prof.get('current_weight', 70.0)} kg")
            profile_info.append(f"🎯 Target Weight: {prof.get('target_weight', 65.0)} kg")

            # Add weight delta if available
            if prof.get("weight_delta_text"):
                profile_info.append(f"📊 Progress Goal: {prof['weight_delta_text']}")

            # Add additional profile info
            if prof.get("lifestyle"):
                profile_info.append(f"🏃 Lifestyle: {prof['lifestyle']}")

            if prof.get("target_calories"):
                profile_info.append(f"🔥 Daily Calorie Target: {prof['target_calories']} kcal")

            profile_display = "\n".join(profile_info)

            # Use AI to generate a natural response asking for confirmation
            try:
                ai_response = AIConversationManager.generate_contextual_response(
                    oai, OPENAI_MODEL, "PROFILE_CONFIRMATION",
                    f"Show user profile and ask if they want workout based on this: {profile_display}",
                    {"profile": prof}
                )
            except:
                ai_response = "Would you like me to create a workout plan based on this profile, or would you like to modify anything first?"

            # Set state to ask for template creation confirmation
            await mem.set_pending(user_id, {
                "state": "PROFILE_CONFIRMATION",
                "profile": prof
            })

            message = f"Here's your current profile:\n\n{profile_display}\n\n{ai_response}"
            print(f"🔍 DEBUG - Profile message being sent: {message}")

            yield _evt({
                "type": "workout_template",
                "status": "profile_shown",
                "message": message,
                "profile_data": prof
            })
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(_fetch_and_show_profile(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # PROFILE_CONFIRMATION STATE - Handle user response to profile display
   elif current_state == "PROFILE_CONFIRMATION":
       # Simple rule-based detection for common responses
       positive_words = ['yes', 'ok', 'okay', 'sure', 'profile', 'based', 'good', 'fine', 'create', 'go']
       user_lower = user_input.lower()
       is_positive = any(word in user_lower for word in positive_words)

       if is_positive:
           # User confirmed - proceed to ask for days
           prof = pend.get("profile", {})
           await mem.set_pending(user_id, {
               "state": FlexibleConversationState.STATES["ASK_DAYS"],
               "profile": prof
           })

           async def _proceed_to_days():
               yield _evt({
                   "type": "workout_template",
                   "status": "ask_days",
                   "message": "Great! How many days per week do you want to work out? (e.g., 3 days, 5 days, 6 days)"
               })
               yield "event: done\ndata: [DONE]\n\n"

           return StreamingResponse(_proceed_to_days(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

       else:
           # Ask for clarification
           async def _ask_clarification():
               yield _evt({
                   "type": "workout_template",
                   "status": "ask_clarification",
                   "message": "Would you like me to create a workout template based on your profile? Please let me know yes or no."
               })
               yield "event: done\ndata: [DONE]\n\n"

           return StreamingResponse(_ask_clarification(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # ASK_DAYS STATE - User provides number of days
   elif current_state == FlexibleConversationState.STATES["ASK_DAYS"]:
       # Simple rule-based days extraction
       import re
       days_match = re.search(r'\b(\d+)\s*(?:days?|workouts?)\b', user_input.lower())
       if days_match:
           days_count = int(days_match.group(1))
       else:
           # Try to find standalone numbers
           number_match = re.search(r'\b([3-7])\b', user_input)
           days_count = int(number_match.group(1)) if number_match else 5
       prof = pend.get("profile", {})
       prof["days_count"] = days_count

       # Move to ASK_NAMES state
       await mem.set_pending(user_id, {
           "state": FlexibleConversationState.STATES["ASK_NAMES"],
           "profile": prof
       })

       

       async def _process_days():
           yield _evt({
               "type": "workout_template",
               "status": "ask_names",
               "message": f"Perfect! {days_count} workout days it is.\n\nDo you want to name your workout days? You can use day names (Monday, Tuesday...) or muscle groups (Push, Pull, Legs) or say 'default' to use standard names."
           })
           yield "event: done\ndata: [DONE]\n\n"

       return StreamingResponse(_process_days(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # ASK_NAMES & DRAFT_GENERATION STATE - Combined for immediate execution
   elif current_state == FlexibleConversationState.STATES["ASK_NAMES"] or current_state == FlexibleConversationState.STATES["DRAFT_GENERATION"] or next_state == "DRAFT_GENERATION":
       print(f"🎯 ENTERING TEMPLATE GENERATION!")
       prof = pend.get("profile", {})

       # If coming from ASK_NAMES, process the names first
       if current_state == FlexibleConversationState.STATES["ASK_NAMES"]:
           days_count = prof.get("days_count", 5)

           # Extract names from user input or use defaults
           if "default" in user_input.lower():
               day_names = [f"Day {i+1}" for i in range(days_count)]
           else:
               # Try to extract names from user input
               extracted_names = ai_analysis.get("day_names", [])
               if extracted_names and len(extracted_names) >= days_count:
                   day_names = extracted_names[:days_count]
               else:
                   # Use standard day names
                   standard_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                   day_names = standard_days[:days_count]

           prof["template_names"] = day_names
           prof["template_count"] = len(day_names)

       async def _generate_template():
           try:
               print("🔍 DEBUG - Starting template generation")

               # Send generating status
               yield _evt({
                   "type": "workout_template",
                   "status": "generating",
                   "message": "Creating your personalized workout template..."
               })

               # Ensure profile has required fields
               if "template_names" not in prof:
                   prof["template_names"] = [f"Day {i+1}" for i in range(prof.get("days_count", 5))]
               if "template_count" not in prof:
                   prof["template_count"] = len(prof["template_names"])

               print(f"🔍 DEBUG - Profile ready: {prof.get('template_names', [])}")

               # Generate template
               print("🔍 DEBUG - Calling template generation")
               try:
                   tpl, why = llm_generate_template_from_profile(oai, OPENAI_MODEL, prof, db)
                   print(f"🔍 DEBUG - Template generated: {type(tpl)}")
               except Exception as gen_error:
                   print(f"🚨 Generation failed: {gen_error}")
                   # Create fallback template
                   template_names = prof.get("template_names", ["Day 1"])
                   tpl = {
                       "name": f"Fallback Workout ({len(template_names)} days)",
                       "goal": "muscle_gain",
                       "days": {},
                       "notes": []
                   }
                   for name in template_names:
                       day_key = name.lower()
                       tpl["days"][day_key] = {
                           "title": name.title(),
                           "muscle_groups": ["full body"],
                           "exercises": [
                               {"name": "Push-ups", "sets": 3, "reps": 10},
                               {"name": "Squats", "sets": 3, "reps": 12},
                               {"name": "Plank", "sets": 3, "reps": "30 seconds"}
                           ]
                       }

               # Process template for display
               print("🔍 DEBUG - Processing template for display")
               tpl = _ensure_unique_exercise_ids(tpl)
               md = render_markdown_from_template(tpl)
               tpl_ids = build_id_only_structure(tpl)

               # Update state
               await mem.set_pending(user_id, {
                   "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                   "profile": prof,
                   "template": tpl
               })

               # Return success response with template in message field for frontend display
               yield _evt({
                   "type": "workout_template",
                   "status": "draft",
                   "template_markdown": md,
                   "template_json": tpl,
                   "template_ids": tpl_ids,
                   "why": "Generated based on your profile",
                   "message": f"Here's your personalized workout template:\n\n{md}"
               })

               # Add the follow-up question like the working version
               yield _evt({
                   "type": "workout_template",
                   "status": "ask_edit_q",
                   "ask": "How does this look? Say 'save it' if you're happy, or tell me what you'd like to change!"
               })

           except Exception as e:
               print(f"❌ Template generation error: {e}")
               import traceback
               traceback.print_exc()

               # Send error response
               yield _evt({
                   "type": "workout_template",
                   "status": "error",
                   "message": "I had trouble generating your workout template. This might be due to a temporary issue. Would you like to try again?"
               })

           yield "event: done\ndata: [DONE]\n\n"

       return StreamingResponse(_generate_template(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # EDIT_DECISION STATE - User decides to edit or save template
   elif current_state == FlexibleConversationState.STATES["EDIT_DECISION"]:
       prof = pend.get("profile", {})
       tpl = pend.get("template", {})

       # Check if user wants to save
       save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'perfect', 'looks good', 'good to go',
                       'finalize', 'finalize it', 'done', 'ready', 'confirm', 'approved', 'accept', 'yes save',
                       'save template', 'save plan', 'save workout', 'this is good', 'looks great', 'all set']
       if any(cmd in user_input.lower() for cmd in save_commands):
           # User wants to save - move to CONFIRM_SAVE
           await mem.set_pending(user_id, {
               "state": FlexibleConversationState.STATES["CONFIRM_SAVE"],
               "profile": prof,
               "template": tpl
           })

           async def _confirm_save():
               yield _evt({
                   "type": "workout_template",
                   "status": "confirm_save",
                   "message": "Perfect! Are you sure you want to save this workout template?"
               })
               yield "event: done\ndata: [DONE]\n\n"

           return StreamingResponse(_confirm_save(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

       else:
           # Check if user input contains clear edit instructions
           clear_edit_patterns = [
               # Day name changes
               'change', 'rename', 'call it', 'make it', 'name it',
               # Exercise modifications
               'add', 'remove', 'replace', 'substitute', 'swap', 'include', 'delete',
               # Adjustments
               'increase', 'decrease', 'more', 'less', 'easier', 'harder',
               # Specific instructions
               'from', 'to', 'instead of', 'with', 'for'
           ]

           user_lower = user_input.lower()
           has_clear_instruction = any(pattern in user_lower for pattern in clear_edit_patterns)

           # Also check for specific structures like "X to Y" or "change X"
           has_specific_structure = (
               ' to ' in user_lower or
               ' from ' in user_lower or
               'change ' in user_lower or
               'rename ' in user_lower or
               'add ' in user_lower or
               'remove ' in user_lower
           )

           if has_clear_instruction or has_specific_structure:
               # User gave clear instructions - apply edit directly without asking
               print(f"🎯 Clear edit instruction detected: {user_input}")

               async def _apply_direct_edit():
                   try:
                       # Apply the edit using the same logic as APPLY_EDIT state
                       from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import enhanced_edit_template

                       new_tpl, summary = enhanced_edit_template(oai, OPENAI_MODEL, tpl, user_input, prof, db)

                       # Ensure unique exercise IDs before rendering
                       new_tpl = _ensure_unique_exercise_ids(new_tpl)
                       md = render_markdown_from_template(new_tpl)
                       tpl_ids = build_id_only_structure(new_tpl)

                       await mem.set_pending(user_id, {
                           "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                           "profile": prof,
                           "template": new_tpl
                       })

                       yield _evt({
                           "type": "workout_template",
                           "status": "draft",
                           "template_markdown": md,
                           "template_json": new_tpl,
                           "template_ids": tpl_ids,
                           "message": f"Great! I've made that change:\n\n{md}",
                           "why": summary or "Applied your requested change"
                       })

                       yield _evt({
                           "type": "workout_template",
                           "status": "ask_edit_q",
                           "ask": "How does this look now? Say 'save it' if you're happy, or tell me what else you'd like to change!"
                       })

                   except Exception as e:
                       print(f"Direct edit error: {e}")
                       yield _evt({
                           "type": "workout_template",
                           "status": "error",
                           "message": "I had trouble making that change. Could you try describing it differently?"
                       })

                   yield "event: done\ndata: [DONE]\n\n"

               return StreamingResponse(_apply_direct_edit(), media_type="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

           else:
               # User input is unclear - ask for clarification
               await mem.set_pending(user_id, {
                   "state": FlexibleConversationState.STATES["APPLY_EDIT"],
                   "profile": prof,
                   "template": tpl
               })

               async def _ask_for_edits():
                   yield _evt({
                       "type": "workout_template",
                       "status": "ask_for_edits",
                       "message": "What would you like to change? You can say things like:\n• 'Change day name from Monday to Push Day'\n• 'Add more chest exercises'\n• 'Remove squats and add lunges'\n• 'Make it easier'"
                   })
                   yield "event: done\ndata: [DONE]\n\n"

               return StreamingResponse(_ask_for_edits(), media_type="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # APPLY_EDIT STATE
   elif current_state == FlexibleConversationState.STATES["APPLY_EDIT"]:
    prof = pend.get("profile", {})
    tpl = pend.get("template")

    # If no current template, try to get saved one
    if not tpl:
        saved = await _get_saved_template(mem, db, user_id)
        if saved:
            tpl = saved.get("template", {})
            prof = prof or {}
        else:
            async def _need_template():
                yield _evt({
                    "type": "workout_template",
                    "status": "hint",
                    "message": "🎯 I need a template to edit first!\n\n🆕 Say 'create template' to make a new one\n📋 Say 'show template' if you have one saved\n💪 Let's get your workout ready!"
                })
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_need_template(), media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _apply_edit():
        try:
            # Validate exercises using AI before editing
            exercise_validation = AIConversationManager.validate_and_map_exercises(oai, OPENAI_MODEL, user_input, db)

            # Call enhanced edit function with validated exercises
            new_tpl, summary = enhanced_edit_template(oai, OPENAI_MODEL, tpl, user_input, prof, db)

            # Ensure unique exercise IDs before rendering
            new_tpl = _ensure_unique_exercise_ids(new_tpl)
            md = render_markdown_from_template(new_tpl)
            tpl_ids = build_id_only_structure(new_tpl)

            await mem.set_pending(user_id, {
                "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                "profile": prof,
                "template": new_tpl
            })

            yield _evt({
                "type": "workout_template",
                "status": "edited",
                "template_markdown": md,
                "template_json": new_tpl,
                "template_ids": tpl_ids,
                "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
            })

        except Exception as e:
            print(f"Enhanced edit error: {e}")
            yield _evt({
                "type": "workout_template",
                "status": "error",
                "message": "I had trouble making that change. Could you try describing it differently?"
            })

        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(_apply_edit(), media_type="text/event-stream",
                           headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # CONFIRM_SAVE STATE
   elif current_state == FlexibleConversationState.STATES["CONFIRM_SAVE"]:
    prof = pend.get("profile", {})
    tpl = pend.get("template", {})

    if ai_analysis.get("positive_sentiment") or user_intent in ["save", "yes"] or user_input.lower().strip() in ['save', 'yes', 'confirm']:
        # Save the template
        template_name = tpl.get("name") or f"Custom Workout ({', '.join(prof.get('template_names', ['Multi-Day']))})"

        async def _final_save():
            # Validate template before saving
            if not _validate_template_integrity(tpl):
                yield _evt({
                    "type": "workout_template",
                    "status": "error",
                    "message": "The template appears to be corrupted or empty. Let me help you create a new one. Say 'create template' to start fresh."
                })
                yield "event: done\ndata: [DONE]\n\n"
                return

            success = await _store_template(mem, db, user_id, tpl, template_name)

            if success:
                # Save structured template using workout_structured functions
                try:
                    # Ensure template has exercise IDs before saving
                    tpl_with_ids = await _ensure_template_has_ids(tpl, db)
                    if not tpl_with_ids:
                        yield _evt({
                            "type": "workout_template",
                            "status": "error",
                            "message": "❌ Unable to assign exercise IDs to template. Please try creating the template again."
                        })
                        yield "event: done\ndata: [DONE]\n\n"
                        return

                    # 1) Collect ids by day and fetch all rows in one shot
                    per_day_ids = _gather_ids(tpl_with_ids)
                    all_ids = [eid for ids in per_day_ids.values() for eid in ids]
                    id_map = _fetch_qr_rows(db, all_ids)

                    # 2) Build payload per day and upsert
                    results = []
                    for day in per_day_ids.keys():
                        day_ids = per_day_ids.get(day, [])
                        if not day_ids:
                            continue

                        payload = _build_day_payload(day_ids, id_map)

                        # Validate payload is not empty
                        if not payload or len(payload) == 0:
                            continue

                        results.append(_persist_payload(db, user_id, day, payload))

                    # 3) Commit the changes
                    if results:
                        db.commit()
                        print(f"✅ Successfully saved structured template for client {user_id} with {len(results)} days")

                        # Clear pending state after successful save
                        await mem.clear_pending(user_id)

                        yield _evt({
                            "type": "workout_template",
                            "status": "saved",
                            "message": f"🎉 Successfully saved your '{template_name}' workout template!\n\n✅ Your personalized plan is ready to use anytime.\n🚀 Ready to start your fitness journey!"
                        })
                    else:
                        db.rollback()
                        yield _evt({
                            "type": "workout_template",
                            "status": "error",
                            "message": "❌ Unable to save template - no valid exercise data found. This might be due to missing exercise IDs or database connectivity issues. The template was saved in memory but not in the structured database."
                        })
                except Exception as e:
                    db.rollback()
                    print(f"🚨 Failed to save structured template: {e}")
                    yield _evt({
                        "type": "workout_template",
                        "status": "error",
                        "message": "Sorry, there was an issue saving your template to the database. The template was saved in chat memory, but the structured version failed. Please try again!"
                    })
            else:
                yield _evt({
                    "type": "workout_template",
                    "status": "error",
                    "message": "Sorry, there was an issue saving your template. Please try again!"
                })

            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(_final_save(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    elif ai_analysis.get("negative_sentiment") or user_intent == "no":
        # Go back to editing
        await mem.set_pending(user_id, {
            "state": FlexibleConversationState.STATES["EDIT_DECISION"],
            "profile": prof,
            "template": tpl
        })

        async def _back_to_edit():
            yield _evt({
                "type": "workout_template",
                "status": "ask_edit_decision",
                "message": "No problem! " + SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
            })
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(_back_to_edit(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # ULTIMATE FALLBACK - AI-powered context-aware responses
   async def _ultra_smart_fallback():
       # AI-powered context-aware fallback responses
       try:
           ai_response = AIConversationManager.generate_contextual_response(
               oai, OPENAI_MODEL, current_state, user_input, pend
           )
           yield _evt({
               "type": "workout_template",
               "status": "hint",
               "message": ai_response
           })
       except Exception as e:
           print(f"AI response generation failed: {e}")
           # Fallback to simple response
           if not pend:
               message = "I'm your workout template assistant! I can help you create personalized plans, show existing templates, or make edits. Just tell me what you need - like 'make me a workout', 'show my plan', or 'change my routine'. What sounds good?"
           elif current_state == FlexibleConversationState.STATES["START"]:
               # Show profile immediately in START state
               prof = _fetch_profile(db, user_id)
               print(f"🔍 DEBUG - Fetched profile in START fallback for client {user_id}: {prof}")

               profile_info = []
               profile_info.append(f"💪 Goal: {prof.get('client_goal', 'muscle gain')}")
               profile_info.append(f"📈 Experience: {prof.get('experience', 'beginner')}")
               profile_info.append(f"🏋️ Current Weight: {prof.get('current_weight', 70.0)} kg")
               profile_info.append(f"🎯 Target Weight: {prof.get('target_weight', 65.0)} kg")

               if prof.get("weight_delta_text"):
                   profile_info.append(f"📊 Progress Goal: {prof['weight_delta_text']}")
               if prof.get("lifestyle"):
                   profile_info.append(f"🏃 Lifestyle: {prof['lifestyle']}")
               if prof.get("target_calories"):
                   profile_info.append(f"🔥 Daily Calorie Target: {prof['target_calories']} kcal")

               profile_display = "\n".join(profile_info)
               message = f"Hi! I'm your workout template assistant. Here's your current profile:\n\n{profile_display}\n\nWould you like me to create a workout plan based on this profile, or do you have any specific preferences?"

               print(f"🔍 DEBUG - START fallback message: {message}")
           else:
               message = "I didn't quite catch that, but I'm here to help! You can describe what you want naturally - like 'yes', 'no', 'change this exercise', 'make it harder', or tell me exactly what you're thinking. What would you like to do?"

           yield _evt({
               "type": "workout_template",
               "status": "hint",
               "message": message
           })

       yield "event: done\ndata: [DONE]\n\n"

   return StreamingResponse(_ultra_smart_fallback(), media_type="text/event-stream",
                          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
