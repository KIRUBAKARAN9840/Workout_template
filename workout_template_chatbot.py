from __future__ import annotations
import os, orjson, uuid, re, secrets
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import enhanced_edit_template
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import SmartWorkoutEditor
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_llm_helper import extract_bulk_operation_info
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.workout_structured import StructurizeAndSaveRequest, structurize_and_save

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
from app.models.fittbot_models import Client, WeightJourney, WorkoutTemplate


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
        save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it']
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
        save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'perfect', 'looks good', 'good to go']
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
def _fetch_profile(db: Session, client_id: int) -> Dict[str, Any]:
   """Enhanced profile fetching with fallbacks"""
   try:
       w = (
           db.query(WeightJourney)
           .where(WeightJourney.client_id == client_id)
           .order_by(WeightJourney.id.desc())
           .first()
       )
       current_weight = float(w.actual_weight) if w and w.actual_weight is not None else None
       target_weight = float(w.target_weight) if w and w.target_weight is not None else None
       weight_delta_text = None
       if current_weight is not None and target_weight is not None:
           diff = round(target_weight - current_weight, 1)
           if diff > 0:
               weight_delta_text = f"Gain {abs(diff)} kg (from {current_weight} → {target_weight})"
           elif diff < 0:
               weight_delta_text = f"Lose {abs(diff)} kg (from {current_weight} → {target_weight})"
           else:
               weight_delta_text = f"Maintain {current_weight} kg"
       c = db.query(Client).where(Client.client_id == client_id).first()
       goal = (getattr(c, "goals", None) or getattr(c, "goal", None) or "muscle gain") if c else "muscle gain"
       experience = (getattr(c, "experience", None) or "beginner") if c else "beginner"
       return {
           "current_weight": current_weight,
           "target_weight": target_weight,
           "weight_delta_text": weight_delta_text,
           "client_goal": goal,
           "experience": experience,
           "profile_complete": True
       }
   except Exception as e:
       print(f"Profile fetch error: {e}")
       return {
           "current_weight": None,
           "target_weight": None,
           "weight_delta_text": None,
           "client_goal": "muscle gain",
           "experience": "beginner",
           "profile_complete": False
       }
async def _store_template(mem, db: Session, client_id: int, template: dict, name: str) -> bool:
   """Enhanced template storage with error handling"""
   try:
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


   # Parse user intent with context
   user_intent, intent_confidence = UltraFlexibleParser.extract_intent(user_input, pend)

   # Determine next state
   next_state = FlexibleConversationState.determine_next_state(
       current_state, user_input, user_intent, intent_confidence, pend
   )
  
   print(f"🤖 Ultra-flexible transition: {current_state} → {next_state} (intent: {user_intent}, conf: {intent_confidence:.2f})")

   # Skip processing if no real state change (avoid duplicate processing)
   if current_state == next_state and current_state != FlexibleConversationState.STATES["START"]:
       async def _no_change():
           yield "event: done\ndata: [DONE]\n\n"
       return StreamingResponse(_no_change(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

   # ═══════════════════════════════════════════════════════════════
   # STATE HANDLERS WITH ULTRA-FLEXIBILITY
   # ═══════════════════════════════════════════════════════════════
  
   # SHOW TEMPLATE - Can be accessed from anywhere
   if next_state == "SHOW_TEMPLATE" or (user_intent == "show" and intent_confidence > 0.25):
       saved = await _get_saved_template(mem, db, user_id)
       if saved:
           tpl = saved.get("template", {})
           md = render_markdown_from_template(tpl)
           tpl_ids = saved.get("template_ids", build_id_only_structure(tpl))
          
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
                   "status": "hint",
                   "message": "✨ Want to customize your workout?\n\n🔧 Tell me what to change (e.g., 'add more chest exercises')\n🆕 Say 'create new template' to start fresh\n💬 I'm here to help with whatever you need!"
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
  
   # FETCH_PROFILE STATE
   if next_state == FlexibleConversationState.STATES["FETCH_PROFILE"]:
        import asyncio
        
        # Extract comprehensive info from the initial request
        workout_info = UltraFlexibleParser.extract_comprehensive_workout_info(user_input)

        async def _fetch_and_proceed():
            prof = _fetch_profile(db, user_id)
                    
            # CRITICAL FIX: Only proceed with days info if we actually detected it
            if workout_info['is_muscle_specific_template'] and workout_info['muscle_focus'] and workout_info['has_days_info']:
                prof['days_count'] = workout_info['days_count']
                
                # Use proper day names instead of "Day 1", "Day 2"
                if workout_info['days_count'] <= 7:
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    prof['template_names'] = day_names[:workout_info['days_count']]
                else:
                    prof['template_names'] = [f"Day {i+1}" for i in range(workout_info['days_count'])]
                
                prof['days_per_week'] = workout_info['days_count']
                prof['muscle_focus'] = workout_info['muscle_focus']
                
                # Skip directly to generation with muscle focus
                await mem.set_pending(user_id, {
                    "state": FlexibleConversationState.STATES["DRAFT_GENERATION"],
                    "profile": prof
                })
                
                
                # ... rest of muscle-specific generation logic
                
            else:
                # FIXED: No days info detected - ask for it
                await mem.set_pending(user_id, {
                    "state": FlexibleConversationState.STATES["ASK_DAYS"],
                    "profile": prof
                })

                yield _evt({
                    "type": "workout_template",
                    "status": "ask_days",
                    "message": SmartResponseGenerator.get_contextual_prompt("ASK_DAYS", {"profile": prof})
                })
            
        return StreamingResponse(_fetch_and_proceed(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # ASK_DAYS STATE 
   elif next_state == FlexibleConversationState.STATES["ASK_NAMES"]:
       days_count = UltraFlexibleParser.extract_days_count(user_input)
       prof = pend.get("profile", {})
       prof["days_count"] = days_count
      
       await mem.set_pending(user_id, {
           "state": FlexibleConversationState.STATES["ASK_NAMES"],
           "profile": prof
       })
      
       async def _ask_names():
           yield _evt({
               "type": "workout_template",
               "status": "ask_names",
               "message": f"🎯 Perfect - {days_count} workout days it is! 🎯\n\n" + SmartResponseGenerator.get_contextual_prompt("ASK_NAMES")
           })
           yield "event: done\ndata: [DONE]\n\n"
          
       return StreamingResponse(_ask_names(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # DRAFT_GENERATION STATE
   elif next_state == FlexibleConversationState.STATES["DRAFT_GENERATION"]:
    prof = pend.get("profile", {})
    days_count = prof.get("days_count", 6)
    template_names = UltraFlexibleParser.extract_template_names(user_input, days_count)
    
    # FIXED: Better muscle detection for single-word inputs
    detected_muscle = None
    user_input_clean = user_input.lower().strip()
    
    # Direct muscle mappings for common single-word requests
    direct_muscle_map = {
        'leg': 'legs',
        'legs': 'legs', 
        'chest': 'chest',
        'back': 'back',
        'arm': 'upper',
        'arms': 'upper',
        'shoulder': 'shoulders',
        'shoulders': 'shoulders',
        'core': 'core',
        'abs': 'core',
        'upper': 'upper',
        'lower': 'legs'
    }
    
    # Check if the entire input is just a muscle group
    if user_input_clean in direct_muscle_map:
        detected_muscle = direct_muscle_map[user_input_clean]
        print(f"🎯 Direct muscle detection: '{user_input_clean}' -> {detected_muscle}")
    
    # If not direct match, check template names for muscle groups
    if not detected_muscle:
        muscle_keywords = {
            'leg': 'legs', 'legs': 'legs', 
            'chest': 'chest', 
            'back': 'back',
            'arm': 'upper', 'arms': 'upper',
            'shoulder': 'shoulders', 'shoulders': 'shoulders',
            'core': 'core', 'abs': 'core',
            'upper': 'upper', 'lower': 'legs'
        }
        
        for name in template_names:
            name_lower = name.lower().strip()
            if name_lower in muscle_keywords:
                detected_muscle = muscle_keywords[name_lower]
                print(f"🎯 Template name muscle detection: {name_lower} -> {detected_muscle}")
                break
    
    if detected_muscle:
        # FIXED: Use SmartWorkoutEditor directly for muscle-specific templates
        prof["muscle_focus"] = detected_muscle
        
        # Use proper day names instead of just "monday"
        if days_count <= 7:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            prof['template_names'] = day_names[:days_count]
        else:
            prof['template_names'] = [f"Day {i+1}" for i in range(days_count)]
        
        prof['days_per_week'] = days_count

        async def _generate_muscle_specific():
            import asyncio
            
            try:
                # CRITICAL FIX: Use SmartWorkoutEditor directly instead of LLM
                muscle_distributions = {detected_muscle: days_count}
                tpl, why = SmartWorkoutEditor.create_muscle_specific_template(
                    prof['template_names'], 
                    muscle_distributions, 
                    db
                )
                
                # Validate that we got appropriate exercises
                if tpl and tpl.get('days'):
                    first_day_key = list(tpl['days'].keys())[0]
                    first_day = tpl['days'][first_day_key]
                    exercises = first_day.get('exercises', [])
                    
                    if exercises:
                        exercise_names = [ex.get('name', '').lower() for ex in exercises]
                        print(f"✅ Generated {detected_muscle} exercises: {exercise_names}")
                        
                        # Validate muscle match
                        validation = SmartWorkoutEditor.validate_exercise_match(detected_muscle, exercises)
                        print(f"📊 Validation: {validation['message']}")
                        
                        if not validation['valid']:
                            print(f"❌ Validation failed - regenerating with database approach")
                            # Force regeneration with direct database query
                            tpl, why = SmartWorkoutEditor.create_muscle_specific_template(
                                ["monday"], 
                                {detected_muscle: 1}, 
                                db
                            )
                else:
                    print(f"❌ No template generated, using fallback")
                    tpl = {"name": f"{detected_muscle.title()} Workout", "goal": "muscle_gain", "days": {}, "notes": []}
                    why = f"Could not generate {detected_muscle} workout"
                
                # Fix day keys to match expected format
                if tpl and 'days' in tpl and prof.get('template_names'):
                    new_days = {}
                    template_names = prof['template_names']
                    
                    old_keys = list(tpl['days'].keys())
                    for i, old_key in enumerate(old_keys):
                        if i < len(template_names):
                            # Use custom template name as day key
                            new_key = template_names[i].lower().replace(' ', '_')
                            new_days[new_key] = tpl['days'][old_key]
                            # Update title to use custom name
                            new_days[new_key]['title'] = f"{template_names[i]} — {detected_muscle.title()} Day"
                        else:
                            new_days[old_key] = tpl['days'][old_key]
                    tpl['days'] = new_days
                
                md = render_markdown_from_template(tpl)
                tpl_ids = build_id_only_structure(tpl)
                
                await mem.set_pending(user_id, {
                    "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                    "profile": prof,
                    "template": tpl
                })
                
                yield _evt({
                    "type": "workout_template",
                    "status": "draft",
                    "template_markdown": md,
                    "template_json": tpl,
                    "template_ids": tpl_ids,
                    "message": _format_template_for_display(tpl), 
                    "why": why or f"{days_count}-day {detected_muscle} focused template created!"
                })
                
                yield _evt({
                    "type": "workout_template",
                    "status": "ask_edit_decision",
                    "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
                })
                
            except Exception as e:
                print(f"Muscle-specific generation error: {e}")
                yield _evt({
                    "type": "workout_template",
                    "status": "error",
                    "message": "Had trouble generating your leg workout. Want to try again with a different approach?"
                })
            
            yield "event: done\ndata: [DONE]\n\n"
        
        return StreamingResponse(_generate_muscle_specific(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    else:
        # Continue with normal template generation for non-muscle-specific requests
        prof["template_names"] = template_names
        prof["days_per_week"] = days_count

  
   # APPLY_EDIT STATE
   elif next_state == FlexibleConversationState.STATES["APPLY_EDIT"]:
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
            # Check for bulk operations - using imported function
            bulk_info = extract_bulk_operation_info(user_input)  # Use imported function
            
            
            # Call enhanced edit function (rest remains the same)
            new_tpl, summary = enhanced_edit_template(oai, OPENAI_MODEL, tpl, user_input, prof, db)
            
            # Rest of the validation and response logic remains the same...
            md = render_markdown_from_template(new_tpl)
            tpl_ids = build_id_only_structure(new_tpl)
            
            await mem.set_pending(user_id, {
                "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                "profile": prof,
                "template": new_tpl
            })
            
            yield _evt({
                "type": "workout_template",
                "status": "edit_applied",
                "template_markdown": md,
                "template_json": new_tpl,
                "template_ids": tpl_ids,
                "message": _format_template_for_display(new_tpl),  
                "summary": summary or "Made those changes for you!"
            })
            
            yield _evt({
                "type": "workout_template",
                "status": "ask_edit_decision",
                "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
            })
            
        except Exception as e:
            print(f"Enhanced edit error: {e}")
            yield _evt({
                "type": "workout_template",
                "status": "error",
                "message": "🤔 Had trouble applying that change.\n\n💡 Can you try describing it differently?\n✨ I'm here to help make it perfect!"
            })
        
        yield "event: done\ndata: [DONE]\n\n"
    
    return StreamingResponse(_apply_edit(), media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

  
   # CONFIRM_SAVE STATE
   elif next_state == FlexibleConversationState.STATES["CONFIRM_SAVE"]:
       prof = pend.get("profile", {})
       tpl = pend.get("template", {})
      
       await mem.set_pending(user_id, {
           "state": FlexibleConversationState.STATES["CONFIRM_SAVE"],
           "profile": prof,
           "template": tpl
       })
      
       async def _confirm_save():
           yield _evt({
               "type": "workout_template",
               "status": "confirm_save",
               "message": SmartResponseGenerator.get_contextual_prompt("CONFIRM_SAVE")
           })
           yield "event: done\ndata: [DONE]\n\n"
          
       return StreamingResponse(_confirm_save(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # DONE STATE (Save template)
   elif next_state == FlexibleConversationState.STATES["DONE"]:
       prof = pend.get("profile", {})
       tpl = pend.get("template", {})
       template_name = tpl.get("name") or f"Custom Workout ({', '.join(prof.get('template_names', ['Multi-Day']))})"
      
       async def _save_and_complete():
          
           success = await _store_template(mem, db, user_id, tpl, template_name)
          
           if success:
               # Call structurize function directly
               try:
                   request_data = StructurizeAndSaveRequest(client_id=user_id, template=tpl)
                   await structurize_and_save(request_data, db)
               except Exception as e:
                   print("structurize_and_save failed:", e)
              
               await mem.set_pending(user_id, None)  # Clear state
              
               yield _evt({
                   "type": "workout_template",
                   "status": "stored",
                   "template_name": template_name,
                   "info": "🎉 Your workout template has been saved successfully! 🎉"
               })
              
               yield _evt({
                   "type": "workout_template",
                   "status": "done",
                   "message": "🏆 Perfect! Your personalized workout is ready now! 🏆\n\n💪 Time to crush those fitness goals!\n🎯 Your template is saved and ready to use anytime!"
               })
           else:
               yield _evt({
                   "type": "workout_template",
                   "status": "error",
                   "message": "⚠️ Oops! Couldn't save the template right now.\n\n🔄 Want to try again? Just say 'save it' once more!\n💬 I'm here to help you get this sorted!"
               })
          
           yield "event: done\ndata: [DONE]\n\n"
          
       return StreamingResponse(_save_and_complete(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # ═══════════════════════════════════════════════════════════════
   # CONTEXT-AWARE STATE HANDLING - ULTRA FLEXIBLE RESPONSES
   # ═══════════════════════════════════════════════════════════════
  
   # Handle EDIT_DECISION state responses
   if current_state == FlexibleConversationState.STATES["EDIT_DECISION"]:
    prof = pend.get("profile", {})
    tpl = pend.get("template", {})
    
    # Check for explicit save commands
    save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'perfect', 'looks good', 'good to go']
    if any(cmd in user_input.lower() for cmd in save_commands):
        # User wants to save, go to confirm save
        await mem.set_pending(user_id, {
            "state": FlexibleConversationState.STATES["CONFIRM_SAVE"],
            "profile": prof,
            "template": tpl
        })
        
        async def _go_to_save():
            yield _evt({
                "type": "workout_template",
                "status": "confirm_save",
                "message": SmartResponseGenerator.get_contextual_prompt("CONFIRM_SAVE")
            })
            yield "event: done\ndata: [DONE]\n\n"
            
        return StreamingResponse(_go_to_save(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    elif UltraFlexibleParser.is_negative_response(user_input) and 'change' not in user_input.lower() and 'edit' not in user_input.lower():
        # User is satisfied, go to confirm save
        await mem.set_pending(user_id, {
            "state": FlexibleConversationState.STATES["CONFIRM_SAVE"],
            "profile": prof,
            "template": tpl
        })
        
        async def _go_to_save():
            yield _evt({
                "type": "workout_template",
                "status": "confirm_save",
                "message": SmartResponseGenerator.get_contextual_prompt("CONFIRM_SAVE")
            })
            yield "event: done\ndata: [DONE]\n\n"
            
        return StreamingResponse(_go_to_save(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    else:
        # User gave specific edit instructions directly - PRESERVE TEMPLATE
        async def _direct_edit():
            try:
                # CRITICAL: Ensure template is preserved
                current_template = tpl.copy() if tpl else {}
                
                if not current_template or not current_template.get('days'):
                    yield _evt({
                        "type": "workout_template",
                        "status": "error",
                        "message": "I seem to have lost your template. Let me recreate it for you. Could you say 'create new template' to start fresh?"
                    })
                    return
                
                # Analyze the edit request intelligently
                analysis = SmartWorkoutEditor.analyze_edit_request(user_input, current_template)
                
                # Check if request is feasible
                if analysis['target_day']:
                    day_exercises = current_template.get('days', {}).get(analysis['target_day'], {}).get('exercises', [])
                    limits = SmartWorkoutEditor.check_exercise_limits(day_exercises, analysis['target_muscle'])
                    
                    # Handle overloaded day
                    if limits['is_overloaded'] and analysis['should_add']:
                        yield _evt({
                            "type": "workout_template",
                            "status": "constraint_message",
                            "message": f"{analysis['target_day'].title()} already has {limits['total_count']} exercises, which is quite comprehensive! Would you like me to replace an existing exercise instead?"
                        })
                        yield _evt({
                            "type": "workout_template",
                            "status": "ask_edit_decision",
                            "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
                        })
                        return
                
                
                # Generate enhanced prompt for LLM
                smart_prompt = SmartWorkoutEditor.generate_smart_edit_prompt(user_input, analysis, current_template)
                
                new_tpl, summary = enhanced_edit_template(oai, OPENAI_MODEL, current_template, smart_prompt, prof, db)
                
                # CRITICAL: Validate that template wasn't corrupted
                if not new_tpl or not new_tpl.get('days') or all(not day.get('exercises') for day in new_tpl.get('days', {}).values()):
                    # LLM corrupted the template, keep original
                    new_tpl = current_template
                    summary = "I had trouble making that change while preserving your workout structure. The template has been kept as-is. Could you try rephrasing your request?"
                
                md = render_markdown_from_template(new_tpl)
                tpl_ids = build_id_only_structure(new_tpl)
               
                await mem.set_pending(user_id, {
                    "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                    "profile": prof,
                    "template": new_tpl
                })
               
                yield _evt({
                    "type": "workout_template",
                    "status": "edit_applied",
                    "template_markdown": md,
                    "template_json": new_tpl,
                    "template_ids": tpl_ids,
                    "message": _format_template_for_display(new_tpl),
                    "summary": summary or "Updated as requested!"
                })
               
                yield _evt({
                    "type": "workout_template",
                    "status": "ask_edit_decision",
                    "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
                })
               
            except Exception as e:
                print(f"Direct edit error: {e}")
                yield _evt({
                    "type": "workout_template",
                    "status": "error",
                    "message": "Had trouble with that change. Can you try rephrasing what you'd like me to modify?"
                })
           
            yield "event: done\ndata: [DONE]\n\n"
           
        return StreamingResponse(_direct_edit(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # Handle CONFIRM_SAVE state responses 
   if current_state == FlexibleConversationState.STATES["CONFIRM_SAVE"]:
    prof = pend.get("profile", {})
    tpl = pend.get("template", {})
    
    if UltraFlexibleParser.is_positive_response(user_input) or user_input.lower().strip() in ['save', 'yes', 'confirm']:
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
                try:
                    request_data = StructurizeAndSaveRequest(client_id=user_id, template=tpl)
                    await structurize_and_save(request_data, db)
                except Exception as e:
                    print("structurize_and_save failed:", e)
                
                await mem.set_pending(user_id, None)
                
                yield _evt({
                    "type": "workout_template",
                    "status": "stored",
                    "template_name": template_name,
                    "info": "🎉 Awesome! Your workout template is now saved and ready to use! 🎉"
                })
                
                yield _evt({
                    "type": "workout_template",
                    "status": "done",
                    "message": "🎯 All set! Your workout is ready to rock! 🎯\n\n📋 Say 'show my template' to view it anytime\n✏️ Say 'edit template' to create variations\n💬 What else can I help you with?"
                })
            else:
                yield _evt({
                    "type": "workout_template",
                    "status": "error",
                    "message": "Couldn't save right now - want to try again? Just say 'save' or let me know!"
                })
            
            yield "event: done\ndata: [DONE]\n\n"
            
        return StreamingResponse(_final_save(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    elif UltraFlexibleParser.is_negative_response(user_input):
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
    
    else:
        # Treat as edit instruction - but preserve template
        async def _edit_from_save():
            try:
                current_template = tpl.copy() if tpl else {}
                
                if not _validate_template_integrity(current_template):
                    yield _evt({
                        "type": "workout_template",
                        "status": "error",
                        "message": "The template seems corrupted. Let me help create a new one. Say 'create template' to start over."
                    })
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                
                # Rest of edit logic...
                analysis = SmartWorkoutEditor.analyze_edit_request(user_input, current_template)
                
                
                smart_prompt = SmartWorkoutEditor.generate_smart_edit_prompt(user_input, analysis, current_template)
               
                new_tpl, summary = enhanced_edit_template(oai, OPENAI_MODEL, current_template, smart_prompt, prof, db)
                
                # Validate result
                if not _validate_template_integrity(new_tpl):
                    new_tpl = current_template
                    summary = "Had trouble making that change while preserving your workout. Template kept as-is."
                
                md = render_markdown_from_template(new_tpl)
                tpl_ids = build_id_only_structure(new_tpl)
               
                await mem.set_pending(user_id, {
                    "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                    "profile": prof,
                    "template": new_tpl
                })
               
                yield _evt({
                    "type": "workout_template",
                    "status": "edit_applied",
                    "template_markdown": md,
                    "template_json": new_tpl,
                    "template_ids": tpl_ids,
                    "message": _format_template_for_display(new_tpl),
                    "summary": summary or "Made that change!"
                })
               
                yield _evt({
                    "type": "workout_template",
                    "status": "ask_edit_decision",
                    "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
                })
               
            except Exception as e:
                print(f"Edit from save error: {e}")
                yield _evt({
                    "type": "workout_template",
                    "status": "confirm_save",
                    "message": "Didn't catch that change. " + SmartResponseGenerator.get_contextual_prompt("CONFIRM_SAVE")
                })
           
            yield "event: done\ndata: [DONE]\n\n"
           
        return StreamingResponse(_edit_from_save(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # ═══════════════════════════════════════════════════════════════
   # ULTRA-FLEXIBLE INPUT HANDLING FOR SPECIFIC STATES
   # ═══════════════════════════════════════════════════════════════
  
  
   # Handle ASK_NAMES state when user provides names
   if current_state == FlexibleConversationState.STATES["ASK_NAMES"]:
       prof = pend.get("profile", {})
       days_count = prof.get("days_count", 6)
       template_names = UltraFlexibleParser.extract_template_names(user_input, days_count)
    
       prof["template_names"] = template_names
       prof["days_per_week"] = days_count
    
       async def _got_names():
           import asyncio
           names_str = ", ".join(template_names)
        
           try:
               tpl, why = llm_generate_template_from_profile(oai, OPENAI_MODEL, prof, db)

                # FIXED: Use custom day names instead of overwriting with standard names
               if tpl and 'days' in tpl and template_names:
                    new_days = {}
                    old_keys = list(tpl['days'].keys())

                    for i, template_name in enumerate(template_names):
                        if i < len(old_keys):
                            old_key = old_keys[i]
                            # Use the custom name as the day key (lowercase)
                            new_key = template_name.lower().replace(' ', '_')
                            new_days[new_key] = tpl['days'][old_key].copy()

                            # CRITICAL FIX: Update the title to show custom day name
                            original_title = new_days[new_key].get('title', '')
                            if original_title and original_title.lower() != template_name.lower():
                                # Combine custom name with muscle focus if available
                                new_days[new_key]['title'] = f"{template_name} — {original_title}"
                            else:
                                # Just use the custom name
                                new_days[new_key]['title'] = template_name

                    tpl['days'] = new_days
               md = render_markdown_from_template(tpl)
               tpl_ids = build_id_only_structure(tpl)
              
               await mem.set_pending(user_id, {
                   "state": FlexibleConversationState.STATES["EDIT_DECISION"],
                   "profile": prof,
                   "template": tpl
               })
              
               yield _evt({
                    "type": "workout_template",
                    "status": "draft",
                    "template_markdown": md,
                    "template_json": tpl,
                    "template_ids": tpl_ids,
                    "message": _format_template_for_display(tpl), 
                    "why": why or "Tailored specifically for your fitness journey!"
                })
              
               yield _evt({
                   "type": "workout_template",
                   "status": "ask_edit_decision",
                   "message": SmartResponseGenerator.get_contextual_prompt("EDIT_DECISION")
               })
              
           except Exception as e:
               print(f"Names generation error: {e}")
               yield _evt({
                   "type": "workout_template",
                   "status": "error",
                   "message": "Had trouble creating your template. Want to try different names or start over? I'm flexible!"
               })
          
           yield "event: done\ndata: [DONE]\n\n"
          
       return StreamingResponse(_got_names(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
  
   # ═══════════════════════════════════════════════════════════════
   # ULTIMATE FALLBACK - CONTEXT-AWARE SMART RESPONSES
   # ═══════════════════════════════════════════════════════════════
  
   async def _ultra_smart_fallback():
       # Ultra-context-aware fallback responses
       if not pend:
           # No context - offer main options with personality
           yield _evt({
               "type": "workout_template",
               "status": "hint",
               "message": "I'm your workout template assistant! I can help you create personalized plans, show existing templates, or make edits. Just tell me what you need - like 'make me a workout', 'show my plan', or 'change my routine'. What sounds good?"
           })
      
       elif current_state == FlexibleConversationState.STATES["START"]:
           # Just starting - be encouraging
           yield _evt({
               "type": "workout_template",
               "status": "hint",
               "message": "Let's create something amazing! Tell me you want a workout plan and I'll build one just for you. You can say anything like 'create workout', 'make me a routine', or 'I need a plan' - I understand natural language!"
           })
          
          
          
       else:
           # Generic but helpful
           yield _evt({
               "type": "workout_template",
               "status": "hint",
               "message": "I didn't quite catch that, but I'm here to help! You can describe what you want naturally - like 'yes', 'no', 'change this exercise', 'make it harder', or tell me exactly what you're thinking. What would you like to do?"
           })
      
       yield "event: done\ndata: [DONE]\n\n"
  
   return StreamingResponse(_ultra_smart_fallback(), media_type="text/event-stream",
                          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
# ═══════════════════════════════════════════════════════════════
# ADDITIONAL ENHANCED HELPER ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@router.post("/reset_conversation")
async def reset_conversation(user_id: int, mem = Depends(get_mem)):
   """Reset conversation state for user"""
   await mem.set_pending(user_id, None)
   return {
       "status": "reset",
       "message": "Conversation state cleared - ready for a fresh start!",
       "user_id": user_id
   }
@router.get("/conversation_status")
async def get_conversation_status(user_id: int, mem = Depends(get_mem)):
   """Get detailed conversation state"""
   pend = await mem.get_pending(user_id)
   profile = pend.get("profile", {}) if pend else {}
  
   return {
       "user_id": user_id,
       "current_state": pend.get("state") if pend else "none",
       "has_profile": bool(pend.get("profile")) if pend else False,
       "has_template": bool(pend.get("template")) if pend else False,
       "profile_details": {
           "days_count": profile.get("days_count"),
           "template_names": profile.get("template_names"),
           "client_goal": profile.get("client_goal"),
           "experience": profile.get("experience")
       } if profile else None,
       "can_transition_to": "any_state",
       "flexibility_level": "ultra_high"
   }
@router.get("/debug_intent")
async def debug_intent_parsing(text: str):
   """Debug endpoint to test intent parsing"""
   user_intent, confidence = UltraFlexibleParser.extract_intent(text)
   days_count = UltraFlexibleParser.extract_days_count(text)
   template_names = UltraFlexibleParser.extract_template_names(text, days_count)
   is_positive = UltraFlexibleParser.is_positive_response(text)
   is_negative = UltraFlexibleParser.is_negative_response(text)
  
   return {
       "input": text,
       "parsing_results": {
           "intent": user_intent,
           "confidence": confidence,
           "days_count": days_count,
           "template_names": template_names,
           "is_positive": is_positive,
           "is_negative": is_negative
       },
       "flexibility_features": [
           "typo_tolerance",
           "natural_language_processing",
           "context_awareness",
           "fuzzy_matching",
           "multi_pattern_detection"
       ]
   }
# ═══════════════════════════════════════════════════════════════
# ENHANCED ERROR HANDLING AND RECOVERY
# ═══════════════════════════════════════════════════════════════
@router.post("/recover_conversation")
async def recover_conversation(
   user_id: int,
   recovery_data: Dict[str, Any],
   mem = Depends(get_mem)
):
   """Recover conversation from a specific state with data"""
   try:
       await mem.set_pending(user_id, recovery_data)
       return {
           "status": "recovered",
           "message": "Conversation state has been restored",
           "recovered_state": recovery_data.get("state"),
           "user_id": user_id
       }
   except Exception as e:
       return {
           "status": "error",
           "message": f"Failed to recover conversation: {str(e)}",
           "user_id": user_id
       }
@router.get("/health_check")
async def health_check():
   """Health check endpoint for the ultra-flexible workout chatbot"""
   return {
       "status": "healthy",
       "service": "ultra_flexible_workout_chatbot",
       "features": {
           "ultra_flexible_parsing": True,
           "typo_tolerance": True,
           "natural_language_understanding": True,
           "context_aware_transitions": True,
           "free_form_conversation": True,
           "multi_state_jumping": True
       },
       "supported_intents": ["create", "show", "edit", "unknown"],
       "supported_states": list(FlexibleConversationState.STATES.values()),
       "flexibility_level": "maximum"
   }
# ═══════════════════════════════════════════════════════════════
# ADDITIONAL UTILITY FUNCTIONS FOR ENHANCED FLEXIBILITY
# ═══════════════════════════════════════════════════════════════
def validate_template_structure(template: Dict[str, Any]) -> bool:
   """Validate template structure for safety"""
   required_keys = ["name", "goal", "days"]
   return all(key in template for key in required_keys)
def sanitize_user_input(text: str) -> str:
   """Sanitize user input while preserving natural language flexibility"""
   # Remove potentially harmful characters but keep natural language intact
   sanitized = re.sub(r'[<>{}[\]\\]', '', text)
   # Preserve emojis and natural punctuation
   sanitized = re.sub(r'\s+', ' ', sanitized).strip()
   return sanitized[:1000]  # Reasonable length limit
class ConversationLogger:
   """Log conversation flows for debugging and improvement"""

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
        # CRITICAL FIX: Only advance if we actually extracted valid day information
        extracted_days = UltraFlexibleParser.extract_days_count(user_input)
        context_days = context.get('profile', {}).get('days_count') if context else None
        
        # Only proceed if we have a valid number (not None and > 0)
        if (extracted_days is not None and extracted_days > 0) or (context_days is not None and context_days > 0):
            return FlexibleConversationState.STATES["ASK_NAMES"]
        return current_state  # Stay in ASK_DAYS and ask again
        
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
        return FlexibleConversationState.STATES["EDIT_DECISION"]
        
    elif current_state == FlexibleConversationState.STATES["EDIT_DECISION"]:
        # Check for explicit save commands first
        save_commands = ['save', 'save it', 'store', 'store it', 'keep', 'keep it', 'perfect', 'looks good', 'good to go']
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
  
   @staticmethod
   def log_transition(user_id: int, from_state: str, to_state: str,
                     user_input: str, intent: str, confidence: float):
       print(f"🔄 USER {user_id}: {from_state} → {to_state}")
       print(f"   Input: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
       print(f"   Intent: {intent} (confidence: {confidence:.2f})")
  
   @staticmethod
   def log_parsing_result(user_input: str, parsing_results: Dict[str, Any]):
       print(f"🔍 PARSING: '{user_input}'")
       print(f"   Results: {parsing_results}")
