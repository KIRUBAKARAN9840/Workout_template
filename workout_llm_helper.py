from __future__ import annotations
import orjson, json
from typing import Dict, Any, Tuple, List, Optional
from sqlalchemy.orm import Session
from .exercise_catalog_db import load_catalog, id_for_name, pick_from_muscles
import copy
import re


class AIConversationManager:
    """AI-powered conversation manager for natural workout template creation"""

    @staticmethod
    def analyze_user_intent(oai, model: str, user_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use AI to understand user intent naturally, handling typos and variations"""
        context = conversation_context or {}

        system_prompt = """You are an AI assistant helping users create workout templates. Analyze the user's input and determine their intent.

Available intents:
- "create": User wants to create a new workout template
- "show": User wants to see their existing template
- "edit": User wants to modify their template
- "save": User wants to save their template
- "yes": User is agreeing/confirming something
- "no": User is disagreeing/declining something
- "specify_days": User is specifying number of workout days
- "specify_names": User is providing day names/titles
- "ask_question": User is asking a question
- "unclear": Intent is unclear

Additional information to extract:
- days_count: If user mentions number of days (1-7)
- day_names: If user provides specific day names
- muscle_groups: Any muscle groups mentioned
- positive_sentiment: true/false if response seems positive
- negative_sentiment: true/false if response seems negative
- exercise_requests: Any specific exercises mentioned

Handle typos, variations, and natural language. Be flexible and understanding.

Respond in JSON format with: intent, confidence (0-1), days_count, day_names (array), muscle_groups (array), positive_sentiment, negative_sentiment, exercise_requests (array), reasoning"""

        user_prompt = f"""User input: "{user_input}"

Context:
Current conversation state: {context.get('state', 'unknown')}
Has existing template: {bool(context.get('template'))}
Profile info: {context.get('profile', {})}

Analyze this input and determine what the user wants to do."""

        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            result = json.loads(resp.choices[0].message.content or "{}")

            # Ensure all expected fields are present
            return {
                "intent": result.get("intent", "unclear"),
                "confidence": float(result.get("confidence", 0.0)),
                "days_count": result.get("days_count"),
                "day_names": result.get("day_names", []),
                "muscle_groups": result.get("muscle_groups", []),
                "positive_sentiment": result.get("positive_sentiment", False),
                "negative_sentiment": result.get("negative_sentiment", False),
                "exercise_requests": result.get("exercise_requests", []),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            print(f"AI intent analysis failed: {e}")
            # Fallback to basic analysis
            return {
                "intent": "unclear",
                "confidence": 0.0,
                "days_count": None,
                "day_names": [],
                "muscle_groups": [],
                "positive_sentiment": False,
                "negative_sentiment": False,
                "exercise_requests": [],
                "reasoning": f"Failed to analyze: {e}"
            }

    @staticmethod
    def determine_conversation_flow(oai, model: str, user_input: str, current_state: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered conversation flow determination"""

        system_prompt = """You are managing a workout template creation conversation. Based on the user input and current context, determine what should happen next.

Available states:
- "FETCH_PROFILE": Get user's fitness profile and show it to them
- "PROFILE_CONFIRMATION": Show profile and ask for confirmation
- "ASK_DAYS": Ask how many workout days per week
- "ASK_NAMES": Ask for day names/titles
- "DRAFT_GENERATION": Create the workout template
- "SHOW_TEMPLATE": Display the current template
- "EDIT_DECISION": Ask if user wants to edit
- "APPLY_EDIT": Apply user's edit request
- "CONFIRM_SAVE": Ask to confirm saving
- "DONE": Conversation complete
- "STAY": Stay in current state, ask for clarification

IMPORTANT FLOW RULES:
1. For initial greetings ("Hi", "Hello"), always go to "START" to show profile immediately
2. If user says "yes", "create", "workout" after seeing profile, proceed to workout creation flow
3. START state should show the user's existing profile and ask for preferences
4. Always prioritize showing existing profile data first before asking questions
5. After profile confirmation, proceed directly to workout days/template creation

Also determine:
- should_proceed: true/false if we have enough info to move forward
- response_message: What to tell the user
- extracted_info: Any specific information extracted from input

Be flexible with user responses. Handle typos, variations, and natural speech patterns."""

        user_prompt = f"""Current state: {current_state}
User input: "{user_input}"

Context:
- Has profile: {bool(context.get('profile'))}
- Has template: {bool(context.get('template'))}
- Profile: {context.get('profile', {})}
- Template exists: {bool(context.get('template', {}).get('days'))}

IMPORTANT:
- If user says "Hi", "Hello", etc., go to "START" to show profile immediately
- If current state is "start" and user shows interest (says "yes", "create", etc.), proceed to workout creation
- Always show profile data first before asking for workout preferences

What should happen next? Respond in JSON format with next_state, should_proceed, response_message, extracted_info."""

        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            content = resp.choices[0].message.content or ""
            print(f"🔍 AI Flow Raw Response: {content}")

            # Try to parse JSON, with fallbacks
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except:
                        result = {}
                else:
                    result = {}

            return {
                "next_state": result.get("next_state", "STAY"),
                "should_proceed": result.get("should_proceed", True),
                "response_message": result.get("response_message", "I'm not sure what you mean. Could you clarify?"),
                "extracted_info": result.get("extracted_info", {})
            }
        except Exception as e:
            print(f"🚨 AI Flow Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "next_state": "STAY",
                "should_proceed": False,
                "response_message": f"I'm having trouble understanding. Could you try rephrasing? (Error: {e})",
                "extracted_info": {}
            }

    @staticmethod
    def validate_and_map_exercises(oai, model: str, user_request: str, db: Session) -> Dict[str, Any]:
        """AI-powered exercise validation - ensures only database exercises are used"""

        # Load exercise catalog
        catalog = load_catalog(db)
        available_exercises = []
        for exercise_id, exercise_data in catalog["by_id"].items():
            available_exercises.append({
                "id": exercise_id,
                "name": exercise_data["name"],
                "muscle_group": exercise_data["muscle_group"],
                "isCardio": exercise_data["isCardio"],
                "isBodyWeight": exercise_data["isBodyWeight"]
            })

        system_prompt = """You are helping map user exercise requests to available exercises in our database.

CRITICAL RULES:
1. ONLY use exercises from the provided database list
2. If user requests an exercise not in database, find the closest alternative from database
3. If no suitable alternative exists, explain this to the user
4. Never invent or create new exercises

For each requested exercise, provide:
- database_id: The exact ID from our database (must exist)
- database_name: The exact name from database
- user_requested: What the user originally asked for
- is_match: true if exact match, false if alternative
- explanation: Brief explanation if using alternative

Respond in JSON format with an array of exercise mappings."""

        exercises_list = json.dumps(available_exercises, indent=2)

        user_prompt = f"""User request: "{user_request}"

Available exercises in database:
{exercises_list}

Map the user's request to valid database exercises only. If they ask for something not available, find the best alternatives from the database."""

        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            result = json.loads(resp.choices[0].message.content or "{}")
            return {
                "exercise_mappings": result.get("exercise_mappings", []),
                "success": True,
                "message": "Exercise mapping completed"
            }

        except Exception as e:
            print(f"Exercise validation failed: {e}")
            return {
                "exercise_mappings": [],
                "success": False,
                "message": f"Failed to validate exercises: {e}"
            }

    @staticmethod
    def generate_contextual_response(oai, model: str, conversation_state: str, user_input: str, context: Dict[str, Any]) -> str:
        """Generate natural, contextual responses for any conversation state"""

        system_prompt = """You are a friendly, encouraging fitness assistant helping users create workout templates.

Generate natural, conversational responses that:
1. Acknowledge what the user said
2. Provide clear guidance on next steps
3. Stay encouraging and positive
4. Handle typos and unclear input gracefully
5. Ask clarifying questions when needed

Keep responses concise but warm. Use emojis sparingly and appropriately."""

        context_info = {
            "state": conversation_state,
            "has_profile": bool(context.get("profile")),
            "has_template": bool(context.get("template")),
            "user_info": context.get("profile", {})
        }

        user_prompt = f"""Current conversation state: {conversation_state}
User just said: "{user_input}"
Context: {json.dumps(context_info)}

Generate an appropriate response for this situation."""

        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            return "I'm here to help you create a great workout plan! Could you tell me more about what you're looking for?"



def extract_bulk_operation_info(text: str) -> Dict[str, Any]:
    """Extract information for bulk operations like 'add biceps to all days'"""
    import re
    
    # Define patterns locally in this function
    SPECIFIC_COUNT_PATTERNS = [
        r'(?:for|on)\s*(\d+)\s*days?',
        r'(\d+)\s*days?',
        r'(?:for|on)\s*(?:the\s*)?(?:first|last)\s*(\d+)\s*days?',
    ]

    MUSCLE_CHANGE_PATTERNS = {
        'legs': [
            r'leg\s*(?:exercise|workout|training)',
            r'lower\s*body',
            r'lowerbody',  # Added this
            r'quadriceps?',
            r'hamstrings?',
            r'glutes?',
            r'calves?'
        ],
        'upper': [
            r'upper\s*body',
            r'upperbody',  # Added this
            r'upper\s*(?:exercise|workout)',
            r'chest\s*and\s*arms?',
            r'arms?\s*and\s*chest'
        ],
        'core': [r'core\s*(?:exercise|workout)', r'ab\s*(?:exercise|workout)', r'abdominal'],
        'chest': [r'chest\s*(?:exercise|workout)', r'pec\s*(?:exercise|workout)'],
        'back': [r'back\s*(?:exercise|workout)', r'lat\s*(?:exercise|workout)', r'pull\s*(?:exercise|workout)'],
        'biceps': [r'bicep\s*(?:exercise|workout)', r'arm\s*curl', r'bicep\s*curl'],
        'triceps': [r'tricep\s*(?:exercise|workout)', r'tri\s*(?:exercise|workout)'],
        'shoulders': [r'shoulder\s*(?:exercise|workout)', r'delt\s*(?:exercise|workout)'],
        'cardio': [r'cardio\s*(?:exercise|workout)', r'aerobic', r'running', r'cycling']
    }
    
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
    for pattern in SPECIFIC_COUNT_PATTERNS:
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
    for muscle, patterns in MUSCLE_CHANGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                result['target_muscle'] = muscle
                break
        if result['target_muscle']:
            break
    
    return result

class SmartWorkoutEditor:
    """Intelligent workout editor that understands context and exercise relationships"""
    # Exercise database by muscle groups
    EXERCISE_GROUPS = {
        'legs': ['squats', 'leg press', 'lunges', 'leg extensions', 'leg curls', 'calf raises', 'bulgarian split squats', 'step ups', 'wall sits', 'goblet squats', 'leg raises'],
        'quadriceps': ['squats', 'leg press', 'lunges', 'leg extensions', 'bulgarian split squats', 'step ups', 'goblet squats'],
        'hamstrings': ['leg curls', 'romanian deadlifts', 'stiff leg deadlifts', 'good mornings', 'single leg deadlifts'],
        'calves': ['calf raises', 'standing calf raises', 'seated calf raises', 'donkey calf raises'],
        'glutes': ['squats', 'hip thrusts', 'glute bridges', 'lunges', 'bulgarian split squats', 'step ups'],
        'chest': ['bench press', 'push ups', 'chest flyes', 'incline press', 'decline press', 'dips', 'chest dips'],
        'back': ['pull ups', 'lat pulldowns', 'rows', 'deadlifts', 'shrugs', 'bent over rows', 'cable rows'],
        'shoulders': ['shoulder press', 'lateral raises', 'front raises', 'rear delt flyes', 'overhead press'],
        'biceps': ['bicep curls', 'hammer curls', 'chin ups', 'concentration curls', 'preacher curls'],
        'triceps': ['tricep extensions', 'dips', 'close grip press', 'overhead extensions', 'tricep pushdowns'],
        'core': ['planks', 'crunches', 'russian twists', 'mountain climbers', 'leg raises', 'dead bugs']
    }
    
    @classmethod
    def analyze_edit_request(cls, user_input: str, current_template: dict) -> dict:
        """Analyze user edit request and determine appropriate action"""
        user_input_lower = user_input.lower()
        
        analysis = {
            'action': 'unknown',
            'target_day': None,
            'target_muscle': None,
            'exercise_count_limit': 2,
            'specific_exercises': [],
            'should_replace': False,
            'should_add': False,
            'wants_title_change': False,
            'new_title': None,
            'error_message': None
        }
        
        # Extract day mentions
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in user_input_lower or day[:3] in user_input_lower:
                analysis['target_day'] = day
                break
        
        # Extract muscle group mentions (prioritize legs/leg over other matches)
        muscle_priority = ['legs', 'leg', 'quadriceps', 'hamstrings', 'glutes', 'calves']
        all_muscles = list(cls.EXERCISE_GROUPS.keys())
        
        # Check priority muscles first
        for muscle in muscle_priority:
            if muscle in user_input_lower:
                analysis['target_muscle'] = 'legs' if muscle in ['leg', 'legs'] else muscle
                break
        
        # If no priority muscle found, check others
        if not analysis['target_muscle']:
            for muscle_group in all_muscles:
                if muscle_group in user_input_lower:
                    analysis['target_muscle'] = muscle_group
                    break
        
        # Check for title changes
        title_analysis = cls.analyze_title_change(user_input)
        analysis.update(title_analysis)
        
        # Determine action type
        if any(word in user_input_lower for word in ['add', 'include', 'more', 'extra', 'give']):
            analysis['action'] = 'add'
            analysis['should_add'] = True
        elif any(word in user_input_lower for word in ['replace', 'change', 'swap', 'substitute', 'different']):
            analysis['action'] = 'replace'
            analysis['should_replace'] = True
        elif any(word in user_input_lower for word in ['remove', 'delete', 'take out']):
            analysis['action'] = 'remove'
            analysis['should_remove'] = True
        
        return analysis
    
    @classmethod
    def check_exercise_limits(cls, day_exercises: list, target_muscle: str = None) -> dict:
        """Check if day already has too many exercises"""
        total_exercises = len(day_exercises)
        
        # Count exercises by muscle group if specified
        muscle_count = 0
        if target_muscle:
            for exercise in day_exercises:
                exercise_name = exercise.get('name', '').lower()
                if cls._exercise_belongs_to_muscle(exercise_name, target_muscle):
                    muscle_count += 1
        
        return {
            'total_count': total_exercises,
            'muscle_specific_count': muscle_count,
            'can_add_general': total_exercises < 8,
            'can_add_muscle_specific': muscle_count < 4,
            'is_overloaded': total_exercises >= 8
        }
    
    @classmethod
    def _exercise_belongs_to_muscle(cls, exercise_name: str, target_muscle: str) -> bool:
        """Check if exercise belongs to target muscle group"""
        if target_muscle not in cls.EXERCISE_GROUPS:
            return False
        
        target_exercises = cls.EXERCISE_GROUPS[target_muscle]
        return any(target_ex in exercise_name for target_ex in target_exercises)
    
    @classmethod
    def get_suitable_exercises(cls, target_muscle: str, existing_exercises: list, count: int = 2) -> list:
        """Get suitable exercises for the target muscle group"""
        if target_muscle not in cls.EXERCISE_GROUPS:
            return []
        
        existing_names = [ex.get('name', '').lower() for ex in existing_exercises]
        available_exercises = cls.EXERCISE_GROUPS[target_muscle]
        
        # Filter out exercises already in the day
        suitable = []
        for exercise in available_exercises:
            if not any(exercise in existing_name for existing_name in existing_names):
                suitable.append(exercise)
        
        return suitable[:count]
    
    @classmethod
    def validate_exercise_match(cls, requested_muscle: str, actual_exercises: list) -> dict:
        """Validate that exercises actually match the requested muscle group"""
        if requested_muscle not in cls.EXERCISE_GROUPS:
            return {"valid": True, "message": "Unknown muscle group"}
        
        target_exercises = cls.EXERCISE_GROUPS[requested_muscle]
        matched_exercises = []
        unmatched_exercises = []
        
        for exercise in actual_exercises:
            exercise_name = exercise.get('name', '').lower()
            is_match = any(target_ex in exercise_name for target_ex in target_exercises)
            if is_match:
                matched_exercises.append(exercise)
            else:
                unmatched_exercises.append(exercise)
        
        return {
            "valid": len(matched_exercises) > 0,
            "matched_count": len(matched_exercises),
            "unmatched_count": len(unmatched_exercises),
            "matched_exercises": matched_exercises,
            "unmatched_exercises": unmatched_exercises,
            "message": f"Found {len(matched_exercises)} {requested_muscle} exercises, {len(unmatched_exercises)} others"
        }
    
    @classmethod
    def generate_smart_edit_prompt(cls, user_input: str, analysis: dict, template: dict) -> str:
        """Generate intelligent prompt for LLM based on analysis"""
        target_day = analysis.get('target_day')
        target_muscle = analysis.get('target_muscle')
        action = analysis.get('action')
        
        base_prompt = f"User request: {user_input}\n"
        
        # Day-specific constraints
        if target_day:
            day_exercises = template.get('days', {}).get(target_day, {}).get('exercises', [])
            limits = cls.check_exercise_limits(day_exercises, target_muscle)
            
            base_prompt += f"Target day: {target_day.title()} (currently has {limits['total_count']} exercises)\n"
            
            if limits['is_overloaded'] and action == 'add':
                return base_prompt + f"CONSTRAINT: {target_day.title()} already has {limits['total_count']} exercises which is the maximum. Replace existing exercises instead of adding new ones."
        
        # Muscle group specific guidance
        if target_muscle and target_muscle in cls.EXERCISE_GROUPS:
            suitable_exercises = cls.EXERCISE_GROUPS[target_muscle][:5]  # Top 5 examples
            base_prompt += f"MUSCLE GROUP: {target_muscle.upper()}\n"
            base_prompt += f"VALID {target_muscle.upper()} EXERCISES: {', '.join(suitable_exercises)}\n"
            base_prompt += f"IMPORTANT: Only use exercises that actually target {target_muscle}. Do not use chest, arm, or other muscle group exercises when {target_muscle} is requested.\n"
        
        # Action-specific instructions
        if action == 'add':
            base_prompt += "ACTION: Add new exercises to the specified day.\n"
        elif action == 'replace':
            base_prompt += "ACTION: Replace existing exercises with new ones.\n"
        elif action == 'remove':
            base_prompt += "ACTION: Remove specified exercises.\n"
        
        # Title change handling
        if 'change' in user_input.lower() and any(day in user_input.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            # Check if user wants to change day title
            words = user_input.lower().split()
            if 'to' in words:
                to_index = words.index('to')
                if to_index < len(words) - 1:
                    new_title = ' '.join(words[to_index + 1:]).title()
                    base_prompt += f"TITLE CHANGE: Also update the day title to '{new_title}'\n"
        
        base_prompt += "CONSTRAINTS: Maximum 6 exercises per day. Keep existing structure unless specifically asked to change."
        
        return base_prompt
    
    @classmethod
    def analyze_title_change(cls, user_input: str) -> dict:
        """Analyze if user wants to change day title - enhanced pattern matching"""
        user_input_lower = user_input.lower()
        result = {
            'wants_title_change': False,
            'target_day': None,
            'new_title': None
        }

        print(f"🔍 Analyzing title change for: '{user_input}'")  # Debug log
        
        # Enhanced title change patterns
        title_patterns = [
            r'change\s+(\w+day)\s+(?:to|as)\s+(.+)',          # "change tuesday to/as something"
            r'rename\s+(\w+day)\s+(?:to|as)\s+(.+)',          # "rename tuesday to/as something"
            r'call\s+(\w+day)\s+(.+)',                        # "call tuesday something"
            r'(\w+day)\s+(?:to|as)\s+(.+)',                   # "tuesday to/as something"
            r'change\s+(\w+)\s+(?:to|as)\s+(.+)',             # "change tuesday as mooonday"
            r'make\s+(\w+day)\s+(?:called|named)\s+(.+)',     # "make tuesday called something"
            # NEW: Handle "day 1", "day 2", etc. patterns
            r'change\s+day\s*(\d+)\s+(?:to|as)\s+(.+)',       # "change day 1 as monster"
            r'rename\s+day\s*(\d+)\s+(?:to|as)\s+(.+)',       # "rename day 1 to monster"
            r'call\s+day\s*(\d+)\s+(.+)',                     # "call day 1 monster"
            r'day\s*(\d+)\s+(?:to|as)\s+(.+)',                # "day 1 as monster"
            r'make\s+day\s*(\d+)\s+(?:called|named)\s+(.+)',  # "make day 1 called monster"
            # Handle "day X name" patterns
            r'change\s+day\s*(\d+)\s+name\s+(?:to|as)\s+(.+)',  # "change day 1 name as night shift"
            r'rename\s+day\s*(\d+)\s+name\s+(?:to|as)\s+(.+)',  # "rename day 1 name to night shift"
            r'call\s+day\s*(\d+)\s+name\s+(.+)',                # "call day 1 name monster"
            r'day\s*(\d+)\s+name\s+(?:to|as)\s+(.+)',           # "day 1 name as monster"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                target_day = match.group(1)
                new_title = match.group(2).strip()
                print(f"🎯 Pattern matched: '{pattern}' -> target_day: '{target_day}', new_title: '{new_title}'")

                # Validate that target_day looks like a day or day number
                day_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

                # Check if it's a regular day name or a day number (1-7)
                is_valid_day = (any(day_key in target_day for day_key in day_keywords) or
                               (target_day.isdigit() and 1 <= int(target_day) <= 7))

                print(f"🔍 Day validation: target_day='{target_day}', is_valid={is_valid_day}")

                if is_valid_day:
                    result['wants_title_change'] = True
                    result['target_day'] = target_day
                    result['new_title'] = new_title.title()
                    print(f"✅ Title change detected: {result}")
                    break

        print(f"🔍 Final result: {result}")  # Debug log
        
        return result
    @classmethod
    def apply_title_change(cls, template: dict, target_day: str, new_title: str) -> Tuple[dict, str]:
        """Apply day name change - changes both the day key and the display title"""
        print(f"🔄 Applying title change: target_day='{target_day}', new_title='{new_title}'")
        updated = template.copy()
        days = updated.get('days', {})
        print(f"🔍 Current days in template: {list(days.keys())}")

        # Find the actual day key in the template
        matching_day_key = None

        # Handle day numbers (1, 2, 3, etc.)
        if target_day.isdigit():
            day_number = int(target_day)
            day_keys = list(days.keys())
            print(f"🔢 Day number {day_number}, available days: {day_keys}")
            if 1 <= day_number <= len(day_keys):
                matching_day_key = day_keys[day_number - 1]  # Convert to 0-based index
                print(f"✅ Found day by number: {matching_day_key}")
            else:
                print(f"❌ Day number {day_number} out of range (1-{len(day_keys)})")
        else:
            # Handle day names (monday, tuesday, etc.)
            print(f"🔤 Looking for day name: '{target_day}'")
            for day_key in days.keys():
                if target_day.lower() in day_key.lower() or day_key.lower() in target_day.lower():
                    matching_day_key = day_key
                    print(f"✅ Found day by name: {matching_day_key}")
                    break
            if not matching_day_key:
                print(f"❌ No matching day found for '{target_day}'")

        print(f"🎯 Final matching_day_key: {matching_day_key}")

        if matching_day_key and matching_day_key in days:
            # Get the day data and update ONLY the title (keep the same key)
            day_data = days[matching_day_key].copy()

            # IMPORTANT: Only change the title, keep the same day key
            day_data['title'] = new_title
            print(f"🎯 Updated title: '{day_data['title']}' for day key: '{matching_day_key}'")

            # Update the template with the same key but new title
            new_days = days.copy()
            new_days[matching_day_key] = day_data  # Same key, updated title

            updated['days'] = new_days
            original_title = days[matching_day_key].get('title', matching_day_key.title())
            return updated, f"Changed day from '{original_title}' to '{new_title}'"
        else:
            return template, f"Could not find day '{target_day}' in template"
        

    
        


    @classmethod
    def handle_bulk_muscle_change(cls, template: dict, muscle_group: str, operation: str, target_days: str, specific_count: int = None, db = None) -> Tuple[dict, str]:
        """Handle bulk operations like 'change all days to leg exercises'"""
        if not db:
            return template, "Database connection required for bulk operations"
        
        from .exercise_catalog_db import load_catalog, pick_from_muscles
        cat = load_catalog(db)
        if not cat:
            return template, "Could not load exercise database"
        
        updated = template.copy()
        days = updated.get('days', {})
        day_keys = list(days.keys())
        
        # Determine which days to modify
        target_day_keys = []
        if target_days == 'all':
            target_day_keys = day_keys
        elif target_days == 'specific_count' and specific_count:
            target_day_keys = day_keys[:min(specific_count, len(day_keys))]
        
        if not target_day_keys:
            return template, "No valid days found to modify"
        
        # IMPROVED: Better muscle group mapping with proper database keywords
        muscle_mapping = {
            'legs': ['lower body', 'legs', 'quadriceps', 'hamstrings', 'glutes', 'calves'],
            'upper': ['upper body', 'chest', 'back', 'shoulders', 'arms'],
            'core': ['core', 'abs', 'abdominal'],
            'chest': ['chest', 'pectorals'],
            'back': ['back', 'lats', 'rhomboids'],
            'biceps': ['biceps', 'arms'],
            'triceps': ['triceps', 'arms'],
            'shoulders': ['shoulders', 'deltoids'],
            'cardio': ['cardio', 'aerobic']
        }
        
        muscle_targets = muscle_mapping.get(muscle_group, [muscle_group])
        print(f"🎯 Targeting muscle groups: {muscle_targets} for {muscle_group}")
        
        # Global exercise ID tracker to avoid duplicates across all days
        global_used_ids = set()
        modified_days = []
        
        for day_key in target_day_keys:
            if day_key not in days:
                continue
                
            day_data = days[day_key].copy()
            
            if operation == 'replace':
                # Replace all exercises with new muscle group exercises
                new_exercises = []
                day_used_ids = set()
                
                # Try each muscle target to get diverse exercises
                exercises_needed = 6 # Target 6 exercises per day
                
                for muscle_target in muscle_targets:
                    if len(new_exercises) >= exercises_needed:
                        break
                        
                    print(f"🔍 Looking for {muscle_target} exercises...")
                    exercise_ids = pick_from_muscles([muscle_target], cat, used_ids=global_used_ids.union(day_used_ids), n=3)
                    
                    for eid in exercise_ids:
                        if len(new_exercises) >= exercises_needed:
                            break
                            
                        if eid in cat['by_id'] and eid not in global_used_ids and eid not in day_used_ids:
                            exercise_data = cat['by_id'][eid]
                            print(f"✅ Adding exercise: {exercise_data['name']} (ID: {eid})")
                            
                            new_exercises.append({
                                'id': eid,
                                'name': exercise_data['name'],
                                'sets': 3,
                                'reps': 10,
                                'note': None
                            })
                            day_used_ids.add(eid)
                            global_used_ids.add(eid)
                
                # If we didn't get enough exercises, try without the global restriction
                if len(new_exercises) < 2:
                    print(f"⚠️ Only got {len(new_exercises)} exercises, trying without global restrictions...")
                    for muscle_target in muscle_targets:
                        if len(new_exercises) >= exercises_needed:
                            break
                        exercise_ids = pick_from_muscles([muscle_target], cat, used_ids=day_used_ids, n=5)
                        for eid in exercise_ids:
                            if len(new_exercises) >= exercises_needed:
                                break
                            if eid in cat['by_id'] and eid not in day_used_ids:
                                exercise_data = cat['by_id'][eid]
                                new_exercises.append({
                                    'id': eid,
                                    'name': exercise_data['name'],
                                    'sets': 3,
                                    'reps': 10,
                                    'note': None
                                })
                                day_used_ids.add(eid)
                
                day_data['exercises'] = new_exercises
                day_data['muscle_groups'] = muscle_targets
                print(f"📝 Day {day_key}: {len(new_exercises)} exercises added")
                
            elif operation == 'add':
                # Add one exercise from the muscle group
                existing_exercises = day_data.get('exercises', [])
                if len(existing_exercises) >= 8:
                    continue  # Skip if day is full
                
                used_ids = set(ex.get('id') for ex in existing_exercises if ex.get('id'))
                
                for muscle_target in muscle_targets:
                    exercise_ids = pick_from_muscles([muscle_target], cat, used_ids=used_ids, n=1)
                    if exercise_ids and exercise_ids[0] in cat['by_id']:
                        eid = exercise_ids[0]
                        exercise_data = cat['by_id'][eid]
                        new_exercise = {
                            'id': eid,
                            'name': exercise_data['name'],
                            'sets': 3,
                            'reps': 10,
                            'note': None
                        }
                        existing_exercises.append(new_exercise)
                        day_data['exercises'] = existing_exercises
                        
                        # Update muscle groups if not already included
                        current_muscles = set(day_data.get('muscle_groups', []))
                        current_muscles.update(muscle_targets)
                        day_data['muscle_groups'] = list(current_muscles)
                        break
            
            days[day_key] = day_data
            modified_days.append(day_key.title())
        
        updated['days'] = days
        
        if operation == 'replace':
            summary = f"Changed {len(modified_days)} days to focus on {muscle_group} exercises: {', '.join(modified_days)}"
        else:
            summary = f"Added {muscle_group} exercises to {len(modified_days)} days: {', '.join(modified_days)}"
        
        return updated, summary

    @classmethod
    def create_muscle_specific_template(cls, template_names: list, muscle_distributions: dict, db = None) -> Tuple[dict, str]:
        """Create template with specific muscle distributions - FIXED VERSION"""
        if not db:
            return {}, "Database connection required"
        
        from .exercise_catalog_db import load_catalog, pick_from_muscles
        cat = load_catalog(db)
        if not cat:
            return {}, "Could not load exercise database"
        
        # FIXED: Better muscle mapping with exact database terms
        muscle_mapping = {
            'legs': ['legs', 'quadriceps', 'hamstrings', 'glutes', 'calves', 'lower body'],
            'leg': ['legs', 'quadriceps', 'hamstrings', 'glutes', 'calves', 'lower body'],
            'upper': ['chest', 'back', 'shoulders', 'arms', 'biceps', 'triceps', 'upper body'],
            'uper': ['chest', 'back', 'shoulders', 'arms', 'biceps', 'triceps', 'upper body'],
            'chest': ['chest', 'pectorals'],
            'back': ['back', 'lats'],
            'shoulders': ['shoulders', 'deltoids'],
            'arms': ['arms', 'biceps', 'triceps'],
            'core': ['core', 'abs', 'abdominal'],
            'cardio': ['cardio', 'aerobic']
        }
        
        template = {
            "name": f"Custom Muscle Split ({len(template_names)} days)",
            "goal": "muscle_gain",
            "days": {},
            "notes": []
        }
        
        day_index = 0
        used_exercise_count = {}  # Track how many times each exercise is used

        for muscle_group, day_count in muscle_distributions.items():
            muscle_targets = muscle_mapping.get(muscle_group, [muscle_group])
            
            for i in range(min(day_count, len(template_names) - day_index)):
                if day_index >= len(template_names):
                    break
                
                day_name = template_names[day_index] if day_index < len(template_names) else f"Day {day_index + 1}"
                day_key = day_name.lower().replace(' ', '_')
                
                # Create appropriate day title
                if muscle_group == 'legs':
                    day_title = "Leg Day"
                elif muscle_group == 'chest':
                    day_title = "Chest Day"
                elif muscle_group == 'back':
                    day_title = "Back Day"
                else:
                    day_title = f"{muscle_group.title()} Day"
                
                # Get ALL exercises for this specific muscle group
                all_muscle_exercises = []
                
                if muscle_group in ['legs', 'leg']:
                    leg_exercise_names = [
                        'squat', 'lunge', 'leg press', 'leg extension', 'leg curl', 
                        'calf raise', 'bulgarian split squat', 'step up', 'wall sit',
                        'goblet squat', 'romanian deadlift', 'glute bridge', 'hip thrust'
                    ]
                    
                    for eid, exercise_data in cat["by_id"].items():
                        exercise_name = exercise_data["name"].lower()
                        is_leg_exercise = any(leg_name in exercise_name for leg_name in leg_exercise_names)
                        if is_leg_exercise:
                            all_muscle_exercises.append((eid, exercise_data))
                
                elif muscle_group in ['chest']:
                    chest_exercise_names = [
                        'bench press', 'chest press', 'push up', 'chest fly', 'chest flye',
                        'incline press', 'decline press', 'dips', 'pec fly'
                    ]
                    
                    for eid, exercise_data in cat["by_id"].items():
                        exercise_name = exercise_data["name"].lower()
                        is_chest_exercise = any(chest_name in exercise_name for chest_name in chest_exercise_names)
                        if is_chest_exercise:
                            all_muscle_exercises.append((eid, exercise_data))
                
                elif muscle_group in ['back']:
                    back_exercise_names = [
                        'pull up', 'lat pulldown', 'row', 'deadlift', 'shrug',
                        'chin up', 'cable row', 't-bar row'
                    ]
                    
                    for eid, exercise_data in cat["by_id"].items():
                        exercise_name = exercise_data["name"].lower()
                        is_back_exercise = any(back_name in exercise_name for back_name in back_exercise_names)
                        if is_back_exercise:
                            all_muscle_exercises.append((eid, exercise_data))
                
                # Add more muscle groups as needed...
                
                # Sort exercises by usage count (least used first)
                all_muscle_exercises.sort(key=lambda x: used_exercise_count.get(x[0], 0))
                
                # Select 6 exercises, preferring less-used ones
                exercises = []
                for eid, exercise_data in all_muscle_exercises:
                    if len(exercises) >= 6:
                        break
                    
                    exercises.append({
                        'id': eid,
                        'name': exercise_data['name'],
                        'sets': 3,
                        'reps': 10,
                        'note': None
                    })
                    
                    # Track usage
                    used_exercise_count[eid] = used_exercise_count.get(eid, 0) + 1
                
                # If we don't have enough muscle-specific exercises, repeat the available ones
                if len(exercises) < 6 and all_muscle_exercises:
                    while len(exercises) < 6:
                        for eid, exercise_data in all_muscle_exercises:
                            if len(exercises) >= 6:
                                break
                            exercises.append({
                                'id': eid,
                                'name': exercise_data['name'],
                                'sets': 3,
                                'reps': 10,
                                'note': None
                            })
                            used_exercise_count[eid] = used_exercise_count.get(eid, 0) + 1
                
                print(f"📝 Day {day_index + 1} ({day_key}): {[ex['name'] for ex in exercises]} - Pure {muscle_group}")
                
                template['days'][day_key] = {
                    'title': day_title,
                    'muscle_groups': [muscle_group.title()],
                    'exercises': exercises
                }
                
                day_index += 1
        
        summary = f"Created {muscle_group} workout with {len(exercises)} exercises"
        return template, summary


# NEW: import DB catalog helpers
from .exercise_catalog_db import load_catalog, id_for_name, pick_from_muscles
DAYS6 = ["monday","tuesday","wednesday","thursday","friday","saturday"]
DAYS  = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
# ────────────────────────── INTENT ──────────────────────────
_TRIGGER_WORDS = {
    "workout template","training template","create template","make template",
    "build plan","create plan","workout plan","training plan","routine","program",
    "upper lower","push pull legs","ppl","full body","muscle group","split"
}


def is_workout_template_intent(t: str) -> bool:
    tt = (t or "").lower()
    return any(k in tt for k in _TRIGGER_WORDS)
# ────────────────────── RENDER (Markdown) ───────────────────
def render_markdown_from_template(tpl: Dict[str,Any]) -> str:
    """Render template with dynamic day names."""
    name = tpl.get("name") or "Workout Template"
    goal = (tpl.get("goal") or "").replace("_"," ").title()
    days  = tpl.get("days") or {}
    notes = tpl.get("notes") or []
    out = [f"# {name}"]
    if goal:
        out += [f"**Goal:** {goal}", ""]
    # Get all day keys from the template and sort them for consistent rendering
    day_keys = list(days.keys())
    # Try to maintain a sensible order if possible
    day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day_keys.sort(key=lambda x: day_order.index(x) if x in day_order else len(day_order))
    for d in day_keys:
        if d in days:
            day = days[d] or {}
            split_title = (day.get("title") or "").strip()
            heading = f"{d.title()}" + (f" — {split_title}" if split_title else "")
            out.append(f"## {heading}")
            mgs = day.get("muscle_groups") or []
            if mgs:
                out.append(f"**Muscle Focus:** {', '.join(mgs)}")
            for ex in day.get("exercises") or []:
                nm   = ex.get("name") or "Exercise"
                sets = ex.get("sets")
                reps = ex.get("reps")
                note = ex.get("note")
                line = f"- {nm}"
                if sets is not None and reps is not None:
                    line += f" — {sets}×{reps}"
                elif sets is not None:
                    line += f" — {sets} sets"
                if note:
                    line += f" ({note})"
                out.append(line)
            out.append("")
    if notes:
        out.append("**Notes**")
        for n in notes: out.append(f"- {n}")
        out.append("")
    return "\n".join(out).strip()
# ─────────────────────── Utilities ──────────────────────────
def _safe_json(text: str, fallback: Dict[str,Any]) -> Dict[str,Any]:
    try:
        return orjson.loads(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return fallback
def _template_skeleton_mon_sat() -> Dict[str,Any]:
    return {
        "name": "Template (Mon–Sat)",
        "goal": "muscle_gain",
        "days": {d: {"title": d.title(), "muscle_groups": [], "exercises": []} for d in DAYS6},
        "notes": [],
    }
# ─────────────── Catalog Gate (backed by DB) ────────────────
def _enforce_catalog_on_template_db(tpl: Dict[str,Any], db: Session) -> Dict[str,Any]:
    """
    Ensure every exercise comes from qr_code.
    If name unknown → pick sensible replacement from day's muscle groups.
    Result items will be: {id, name, sets, reps, note}
    """
    cat = load_catalog(db)
    days = tpl.get("days") or {}
    for d in DAYS6:
        day = days.get(d) or {}
        muscles = (day.get("muscle_groups") or [])
        used: set[int] = set()
        normalized_list: List[Dict[str,Any]] = []
        for ex in (day.get("exercises") or []):
            eid = ex.get("id") if isinstance(ex, dict) else None
            nm  = ex.get("name") if isinstance(ex, dict) else None
            chosen_id = None
            if isinstance(eid, int) and eid in cat["by_id"]:
                chosen_id = eid
            elif nm:
                nid = id_for_name(nm, cat)
                if nid:
                    chosen_id = nid
            if not chosen_id:
                picked = pick_from_muscles(muscles, cat, used_ids=used, n=1)
                chosen_id = picked[0] if picked else None
            if chosen_id:
                used.add(chosen_id)
                canon = cat["by_id"][chosen_id]
                normalized_list.append({
                    "id":   chosen_id,
                    "name": canon["name"],
                    "sets": ex.get("sets"),
                    "reps": ex.get("reps"),
                    "note": ex.get("note"),
                })
            # else: drop if nothing found
        day["exercises"] = normalized_list
        days[d] = day
    tpl["days"] = days
    return tpl
def build_id_only_structure(tpl: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Produce an ids-only structure by day keys in template:
        {"monday":[...ids...], ..., "saturday":[...ids...]} or custom day names
    """
    out: Dict[str, List[int]] = {}
    days = tpl.get("days") or {}
    # Use whatever day keys are actually in the template (handles custom day names)
    for d in days.keys():
        ids: List[int] = []
        for ex in (days.get(d, {}).get("exercises") or []):
            eid = ex.get("id")
            if isinstance(eid, int):
                ids.append(eid)
        out[d] = ids
    return out
def _template_skeleton_dynamic(template_names: list) -> Dict[str,Any]:
    """Generate skeleton for dynamic template names"""
    days = {}
    for name in template_names:
        day_key = name.lower()
        days[day_key] = {"title": name.title(), "muscle_groups": [], "exercises": []}
    return {
        "name": f"Workout Template ({len(template_names)} days)",
        "goal": "muscle_gain",
        "days": days,
        "notes": []
    }
def _enforce_catalog_on_template_db_dynamic(tpl: Dict[str, Any], db: Session, template_names: list) -> Dict[str, Any]:
    """Dynamic version of catalog enforcement with 6-exercise minimum"""
    print("🔍 DEBUG - Starting catalog enforcement")
    from .exercise_catalog_db import load_catalog, id_for_name, pick_from_muscles

    print("🔍 DEBUG - About to load catalog")
    cat = load_catalog(db)
    print(f"🔍 DEBUG - Catalog loaded: {type(cat)}, keys: {cat.keys() if cat else 'None'}")

    if not cat or "by_id" not in cat:
        print("🔍 DEBUG - No catalog available, returning original template")
        return tpl
    
    days = tpl.get("days") or {}
    global_used = set()  # Track globally used exercises
    
    for name in template_names:
        day_key = name.lower()
        if day_key not in days:
            continue
        
        day = days[day_key] or {}
        muscles = day.get("muscle_groups") or []
        normalized_list = []
        day_used = set()
        
        # Process existing exercises first
        for ex in (day.get("exercises") or []):
            eid = ex.get("id") if isinstance(ex, dict) else None
            nm  = ex.get("name") if isinstance(ex, dict) else None
            chosen_id = None
            
            if isinstance(eid, int) and eid in cat["by_id"]:
                chosen_id = eid
            elif nm:
                nid = id_for_name(nm, cat)
                if nid:
                    chosen_id = nid
            
            if not chosen_id:
                picked = pick_from_muscles(muscles, cat, used_ids=global_used.union(day_used), n=1)
                chosen_id = picked[0] if picked else None
            
            if chosen_id and chosen_id not in day_used:
                day_used.add(chosen_id)
                global_used.add(chosen_id)
                canon = cat["by_id"][chosen_id]
                normalized_list.append({
                    "id":   chosen_id,
                    "name": canon["name"],
                    "sets": ex.get("sets") or 3,
                    "reps": ex.get("reps") or 10,
                    "note": ex.get("note"),
                })
        
        # ENFORCE 6-EXERCISE MINIMUM (with infinite loop protection)
        attempt_count = 0
        max_attempts = 20  # Prevent infinite loops
        while len(normalized_list) < 6 and attempt_count < max_attempts:
            attempt_count += 1
            picked = pick_from_muscles(muscles or ["full body"], cat, used_ids=global_used.union(day_used), n=1)
            if picked and picked[0] in cat["by_id"]:
                eid = picked[0]
                canon = cat["by_id"][eid]
                normalized_list.append({
                    "id": eid,
                    "name": canon["name"],
                    "sets": 3,
                    "reps": 10,
                    "note": None,
                })
                day_used.add(eid)
                global_used.add(eid)
            else:
                # Fallback: use any available exercise
                available_ids = [id for id in cat["by_id"].keys() if id not in day_used]
                if available_ids:
                    eid = available_ids[0]
                    canon = cat["by_id"][eid]
                    normalized_list.append({
                        "id": eid,
                        "name": canon["name"],
                        "sets": 3,
                        "reps": 10,
                        "note": None,
                    })
                    day_used.add(eid)
                    global_used.add(eid)
                else:
                    break  # No more exercises available

        # If still not enough exercises, create basic ones
        if len(normalized_list) < 3:
            basic_exercises = [
                {"id": 9999, "name": "Push-ups", "sets": 3, "reps": 10, "note": None},
                {"id": 9998, "name": "Squats", "sets": 3, "reps": 12, "note": None},
                {"id": 9997, "name": "Plank", "sets": 3, "reps": "30 seconds", "note": None},
            ]
            for basic_ex in basic_exercises:
                if len(normalized_list) < 6:
                    normalized_list.append(basic_ex)
        
        # ENFORCE 8-EXERCISE MAXIMUM
        if len(normalized_list) > 8:
            normalized_list = normalized_list[:8]
        
        day["exercises"] = normalized_list
        days[day_key] = day
    
    tpl["days"] = days
    return tpl
# ───────────────── LLM: generate from profile ───────────────
def generate_system_prompt(template_names: list) -> str:
    day_schema = []
    for name in template_names:
        day_schema.append(f'      "{name.lower()}": {{"title": string, "muscle_groups": string[], "exercises":[{{"name":string,"sets":int|null,"reps":string|int|null,"note":string|null}}]}}')
    return (
        "You are a certified strength & conditioning coach. "
        "Output ONLY strict JSON with this schema:\n"
        "{\n"
        '  "template": {\n'
        '    "name": string,\n'
        '    "goal": "muscle_gain" | "fat_loss" | "strength" | "performance",\n'
        '    "days": {\n'
        + ',\n'.join(day_schema) + '\n'
        "    },\n"
        '    "notes": string[]\n'
        "  },\n"
        '  "rationale": string\n'
        "}\n"
        f"- Create exactly {len(template_names)} workout days with the specified names.\n"
        + "- Each day MUST have exactly 6 exercises (this is mandatory for optimal workout volume).\n"
        "- Choose a sensible split that works well with the given template names.\n"
        "- Use common gym names; system will map to a fixed exercise catalog.\n"
        "- No markdown, ONLY JSON."
    )

def llm_generate_template_from_profile_database_only(oai, model: str, profile: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    """Generate template using ONLY database exercises - new database-first approach"""
    print("🔍 DEBUG - Using database-first template generation")

    try:
        from .database_exercise_manager import DatabaseExerciseManager

        # Extract profile information
        template_names = profile.get("template_names", ["Day 1", "Day 2", "Day 3"])
        template_count = len(template_names)
        goal = profile.get("client_goal", "muscle gain")

        # Define muscle group mappings for different goals
        muscle_group_programs = {
            "muscle gain": ["chest", "back", "legs", "shoulders", "biceps", "triceps"],
            "weight loss": ["full body", "cardio", "legs", "core"],
            "strength": ["chest", "back", "legs", "shoulders"],
            "endurance": ["cardio", "full body", "legs", "core"]
        }

        # Get appropriate muscle groups for the goal
        target_muscle_groups = muscle_group_programs.get(goal.lower(), ["chest", "back", "legs", "shoulders"])

        # Create template structure
        template = {
            "name": f"Database-First {template_count}-Day Program",
            "goal": goal,
            "days": {}
        }

        # Generate exercises for each day
        for i, day_name in enumerate(template_names):
            day_key = day_name.lower().replace(' ', '_').replace('-', '_')

            # Assign muscle groups cyclically
            assigned_muscle_groups = []
            exercises_per_day = max(3, 6 // template_count)  # At least 3 exercises per day

            if template_count <= len(target_muscle_groups):
                # One muscle group per day
                muscle_group = target_muscle_groups[i % len(target_muscle_groups)]
                assigned_muscle_groups = [muscle_group]
            else:
                # Multiple muscle groups per day
                groups_per_day = max(1, len(target_muscle_groups) // template_count)
                start_idx = (i * groups_per_day) % len(target_muscle_groups)
                assigned_muscle_groups = target_muscle_groups[start_idx:start_idx + groups_per_day]

            # Get exercises from database for these muscle groups
            day_exercises = []
            for muscle_group in assigned_muscle_groups:
                muscle_exercises = DatabaseExerciseManager.get_available_exercises_by_muscle(db, muscle_group)

                # Select exercises for this muscle group
                exercises_for_muscle = min(exercises_per_day // len(assigned_muscle_groups) + 1, len(muscle_exercises))
                selected_exercises = muscle_exercises[:exercises_for_muscle]

                for exercise in selected_exercises:
                    # Add default sets/reps
                    exercise_copy = exercise.copy()
                    if exercise.get('isCardio'):
                        exercise_copy['sets'] = 1
                        exercise_copy['reps'] = '20 minutes'
                    else:
                        exercise_copy['sets'] = 3
                        exercise_copy['reps'] = 12 if exercise.get('isBodyWeight') else 10

                    day_exercises.append(exercise_copy)

            # Ensure minimum exercises per day
            while len(day_exercises) < 3:
                # Add more exercises from any available muscle group
                additional_muscle = target_muscle_groups[len(day_exercises) % len(target_muscle_groups)]
                additional_exercises = DatabaseExerciseManager.get_available_exercises_by_muscle(db, additional_muscle)

                if additional_exercises:
                    # Find an exercise not already in the day
                    existing_names = [ex['name'] for ex in day_exercises]
                    for exercise in additional_exercises:
                        if exercise['name'] not in existing_names:
                            exercise_copy = exercise.copy()
                            exercise_copy['sets'] = 3
                            exercise_copy['reps'] = 12 if exercise.get('isBodyWeight') else 10
                            day_exercises.append(exercise_copy)
                            break
                else:
                    break  # No more exercises available

            template["days"][day_key] = {
                "title": day_name.title(),
                "muscle_groups": assigned_muscle_groups,
                "exercises": day_exercises[:6]  # Limit to 6 exercises per day
            }

            print(f"📝 Day {i+1} ({day_name}): {len(day_exercises)} database exercises")

        return template, "Generated using only database exercises"

    except Exception as e:
        print(f"❌ Database-first generation failed: {e}")
        # Return empty template rather than fallback
        return {
            "name": "Empty Template",
            "goal": "muscle gain",
            "days": {}
        }, f"Generation failed: {e}"


def enhanced_edit_template_database_only(oai, model: str, template: Dict[str, Any], user_instruction: str, profile: Dict[str, Any], db: Session, validation_result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Enhanced edit function that handles all editing scenarios with database-only exercises"""
    print("🔍 DEBUG - Using comprehensive database-only template editing")
    print(f"🔍 DEBUG - User instruction: '{user_instruction}'")
    print(f"🔍 DEBUG - Template structure keys: {list(template.keys())}")
    print(f"🔍 DEBUG - Days in template: {list(template.get('days', {}).keys())}")

    try:
        from .database_exercise_manager import DatabaseExerciseManager
        from .ai_exercise_validator import AIExerciseValidator

        # Make a deep copy of the template to work with
        new_template = template.copy()
        new_template['days'] = {}
        for day_key, day_data in template.get('days', {}).items():
            new_template['days'][day_key] = day_data.copy()
            if 'exercises' in day_data:
                new_template['days'][day_key]['exercises'] = [ex.copy() for ex in day_data['exercises']]

        edit_summary = []
        instruction_lower = user_instruction.lower()

        # Parse the instruction using AI to understand intent
        editing_intent = _parse_editing_intent(oai, model, user_instruction)
        print(f"🔍 DEBUG - Parsed intent: {editing_intent}")

        # Track if any changes were made
        changes_made = False

        # 1. Handle day name changes (priority check for rename patterns)
        if editing_intent.get('action') == 'rename_day' or any(phrase in instruction_lower for phrase in ['rename', 'change day', 'change name', 'call day']):
            result = _handle_day_rename(new_template, user_instruction, editing_intent)
            if result['success']:
                edit_summary.append(result['message'])
                changes_made = True
            else:
                edit_summary.append(result['message'])

        # 2. Handle exercise addition to specific days or all days
        elif editing_intent.get('action') == 'add_exercise' or any(word in instruction_lower for word in ['add', 'include', 'give me', 'more']):
            if validation_result.get('validated_exercises'):
                result = _handle_exercise_addition(new_template, user_instruction, validation_result['validated_exercises'], editing_intent)
                edit_summary.extend(result['messages'])
                if result['messages']:
                    changes_made = True
            else:
                # Handle muscle group requests
                result = _handle_muscle_group_addition(new_template, user_instruction, db, oai, model, editing_intent)
                edit_summary.extend(result['messages'])
                if result['messages']:
                    changes_made = True

        # 3. Handle exercise replacement/alternation
        elif editing_intent.get('action') == 'replace_exercise' or any(word in instruction_lower for word in ['replace', 'swap', 'alternate']) or ('change' in instruction_lower and any(word in instruction_lower for word in ['exercise', 'with', 'to'])):
            result = _handle_exercise_replacement(new_template, user_instruction, validation_result, db, oai, model, editing_intent)
            edit_summary.extend(result['messages'])
            if result['messages']:
                changes_made = True

        # 4. Handle exercise removal
        elif editing_intent.get('action') == 'remove_exercise' or any(word in instruction_lower for word in ['remove', 'delete', 'take out']):
            result = _handle_exercise_removal(new_template, user_instruction, editing_intent)
            edit_summary.extend(result['messages'])
            if result['messages']:
                changes_made = True

        # 5. Handle day modifications (make harder, easier, etc.)
        elif editing_intent.get('action') == 'modify_difficulty' or any(word in instruction_lower for word in ['harder', 'easier', 'more reps', 'less reps']):
            result = _handle_difficulty_modification(new_template, user_instruction, editing_intent)
            edit_summary.extend(result['messages'])
            if result['messages']:
                changes_made = True

        # 6. If no specific edits were made, provide guidance
        if not edit_summary:
            edit_summary.append("I understand you want to make changes. Could you be more specific? For example: 'add bench press to all days', 'change Monday to Push Day', 'replace squats with lunges', etc.")

        # Final validation - ensure all exercises are still from database
        final_template = _validate_final_template_exercises(new_template, db)

        # Ensure the template maintains proper structure for dynamic day key storage
        final_template = _ensure_template_structure_compatibility(final_template)

        print(f"🔍 DEBUG - Changes made: {changes_made}")
        print(f"🔍 DEBUG - Final template days: {list(final_template.get('days', {}).keys())}")

        return final_template, '; '.join(edit_summary)

    except Exception as e:
        print(f"❌ Database-only edit failed: {e}")
        import traceback
        traceback.print_exc()
        return template, f"Edit failed: {e}"


def _parse_editing_intent(oai, model: str, user_instruction: str) -> Dict[str, Any]:
    """Use AI to parse the user's editing intent"""
    system_prompt = """You are parsing workout template editing instructions. Determine the user's intent and extract key information.

    ACTIONS:
    - "add_exercise": User wants to add exercises
    - "remove_exercise": User wants to remove exercises
    - "replace_exercise": User wants to replace/swap exercises
    - "rename_day": User wants to change day names
    - "modify_difficulty": User wants to change difficulty/reps/sets
    - "unknown": Intent unclear

    TARGETS:
    - "all_days": Apply to all days
    - "specific_day": Apply to specific day
    - "muscle_group": Target specific muscle group

    Extract:
    - target_day: specific day mentioned (if any)
    - exercise_names: specific exercises mentioned
    - muscle_groups: muscle groups mentioned
    - new_name: new name for day (if renaming)
    - scope: "all" or "specific"

    Respond in JSON: {"action": "...", "target": "...", "target_day": "...", "exercise_names": [...], "muscle_groups": [...], "new_name": "...", "scope": "..."}"""

    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse this instruction: '{user_instruction}'"}
            ],
            temperature=0.1
        )

        import json
        result = json.loads(resp.choices[0].message.content or "{}")
        return result

    except Exception as e:
        print(f"AI intent parsing failed: {e}")
        # Fallback parsing
        return _fallback_parse_intent(user_instruction)


def _fallback_parse_intent(user_instruction: str) -> Dict[str, Any]:
    """Enhanced fallback intent parsing using patterns"""
    instruction_lower = user_instruction.lower()

    # Determine action with more nuanced patterns
    if any(word in instruction_lower for word in ['rename', 'call']) or \
       ('change' in instruction_lower and any(word in instruction_lower for word in ['day', 'name'])) or \
       ('day' in instruction_lower and 'should be' in instruction_lower):
        action = 'rename_day'
    elif any(word in instruction_lower for word in ['add', 'include', 'more', 'give', 'want']) and \
         not ('remove' in instruction_lower or 'delete' in instruction_lower):
        action = 'add_exercise'
    elif any(word in instruction_lower for word in ['remove', 'delete', 'take out', 'hate']) or \
         ('get rid' in instruction_lower):
        action = 'remove_exercise'
    elif any(word in instruction_lower for word in ['replace', 'swap', 'substitute', 'alternate', 'switch']) or \
         ('change' in instruction_lower and any(word in instruction_lower for word in ['with', 'to', 'for'])):
        action = 'replace_exercise'
    elif any(word in instruction_lower for word in ['harder', 'easier', 'difficult', 'intense']) or \
         ('make' in instruction_lower and any(word in instruction_lower for word in ['better', 'tougher'])):
        action = 'modify_difficulty'
    else:
        action = 'unknown'

    # Determine scope with more patterns
    scope = 'all' if any(phrase in instruction_lower for phrase in [
        'all days', 'every day', 'each day', 'all of them', 'everywhere'
    ]) else 'specific'

    # Extract potential day references
    target_day = None
    day_patterns = {
        'day 1': 'day_1', 'day 2': 'day_2', 'day 3': 'day_3',
        'monday': 'monday', 'tuesday': 'tuesday', 'wednesday': 'wednesday',
        'thursday': 'thursday', 'friday': 'friday', 'saturday': 'saturday', 'sunday': 'sunday'
    }

    for pattern, normalized in day_patterns.items():
        if pattern in instruction_lower:
            target_day = pattern
            break

    # Extract potential new name for renaming
    new_name = None
    if action == 'rename_day':
        import re
        name_patterns = [
            r'name as ([^\.]+)',
            r'to ([^\.]+)',
            r'call.*?([^\.]+)',
            r'should be ([^\.]+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                new_name = match.group(1).strip()
                break

    return {
        'action': action,
        'scope': scope,
        'target_day': target_day,
        'exercise_names': [],
        'muscle_groups': [],
        'new_name': new_name
    }


def _handle_day_rename(template: Dict[str, Any], user_instruction: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle day renaming requests"""
    import re

    print(f"🔍 DEBUG - Day rename request: '{user_instruction}'")
    print(f"🔍 DEBUG - Intent data: {intent}")

    # First try to use the parsed intent
    old_name = intent.get('target_day', '')
    new_name = intent.get('new_name', '')

    # If intent parsing didn't work, fall back to regex patterns
    if not old_name or not new_name:
        rename_patterns = [
            r'rename\s+([^t]+)\s+to\s+([^\.]+)',
            r'change\s+([^t]+)\s+to\s+([^\.]+)',
            r'call\s+([^a]+)\s+([^\.]+)',
            r'change\s+([^n]+)\s+name\s+(?:as|to)\s+([^\.]+)',
        ]

        for pattern in rename_patterns:
            match = re.search(pattern, user_instruction.lower())
            if match:
                old_name = match.group(1).strip()
                new_name = match.group(2).strip()
                break

    if not old_name or not new_name:
        return {'success': False, 'message': 'Could not understand which day to rename. Try: "rename Monday to Push Day"'}

    print(f"🔍 DEBUG - Extracted: old_name='{old_name}', new_name='{new_name}'")

    # Find the day to rename - handle both "day 2" and "day_2" formats
    day_keys = list(template['days'].keys())
    target_day_key = None

    print(f"🔍 DEBUG - Available day keys: {day_keys}")

    # Normalize the old_name for comparison
    old_name_normalized = old_name.lower().strip()

    for day_key in day_keys:
        day_title = template['days'][day_key].get('title', '').lower()
        day_key_lower = day_key.lower()

        print(f"🔍 DEBUG - Checking day_key='{day_key}' (title='{day_title}') against '{old_name_normalized}'")

        # Check various matching patterns
        matches = [
            old_name_normalized in day_key_lower,
            old_name_normalized in day_title,
            old_name_normalized.replace(' ', '_') == day_key_lower,
            old_name_normalized.replace('_', ' ') in day_key_lower.replace('_', ' '),
            # Handle "day 2" -> "day_2" conversion
            old_name_normalized.replace(' ', '_') in day_key_lower,
            # Handle numeric day references like "day 2" -> "day_2"
            f"day_{old_name_normalized.split()[-1]}" == day_key_lower if 'day' in old_name_normalized else False,
        ]

        if any(matches):
            target_day_key = day_key
            print(f"🔍 DEBUG - Found matching day: {day_key}")
            break

    if target_day_key:
        # Rename the day
        template['days'][target_day_key]['title'] = new_name.title()
        print(f"🔍 DEBUG - Successfully renamed {target_day_key} to '{new_name.title()}'")
        return {'success': True, 'message': f"Renamed {old_name} to {new_name.title()}"}
    else:
        print(f"🔍 DEBUG - Could not find day to rename")
        return {'success': False, 'message': f"Could not find day '{old_name}' to rename"}


def _handle_exercise_addition(template: Dict[str, Any], user_instruction: str, validated_exercises: List[Dict], intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle adding exercises to days"""
    messages = []
    instruction_lower = user_instruction.lower()

    print(f"🔍 DEBUG - Exercise addition: '{user_instruction}'")
    print(f"🔍 DEBUG - Validated exercises: {[ex.get('name') for ex in validated_exercises]}")
    print(f"🔍 DEBUG - Template days: {list(template['days'].keys())}")

    # Determine if adding to all days or specific day
    add_to_all = any(phrase in instruction_lower for phrase in ['all days', 'every day', 'each day', 'to all'])

    # Find specific day if mentioned
    target_day = None
    for day_key in template['days'].keys():
        if day_key.lower() in instruction_lower or template['days'][day_key].get('title', '').lower() in instruction_lower:
            target_day = day_key
            break

    print(f"🔍 DEBUG - Add to all: {add_to_all}, Target day: {target_day}")

    if add_to_all:
        # Add to all days
        for exercise_data in validated_exercises:
            for day_key, day_data in template['days'].items():
                current_exercises = day_data.get('exercises', [])

                # Check if exercise already exists in this day
                exercise_exists = any(ex.get('name', '').lower() == exercise_data['name'].lower() for ex in current_exercises)

                if not exercise_exists:
                    new_exercise = exercise_data.copy()
                    new_exercise['sets'] = 3
                    new_exercise['reps'] = 12 if exercise_data.get('isBodyWeight') else 10
                    current_exercises.append(new_exercise)
                    template['days'][day_key]['exercises'] = current_exercises
                    print(f"🔍 DEBUG - Added {exercise_data['name']} to {day_key}")

            messages.append(f"Added {exercise_data['name']} to all days")

    elif target_day:
        # Add to specific day
        for exercise_data in validated_exercises:
            current_exercises = template['days'][target_day].get('exercises', [])

            # Check if exercise already exists
            exercise_exists = any(ex.get('name', '').lower() == exercise_data['name'].lower() for ex in current_exercises)

            if not exercise_exists:
                new_exercise = exercise_data.copy()
                new_exercise['sets'] = 3
                new_exercise['reps'] = 12 if exercise_data.get('isBodyWeight') else 10
                current_exercises.append(new_exercise)
                template['days'][target_day]['exercises'] = current_exercises
                messages.append(f"Added {exercise_data['name']} to {template['days'][target_day].get('title', target_day)}")
                print(f"🔍 DEBUG - Added {exercise_data['name']} to {target_day}")
            else:
                messages.append(f"{exercise_data['name']} already exists in {template['days'][target_day].get('title', target_day)}")

    else:
        # Add to first day as default
        day_keys = list(template['days'].keys())
        if day_keys:
            target_day = day_keys[0]
            for exercise_data in validated_exercises:
                current_exercises = template['days'][target_day].get('exercises', [])
                new_exercise = exercise_data.copy()
                new_exercise['sets'] = 3
                new_exercise['reps'] = 12 if exercise_data.get('isBodyWeight') else 10
                current_exercises.append(new_exercise)
                template['days'][target_day]['exercises'] = current_exercises
                messages.append(f"Added {exercise_data['name']} to {template['days'][target_day].get('title', target_day)}")
                print(f"🔍 DEBUG - Added {exercise_data['name']} to {target_day} (default)")

    print(f"🔍 DEBUG - Addition results: {messages}")
    return {'messages': messages}


def _handle_muscle_group_addition(template: Dict[str, Any], user_instruction: str, db: Session, oai, model: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle adding exercises by muscle group"""
    from .ai_exercise_validator import AIExerciseValidator
    messages = []
    instruction_lower = user_instruction.lower()

    # Extract muscle groups
    muscle_groups = ['chest', 'back', 'legs', 'shoulders', 'arms', 'core', 'biceps', 'triceps', 'abs']
    found_muscles = [muscle for muscle in muscle_groups if muscle in instruction_lower]

    if not found_muscles:
        return {'messages': ['Could not identify which muscle group exercises to add']}

    # Determine if adding to all days
    add_to_all = any(phrase in instruction_lower for phrase in ['all days', 'every day', 'each day', 'to all'])

    for muscle in found_muscles:
        suggested_exercises = AIExerciseValidator.suggest_muscle_group_exercises(oai, model, muscle, db, count=2)

        if suggested_exercises:
            if add_to_all:
                # Add to all days
                for day_key, day_data in template['days'].items():
                    current_exercises = day_data.get('exercises', [])
                    for exercise in suggested_exercises:
                        # Check if exercise already exists
                        exercise_exists = any(ex.get('name', '').lower() == exercise['name'].lower() for ex in current_exercises)
                        if not exercise_exists:
                            new_exercise = exercise.copy()
                            new_exercise['sets'] = 3
                            new_exercise['reps'] = 12 if exercise.get('isBodyWeight') else 10
                            current_exercises.append(new_exercise)
                    template['days'][day_key]['exercises'] = current_exercises

                exercise_names = [ex['name'] for ex in suggested_exercises]
                messages.append(f"Added {muscle} exercises ({', '.join(exercise_names)}) to all days")
            else:
                # Add to first day
                day_keys = list(template['days'].keys())
                if day_keys:
                    target_day = day_keys[0]
                    current_exercises = template['days'][target_day].get('exercises', [])
                    for exercise in suggested_exercises:
                        new_exercise = exercise.copy()
                        new_exercise['sets'] = 3
                        new_exercise['reps'] = 12 if exercise.get('isBodyWeight') else 10
                        current_exercises.append(new_exercise)
                    template['days'][target_day]['exercises'] = current_exercises

                    exercise_names = [ex['name'] for ex in suggested_exercises]
                    messages.append(f"Added {muscle} exercises: {', '.join(exercise_names)}")

    return {'messages': messages}


def _handle_exercise_replacement(template: Dict[str, Any], user_instruction: str, validation_result: Dict, db: Session, oai, model: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle exercise replacement requests"""
    from .ai_exercise_validator import AIExerciseValidator
    messages = []

    print(f"🔍 DEBUG - Exercise replacement: '{user_instruction}'")

    # Extract what to replace and what to replace with
    import re
    replacement_patterns = [
        r'replace\s+([^w]+)\s+with\s+([^\.]+)',
        r'change\s+([^t]+)\s+to\s+([^\.]+)',
        r'swap\s+([^f]+)\s+for\s+([^\.]+)',
        r'alternate\s+([^w]+)\s+with\s+([^\.]+)',
        r'substitute\s+([^w]+)\s+with\s+([^\.]+)',
    ]

    target_exercise = None
    replacement_exercise = None

    for pattern in replacement_patterns:
        match = re.search(pattern, user_instruction.lower())
        if match:
            target_exercise = match.group(1).strip()
            replacement_exercise = match.group(2).strip()
            break

    print(f"🔍 DEBUG - Target: '{target_exercise}', Replacement: '{replacement_exercise}'")

    if target_exercise and replacement_exercise:
        # Validate replacement exercise
        from .database_exercise_manager import DatabaseExerciseManager
        exists, exercise_data = DatabaseExerciseManager.validate_exercise_exists(db, replacement_exercise)

        if exists:
            # Replace in all days where the target exercise is found
            replaced_count = 0
            for day_key, day_data in template['days'].items():
                exercises = day_data.get('exercises', [])
                for i, exercise in enumerate(exercises):
                    if target_exercise.lower() in exercise.get('name', '').lower():
                        # Replace the exercise
                        new_exercise = exercise_data.copy()
                        new_exercise['sets'] = exercise.get('sets', 3)
                        new_exercise['reps'] = exercise.get('reps', 10)
                        exercises[i] = new_exercise
                        replaced_count += 1

            if replaced_count > 0:
                messages.append(f"Replaced {target_exercise} with {exercise_data['name']} in {replaced_count} location(s)")
            else:
                messages.append(f"Could not find '{target_exercise}' to replace")
        else:
            # Find similar exercises
            similar = DatabaseExerciseManager.find_similar_exercises(db, replacement_exercise, limit=3)
            if similar:
                suggestions = [ex['name'] for ex in similar]
                messages.append(f"'{replacement_exercise}' not found. Try: {', '.join(suggestions)}")
            else:
                messages.append(f"'{replacement_exercise}' not found in database")

    return {'messages': messages}


def _handle_exercise_removal(template: Dict[str, Any], user_instruction: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle exercise removal requests"""
    messages = []
    instruction_lower = user_instruction.lower()

    print(f"🔍 DEBUG - Exercise removal: '{user_instruction}'")

    # Extract exercise to remove
    import re
    remove_patterns = [
        r'remove\s+([^f]+?)(?:\s+from|\s*$)',
        r'delete\s+([^f]+?)(?:\s+from|\s*$)',
        r'take\s+out\s+([^f]+?)(?:\s+from|\s*$)',
    ]

    target_exercise = None
    for pattern in remove_patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target_exercise = match.group(1).strip()
            break

    # Check if removing from specific day
    target_day = None
    for day_key in template['days'].keys():
        if day_key.lower() in instruction_lower or template['days'][day_key].get('title', '').lower() in instruction_lower:
            target_day = day_key
            break

    print(f"🔍 DEBUG - Target exercise: '{target_exercise}', Target day: '{target_day}'")

    if target_exercise:
        removed_count = 0
        days_to_process = [target_day] if target_day else list(template['days'].keys())

        for day_key in days_to_process:
            day_data = template['days'][day_key]
            exercises = day_data.get('exercises', [])
            original_count = len(exercises)

            # Remove exercises that match
            exercises = [ex for ex in exercises if target_exercise.lower() not in ex.get('name', '').lower()]
            new_count = len(exercises)

            if new_count < original_count:
                template['days'][day_key]['exercises'] = exercises
                removed_count += (original_count - new_count)
                print(f"🔍 DEBUG - Removed {original_count - new_count} exercise(s) from {day_key}")

        if removed_count > 0:
            if target_day:
                day_title = template['days'][target_day].get('title', target_day)
                messages.append(f"Removed {target_exercise} from {day_title}")
            else:
                messages.append(f"Removed {target_exercise} from {removed_count} location(s)")
        else:
            messages.append(f"Could not find '{target_exercise}' to remove")
    else:
        messages.append("Could not understand which exercise to remove")

    return {'messages': messages}


def _handle_difficulty_modification(template: Dict[str, Any], user_instruction: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle difficulty modifications like making workouts harder/easier"""
    messages = []
    instruction_lower = user_instruction.lower()

    if 'harder' in instruction_lower or 'more reps' in instruction_lower:
        # Increase reps/sets
        for day_key, day_data in template['days'].items():
            for exercise in day_data.get('exercises', []):
                if isinstance(exercise.get('reps'), int):
                    exercise['reps'] = min(exercise['reps'] + 2, 20)  # Cap at 20
                if isinstance(exercise.get('sets'), int):
                    exercise['sets'] = min(exercise['sets'] + 1, 5)   # Cap at 5
        messages.append("Made all exercises harder (increased reps and sets)")

    elif 'easier' in instruction_lower or 'less reps' in instruction_lower:
        # Decrease reps/sets
        for day_key, day_data in template['days'].items():
            for exercise in day_data.get('exercises', []):
                if isinstance(exercise.get('reps'), int):
                    exercise['reps'] = max(exercise['reps'] - 2, 5)   # Minimum 5
                if isinstance(exercise.get('sets'), int):
                    exercise['sets'] = max(exercise['sets'] - 1, 2)  # Minimum 2
        messages.append("Made all exercises easier (decreased reps and sets)")

    return {'messages': messages}


def _validate_final_template_exercises(template: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Final validation to ensure all exercises are from database"""
    from .database_exercise_manager import DatabaseExerciseManager

    for day_key, day_data in template['days'].items():
        if isinstance(day_data, dict) and 'exercises' in day_data:
            valid_exercises = []
            for exercise in day_data['exercises']:
                if exercise.get('id'):  # Already has database ID
                    valid_exercises.append(exercise)
                else:
                    # Validate exercise is in database
                    exists, validated_exercise = DatabaseExerciseManager.validate_exercise_exists(
                        db, exercise.get('name', '')
                    )
                    if exists:
                        validated_exercise['sets'] = exercise.get('sets', 3)
                        validated_exercise['reps'] = exercise.get('reps', 10)
                        valid_exercises.append(validated_exercise)
                    else:
                        print(f"⚠️ Filtering out invalid exercise: {exercise.get('name', 'Unknown')}")

            day_data['exercises'] = valid_exercises

    return template


def _ensure_template_structure_compatibility(template: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure template structure is compatible with dynamic day key system"""
    if not template or not isinstance(template, dict):
        return template

    days = template.get('days', {})
    if not isinstance(days, dict):
        return template

    # Ensure each day has the required structure
    for day_key, day_data in days.items():
        if not isinstance(day_data, dict):
            continue

        # Ensure exercises is a list
        if 'exercises' not in day_data:
            day_data['exercises'] = []
        elif not isinstance(day_data['exercises'], list):
            day_data['exercises'] = []

        # Ensure each exercise has proper structure
        valid_exercises = []
        for exercise in day_data['exercises']:
            if isinstance(exercise, dict) and exercise.get('id') and exercise.get('name'):
                # Ensure basic exercise properties
                exercise_copy = exercise.copy()
                exercise_copy.setdefault('sets', 3)
                exercise_copy.setdefault('reps', 10)
                valid_exercises.append(exercise_copy)

        day_data['exercises'] = valid_exercises

        # Ensure day has a title
        if 'title' not in day_data or not day_data['title']:
            day_data['title'] = day_key.title()

    return template


def llm_generate_template_from_profile(oai, model: str, profile: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    print("🔍 DEBUG - Entered llm_generate_template_from_profile function")

    goal = (profile.get("client_goal") or profile.get("goal") or "muscle gain")
    experience = (profile.get("experience") or "beginner")
    cw = profile.get("current_weight")
    tw = profile.get("target_weight")
    delta_txt = profile.get("weight_delta_text") or ""

    print(f"🔍 DEBUG - Basic profile parsed: goal={goal}, experience={experience}")

    # Get template configuration
    template_count = profile.get("template_count", 6)
    template_names = profile.get("template_names", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"])

    print(f"🔍 DEBUG - Template config: count={template_count}, names={template_names}")
    
    # Check if this is a muscle-specific template request
    muscle_focus = profile.get("muscle_focus")

    print(f"🔍 DEBUG - Checking muscle focus: {muscle_focus}")

    if muscle_focus:
        print(f"🎯 Creating muscle-specific template for: {muscle_focus}")
        muscle_distributions = {muscle_focus: len(template_names)}
        return SmartWorkoutEditor.create_muscle_specific_template(template_names, muscle_distributions, db)
    
    # CRITICAL FIX: For single muscle requests like "legs", use direct database approach
    if len(template_names) == 1 and template_names[0].lower() in ['legs', 'leg', 'chest', 'back', 'arms', 'shoulders']:
        muscle_name = template_names[0].lower()
        if muscle_name in ['leg', 'legs']:
            muscle_name = 'legs'
        
        print(f"🎯 Creating single-day {muscle_name} workout")
        muscle_distributions = {muscle_name: 1}
        return SmartWorkoutEditor.create_muscle_specific_template(['monday'], muscle_distributions, db)
    
    # Continue with existing LLM generation for general templates
    print("🔍 DEBUG - Preparing LLM generation for general templates")

    user_prompt = (
        "Build a workout template for this client profile:\n"
        f"- Goal: {goal}\n"
        f"- Number of templates: {template_count}\n"
        f"- Template names: {', '.join(template_names)}\n"
        f"- Experience: {experience}\n"
        f"- Current Weight: {cw}\n"
        f"- Target Weight:  {tw}\n"
        f"- Weight Goal: {delta_txt}\n"
        f"Use these exact day keys: {', '.join([name.lower() for name in template_names])}\n"
        "Pick a split that works well with the given template names and training frequency."
    )

    print("🔍 DEBUG - About to call OpenAI API")

    try:
        system_prompt = generate_system_prompt(template_names)
        print("🔍 DEBUG - Generated system prompt successfully")

        resp = oai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=0.2,
        )
        print("🔍 DEBUG - OpenAI API call completed successfully")
        obj = _safe_json(resp.choices[0].message.content or "{}", {
            "template": _template_skeleton_dynamic(template_names),
            "rationale": ""
        })
        tpl = obj.get("template") or _template_skeleton_dynamic(template_names)
        rat = obj.get("rationale") or ""
        
        if isinstance(tpl.get("days"), dict):
            # Keep only the specified template names
            valid_days = [name.lower() for name in template_names]
            tpl["days"] = {k: v for k, v in tpl["days"].items() if k in valid_days}
            for day_name in template_names:
                day_key = day_name.lower()
                tpl["days"].setdefault(day_key, {"title": day_name.title(), "muscle_groups": [], "exercises": []})
        
        tpl.setdefault("name", f"Workout Template ({template_count} days)")

        print("🔍 DEBUG - About to enforce catalog on template")
        # Enforce DB catalog + attach IDs
        tpl = _enforce_catalog_on_template_db_dynamic(tpl, db, template_names)
        print("🔍 DEBUG - Catalog enforcement completed")

        return tpl, rat
    except Exception as e:
        print(f"🚨 Template generation error: {e}")
        import traceback
        traceback.print_exc()
        return _template_skeleton_dynamic(template_names), f"Fallback skeleton due to generation error: {e}"
# ───────────────────── LLM: edit template ───────────────────
EDIT_SYSTEM = (
    "You are modifying an existing workout template. "
    "Return ONLY strict JSON with schema: {\"template\": <updated>, \"summary\": string}. "
    "Keep unspecified days/exercises unchanged. Respect the instruction precisely. "
    "Use common exercise names; the system will map to a fixed catalog. "
    "No markdown; ONLY JSON. Always keep Monday–Saturday day keys."
)
def llm_edit_template(oai, model: str, template: Dict[str,Any], instruction: str, profile_hint: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    # Extract original day structure to preserve it
    original_days = list(template.get("days", {}).keys())
    instruction_lower = instruction.lower()

    # Check if user wants to change the number of days
    day_reduction_keywords = ['reduce', 'fewer', 'less', 'cut down', 'decrease', 'minimize']
    day_expansion_keywords = ['add', 'more', 'increase', 'expand', 'extra', 'additional']
    user_wants_day_reduction = any(keyword in instruction_lower for keyword in day_reduction_keywords) and 'day' in instruction_lower
    user_wants_day_expansion = any(keyword in instruction_lower for keyword in day_expansion_keywords) and 'day' in instruction_lower

    # ENHANCED: Also check for number-based day changes (e.g., "make it to 4 days", "change to 3 days")
    if ('day' in instruction_lower and not user_wants_day_reduction and not user_wants_day_expansion):
        import re
        # Look for patterns like "to X days", "X days", "make it X days", "for X days"
        number_patterns = [
            r'(?:to|make.*?to|change.*?to|for)\s*(\d+)\s*days?',
            r'(\d+)\s*days?(?:\s+(?:only|total|workout))?',
            r'template.*?for.*?(\d+)\s*days?',
            r'make.*?template.*?(\d+)\s*days?',
        ]

        for pattern in number_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                target_days = int(match.group(1))
                current_days = len(original_days)
                if target_days < current_days:
                    user_wants_day_reduction = True
                    print(f"🎯 Detected day reduction request: {current_days} → {target_days} days")
                elif target_days > current_days:
                    user_wants_day_expansion = True
                    print(f"🎯 Detected day expansion request: {current_days} → {target_days} days")
                break
    template_names = [day.title() for day in original_days]
    if "change all" in instruction.lower() and "exercise" in instruction.lower():
        special_instruction = (
            f"REPLACE ALL EXERCISES with completely different ones for the same muscle groups.\n"
            f"Current exercises to AVOID: {[ex.get('name') for day in template.get('days', {}).values() for ex in day.get('exercises', [])]}\n"
            f"Generate 5 completely different {template.get('days', {}).get(list(template.get('days', {}).keys())[0], {}).get('muscle_groups', ['leg'])[0].lower()} exercises.\n"
            f"Use exercise names like: Goblet Squats, Romanian Deadlifts, Bulgarian Split Squats, Step-ups, Hip Thrusts, Leg Press, Wall Sits, Single Leg Deadlifts, etc.\n"
            f"Make sure they are completely different from current exercises."
        )
        
        msgs = [
            {"role":"system","content":EDIT_SYSTEM},
            {"role":"user","content":(
                f"PRESERVE THESE EXACT DAY KEYS: {original_days}\n"
                "Current template JSON:\n"
                + orjson.dumps(template).decode()
                + f"\n\nOriginal day structure: {original_days}\n"
                + "\n\nSpecial Instruction:\n"
                + special_instruction
                + f"\n\nIMPORTANT: Keep day keys exactly as: {original_days}"
            )},
        ]
    else:
        # Regular instruction handling
        if user_wants_day_reduction:
            day_preservation_msg = f"You may reduce the number of days if requested. Original days: {original_days}"
        elif user_wants_day_expansion:
            day_preservation_msg = f"You may expand to more days if requested. Original days: {original_days}. Create new day keys like day5, day6, etc."
        else:
            day_preservation_msg = f"PRESERVE THESE EXACT DAY KEYS: {original_days}"

        msgs = [
            {"role":"system","content":EDIT_SYSTEM},
            {"role":"user","content":(
                f"{day_preservation_msg}\n"
                "Current template JSON:\n"
                + orjson.dumps(template).decode()
                + "\n\nClient hints (goal/experience/weights):\n"
                + orjson.dumps(profile_hint).decode()
                + f"\n\nOriginal day structure: {original_days}\n"
                + "\n\nInstruction:\n"
                + (instruction or "").strip()
                + (f"\n\nIMPORTANT: Keep day keys exactly as: {original_days}" if not (user_wants_day_reduction or user_wants_day_expansion)
                   else ("\n\nNote: You may reduce days if requested in the instruction." if user_wants_day_reduction
                         else "\n\nNote: You may expand to more days if requested in the instruction. Use day5, day6, etc."))
            )},
        ]
    try:
        # Handle test mode when oai is None
        if oai is None:
            print(f"🧪 Test mode: No LLM available, using manual fallback")
            raise Exception("Test mode: LLM not available")

        resp = oai.chat.completions.create(
            model=model,
            messages=msgs,
            response_format={"type":"json_object"},
            temperature=0,
        )
        obj = _safe_json(resp.choices[0].message.content or "{}", {"template": template, "summary":"No change"})
        updated = obj.get("template") or template
        # CRITICAL: Validate that day structure wasn't corrupted
        if isinstance(updated.get("days"), dict):
            updated_days = list(updated["days"].keys())
            
            # If LLM changed the day structure, check if it's a valid day change
            if set(updated_days) != set(original_days):
                if (user_wants_day_reduction and len(updated_days) < len(original_days)) or \
                   (user_wants_day_expansion and len(updated_days) > len(original_days)):
                    change_type = "reduction" if user_wants_day_reduction else "expansion"
                    print(f"✅ User requested day {change_type}: {len(original_days)} → {len(updated_days)} days")
                    # Valid day change, keep the changes
                    summary = obj.get("summary") or f"Successfully {change_type} template to {len(updated_days)} days"
                else:
                    print(f"❌ LLM corrupted day structure. Original: {original_days}, LLM returned: {updated_days}")
                    updated = template.copy()  # Revert to original
                    summary = "Could not apply change - LLM altered template structure. Template preserved."
            else:
                # Structure preserved - restore any missing days and remove extra days
                # But SKIP this entirely if this is a day change request
                if not ((user_wants_day_reduction and len(updated_days) < len(original_days)) or
                       (user_wants_day_expansion and len(updated_days) > len(original_days))):
                    # Ensure all original days exist
                    for day_key in original_days:
                        if day_key not in updated["days"]:
                            updated["days"][day_key] = template["days"].get(day_key, {
                                "title": day_key.title(),
                                "muscle_groups": [],
                                "exercises": []
                            })

                    # Remove any extra days the LLM might have added
                    updated["days"] = {k: v for k, v in updated["days"].items() if k in original_days}
                else:
                    print(f"🚫 Skipping day restoration - user requested day change from {len(original_days)} to {len(updated_days)} days")

                # Set summary for this path
                summary = obj.get("summary") or "Updated template successfully"

                # Update template name to reflect new day count
                current_day_count = len(updated.get("days", {}))
                updated.setdefault("name", template.get("name") or f"Workout Template ({current_day_count} days)")
                # Enforce DB catalog after edit using dynamic function
                current_days = list(updated.get("days", {}).keys())
                current_day_names = [day.title() for day in current_days]

                if len(current_days) <= 6 and all(day in DAYS6 for day in current_days):
                    # Use original function for standard days
                    updated = _enforce_catalog_on_template_db(updated, db)
                else:
                    # Use dynamic function for custom days
                    updated = _enforce_catalog_on_template_db_dynamic(updated, db, current_day_names)
        else:
            updated = template
            summary = "No valid template structure returned by LLM."
        return updated, summary
    
    except Exception as e:
        print(f"LLM edit error: {e}")
        # Preserve original template structure
        preserved = template.copy()
        if len(original_days) <= 6 and all(day in DAYS6 for day in original_days):
            preserved = _enforce_catalog_on_template_db(preserved, db)
        else:
            preserved = _enforce_catalog_on_template_db_dynamic(preserved, db, template_names)
        return preserved, f"I had trouble processing that request ({str(e)[:50]}). Your template has been preserved. Try rephrasing your request or being more specific."
def find_exercise_in_template(template: Dict[str, Any], exercise_name_fragment: str) -> Tuple[str, int, str]:
    """Find exercise in template by name fragment. Returns (day_key, exercise_index, exercise_name)"""
    exercise_name_fragment = exercise_name_fragment.lower().replace(" ", "")
    for day_key, day_data in template.get("days", {}).items():
        exercises = day_data.get("exercises", [])
        for i, exercise in enumerate(exercises):
            exercise_name = exercise.get("name", "").lower().replace(" ", "")
            if exercise_name_fragment in exercise_name or exercise_name in exercise_name_fragment:
                return day_key, i, exercise.get("name", "")
    return None, -1, ""
def calculate_similarity(str1: str, str2: str) -> float:
    """Enhanced similarity calculation with better spelling mistake handling"""
    str1_clean = str1.lower().strip()
    str2_clean = str2.lower().strip()

    # Exact match
    if str1_clean == str2_clean:
        return 1.0

    # Remove spaces for comparison
    str1_no_space = str1_clean.replace(" ", "")
    str2_no_space = str2_clean.replace(" ", "")

    if str1_no_space == str2_no_space:
        return 0.95

    # Check for substring matches (high priority)
    if str1_no_space in str2_no_space or str2_no_space in str1_no_space:
        return 0.85

    # Token-based matching for multi-word exercises
    tokens1 = str1_clean.split()
    tokens2 = str2_clean.split()

    if len(tokens1) > 1 and len(tokens2) > 1:
        token_matches = 0
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 == token2 or token1 in token2 or token2 in token1:
                    token_matches += 1
                    break

        token_similarity = token_matches / max(len(tokens1), len(tokens2))
        if token_similarity > 0.6:
            return 0.7 + (token_similarity * 0.2)

    # Levenshtein distance for spelling mistakes
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # Calculate normalized edit distance
    edit_distance = levenshtein_distance(str1_no_space, str2_no_space)
    max_len = max(len(str1_no_space), len(str2_no_space))

    if max_len == 0:
        return 0.0

    edit_similarity = 1 - (edit_distance / max_len)

    # Common exercise misspellings correction
    exercise_corrections = {
        'dumbell': 'dumbbell',
        'dumbel': 'dumbbell',
        'dumbbel': 'dumbbell',
        'benchpress': 'bench press',
        'benchpres': 'bench press',
        'pushup': 'push up',
        'pullup': 'pull up',
        'situp': 'sit up',
        'chinup': 'chin up',
        'bicep': 'biceps',
        'tricep': 'triceps',
        'shoulderpress': 'shoulder press',
        'chestpress': 'chest press',
        'legpress': 'leg press',
        'deadlift': 'deadlift',
        'squat': 'squat',
        'lunge': 'lunge'
    }

    # Apply corrections and retry
    str1_corrected = str1_no_space
    str2_corrected = str2_no_space

    for wrong, correct in exercise_corrections.items():
        str1_corrected = str1_corrected.replace(wrong, correct.replace(" ", ""))
        str2_corrected = str2_corrected.replace(wrong, correct.replace(" ", ""))

    if str1_corrected == str2_corrected:
        return 0.9

    if str1_corrected in str2_corrected or str2_corrected in str1_corrected:
        return 0.8

    # Phonetic similarity for sound-alike words
    def soundex(s):
        """Simple soundex implementation for phonetic matching"""
        s = s.upper()
        soundex_code = s[0] if s else ''

        # Mapping consonants to codes
        mapping = {
            'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
            'L': '4', 'MN': '5', 'R': '6'
        }

        for char in s[1:]:
            for group, code in mapping.items():
                if char in group:
                    if len(soundex_code) == 1 or soundex_code[-1] != code:
                        soundex_code += code
                    break

        return (soundex_code + '000')[:4]

    if len(str1_no_space) > 3 and len(str2_no_space) > 3:
        if soundex(str1_no_space) == soundex(str2_no_space):
            return max(0.6, edit_similarity)

    # Return the best similarity score, but with minimum threshold
    return max(edit_similarity, 0.0)
def handle_specific_exercise_addition(template: Dict[str,Any], instruction: str, db: Session) -> Tuple[Dict[str,Any], str]:
    """FIXED: Handle specific exercise addition requests with fuzzy matching and typo tolerance"""
    try:
        cat = load_catalog(db)
        if not cat or "by_id" not in cat:
            return template, "Could not load exercise database"
    except Exception as e:
        print(f"❌ Database error in add exercise: {e}")
        return template, f"Database error: {str(e)}"
    
    instruction_lower = instruction.lower()
    updated = template.copy()
    
    print(f"🔍 Processing add exercise request: '{instruction}'")
    
    # Initialize variables
    exercise_name = None
    target_day_key = None
    exercise_id = None
    
    # Enhanced exercise name extraction patterns
    import re
    
    # Pattern 1: "add [exercise] on [day]" or "add [exercise] [day]"
    add_patterns = [
        r'add\s+([^in]+?)\s+in\s+(all\s+days?|every\s+days?)', # "add exercise in all days"
        r'add\s+([^on]+?)\s+on\s+(\w+)',          # "add exercise on day"
        r'add\s+([^to]+?)\s+to\s+(\w+)',          # "add exercise to day"
        r'add\s+(\w+(?:\s+\w+)?)\s+(\w+day|\w+)', # "add exercise monday"
    ]
    
    for pattern in add_patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            potential_exercise = match.group(1).strip()
            potential_day = match.group(2).strip()
            
            print(f"🎯 Pattern matched - Exercise: '{potential_exercise}', Day: '{potential_day}'")
            
            # Try to find the exercise in database with fuzzy matching
            exercise_id = id_for_name(potential_exercise, cat)
            
            if not exercise_id:
                # Try fuzzy matching with all exercises in database
                best_match = None
                best_score = 0
                
                print(f"🔍 Trying fuzzy match for: '{potential_exercise}'")
                
                for eid, exercise_data in cat["by_id"].items():
                    db_name = exercise_data["name"].lower()
                    
                    # Calculate similarity score
                    score = calculate_similarity(potential_exercise, db_name)
                    
                    if score > best_score and score > 0.5:  # Lower threshold for better matching
                        best_score = score
                        best_match = (eid, exercise_data["name"])
                
                if best_match:
                    exercise_id = best_match[0]
                    exercise_name = best_match[1]
                    print(f"✅ Found fuzzy match: '{potential_exercise}' -> '{exercise_name}' (score: {best_score:.2f})")
            
            if exercise_id:
                exercise_name = cat["by_id"][exercise_id]["name"]
                
                # Find target day
                print(f"🔍 Looking for day: '{potential_day}'")

                # Handle "all days" or "every days" case
                if potential_day.lower() in ['all days', 'every days', 'all day', 'every day']:
                    # Add to all available days
                    for day_key in updated.get("days", {}).keys():
                        day_data = updated["days"][day_key]
                        current_exercises = day_data.get("exercises", [])

                        # Check if exercise already exists in this day
                        exercise_exists = any(ex.get("id") == exercise_id for ex in current_exercises)

                        if not exercise_exists and len(current_exercises) < 8:
                            new_exercise = {
                                "id": exercise_id,
                                "name": exercise_name,
                                "sets": 3,
                                "reps": 10,
                                "note": None
                            }
                            current_exercises.append(new_exercise)
                            print(f"✅ Added '{exercise_name}' to {day_key}")

                    return updated, f"Added '{exercise_name}' to all days"

                # Handle specific day
                for day_key in updated.get("days", {}).keys():
                    if (potential_day.lower() in day_key.lower() or
                        day_key.lower() in potential_day.lower() or
                        potential_day.lower() == day_key.lower()):
                        target_day_key = day_key
                        print(f"✅ Found matching day: '{day_key}'")
                        break
            
            if exercise_id and target_day_key:
                break  # Found both, exit pattern loop
    
    # If no pattern matched, try alternative patterns for common exercises
    if not exercise_name:
        print("🔍 Trying alternative exercise patterns...")
        exercise_patterns = [
            (r'barbell\s*curl', 'Barbell Curl'),
            (r'box\s*jump', 'Box Jumps'),
            (r'push\s*up', 'Plyometric Pushups'),
            (r'squat', 'Dumbell Squats'),
            (r'burpee', 'Burpees'),
            (r'plank', 'Plank Jacks'),
            (r'mountain\s*climb', 'Mountain Climbers'),
            (r'russian\s*twist', 'Russian Twists'),
            (r'high\s*knee', 'High Knees'),
        ]
        
        for pattern, exercise_suggestion in exercise_patterns:
            if re.search(pattern, instruction_lower):
                # Find this exercise in database
                exercise_id = id_for_name(exercise_suggestion, cat)
                if exercise_id:
                    exercise_name = cat["by_id"][exercise_id]["name"]
                    print(f"✅ Matched pattern '{pattern}' to exercise '{exercise_name}'")
                    break
        
        # Still try to find day if we found an exercise
        if exercise_name and not target_day_key:
            day_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for day in day_keywords:
                if day in instruction_lower:
                    for day_key in updated.get("days", {}).keys():
                        if day in day_key.lower() or day_key.lower() in day:
                            target_day_key = day_key
                            print(f"✅ Found day from keywords: '{day_key}'")
                            break
                    if target_day_key:
                        break
    
    # Validation and error handling
    if not target_day_key:
        available_days = list(updated.get("days", {}).keys())
        return template, f"Could not identify which day to add the exercise to. Available days: {', '.join(available_days)}. Please specify clearly (e.g., 'add barbell curl on monday')."
    
    if not exercise_name or not exercise_id:
        return template, f"Could not find exercise '{instruction}' in database. Please check the exercise name. Try: Barbell Curl, Box Jumps, Burpees, Mountain Climbers, etc."
    
    # Check if target day exists
    if target_day_key not in updated.get("days", {}):
        return template, f"Day '{target_day_key}' not found in template. Available days: {', '.join(updated.get('days', {}).keys())}"
    
    # Check day exercise limit
    day_data = updated["days"][target_day_key]
    current_exercises = day_data.get("exercises", [])
    
    if len(current_exercises) >= 8:
        return template, f"Day '{target_day_key}' already has {len(current_exercises)} exercises (maximum is 8). Please remove an exercise first or choose a different day."
    
    # Check if exercise already exists in this day
    for ex in current_exercises:
        if ex.get("id") == exercise_id:
            return template, f"Exercise '{exercise_name}' is already in {target_day_key}."
    
    # Add the exercise
    new_exercise = {
        "id": exercise_id,
        "name": exercise_name,
        "sets": 3,
        "reps": 10,
        "note": None
    }
    
    current_exercises.append(new_exercise)
    day_data["exercises"] = current_exercises
    
    print(f"✅ Successfully added '{exercise_name}' (ID: {exercise_id}) to {target_day_key}")
    return updated, f"Added '{exercise_name}' to {target_day_key.title()} ({len(current_exercises)} exercises total)."
def apply_manual_edit(template: Dict[str,Any], instruction: str, db: Session) -> Tuple[Dict[str,Any], str]:
    """UNIVERSAL: Handle alternatives for ANY exercise in database"""
    import re  # Ensure re is available in this function scope
    instruction_lower = instruction.lower()
    updated = template.copy()
    
    # Handle test mode when db is None
    if db is None:
        # Test mode: Handle day operations manually
        if 'day' in instruction_lower and any(word in instruction_lower for word in ['make', 'change', 'reduce', 'cut', 'expand', 'increase']):
            import re
            # Look for number patterns
            number_patterns = [
                r'(?:to|for|make.*?to|change.*?to)\s*(\d+)\s*days?',
                r'(\d+)\s*days?(?:\s+(?:only|total|workout))?',
            ]

            for pattern in number_patterns:
                match = re.search(pattern, instruction_lower)
                if match:
                    target_days = int(match.group(1))
                    current_days = len(updated.get('days', {}))

                    if target_days < current_days:
                        # Day reduction
                        day_keys = list(updated['days'].keys())
                        # Keep only the first N days
                        for i, day_key in enumerate(day_keys):
                            if i >= target_days:
                                del updated['days'][day_key]

                        return updated, f"🧪 Test mode: Reduced from {current_days} to {target_days} days"

                    elif target_days > current_days:
                        # Day expansion - create new days
                        for i in range(current_days + 1, target_days + 1):
                            new_day_key = f"day{i}"
                            updated['days'][new_day_key] = {
                                "title": f"Day {i}",
                                "muscle_groups": ["full body"],
                                "exercises": [
                                    {"id": f"test_{i}_1", "name": f"Exercise {i}-1", "sets": 3, "reps": "10-12"},
                                    {"id": f"test_{i}_2", "name": f"Exercise {i}-2", "sets": 3, "reps": "10-12"},
                                    {"id": f"test_{i}_3", "name": f"Exercise {i}-3", "sets": 3, "reps": "10-12"},
                                    {"id": f"test_{i}_4", "name": f"Exercise {i}-4", "sets": 3, "reps": "10-12"},
                                    {"id": f"test_{i}_5", "name": f"Exercise {i}-5", "sets": 3, "reps": "10-12"},
                                    {"id": f"test_{i}_6", "name": f"Exercise {i}-6", "sets": 3, "reps": "10-12"}
                                ]
                            }

                        return updated, f"🧪 Test mode: Expanded from {current_days} to {target_days} days"

                    break

        return template, "🧪 Test mode: Database operations skipped"

    # Load exercise catalog from database
    try:
        cat = load_catalog(db)
        if not cat or "by_id" not in cat:
            return template, "Could not load exercise database"
    except Exception as e:
        return template, f"Database error: {str(e)}"
    
    day_muscle_patterns = [
        r'(?:give|add|put|make)\s+(?:only\s+)?(\w+)\s+exercise[s]?\s+(?:on\s+)?(\w+day|\w+)',
        r'(?:change|replace)\s+(\w+day|\w+)\s+(?:to\s+)?(?:only\s+)?(\w+)\s+exercise[s]?',
        r'(\w+day|\w+)\s+(?:should\s+)?(?:have\s+)?(?:only\s+)?(\w+)\s+exercise[s]?'
    ]
    
    for pattern in day_muscle_patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            # Extract muscle and day (order might vary based on pattern)
            group1, group2 = match.group(1), match.group(2)
            
            # Determine which is muscle and which is day
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            muscle = None
            target_day = None
            
            if any(day in group1 for day in days):
                target_day = group1
                muscle = group2
            elif any(day in group2 for day in days):
                target_day = group2
                muscle = group1
            else:
                # Try partial matching
                for day in days:
                    if day[:3] in group1 or group1 in day:
                        target_day = day
                        muscle = group2
                        break
                    elif day[:3] in group2 or group2 in day:
                        target_day = day
                        muscle = group1
                        break
            
            if target_day and muscle:
                print(f"🎯 Day-specific muscle request: {muscle} exercises for {target_day}")
                
                # Find the matching day in template
                matching_day_key = None
                for day_key in updated.get("days", {}).keys():
                    if target_day in day_key.lower() or day_key.lower() in target_day:
                        matching_day_key = day_key
                        break
                
                if matching_day_key:
                    # Get muscle-specific exercises
                    muscle_exercise_names = []
                    if muscle in ['chest']:
                        muscle_exercise_names = [
                            'bench press', 'chest press', 'push up', 'chest fly', 'chest flye',
                            'incline press', 'decline press', 'dips', 'pec fly'
                        ]
                    elif muscle in ['back']:
                        muscle_exercise_names = [
                            'pull up', 'lat pulldown', 'row', 'deadlift', 'shrug'
                        ]
                    elif muscle in ['leg', 'legs']:
                        muscle_exercise_names = [
                            'squat', 'lunge', 'leg press', 'leg extension', 'leg curl', 
                            'calf raise', 'step up'
                        ]
                    # Add more muscle groups as needed
                    
                    # Find matching exercises in database
                    new_exercises = []
                    used_ids = set()
                    
                    for eid, exercise_data in cat["by_id"].items():
                        if len(new_exercises) >= 6:  # Target 6 exercises
                            break
                        
                        exercise_name = exercise_data["name"].lower()
                        
                        # Check if this exercise matches the requested muscle
                        is_target_muscle = any(muscle_name in exercise_name for muscle_name in muscle_exercise_names)
                        
                        if is_target_muscle and eid not in used_ids:
                            new_exercises.append({
                                "id": eid,
                                "name": exercise_data["name"],
                                "sets": 3,
                                "reps": 10,
                                "note": None
                            })
                            used_ids.add(eid)
                    
                    if new_exercises:
                        # ENFORCE 6-EXERCISE MINIMUM
                        while len(new_exercises) < 6:
                            # Add more exercises from the same muscle group
                            for eid, exercise_data in cat["by_id"].items():
                                if len(new_exercises) >= 6:
                                    break
                                if eid not in used_ids:
                                    exercise_name = exercise_data["name"].lower()
                                    is_target_muscle = any(muscle_name in exercise_name for muscle_name in muscle_exercise_names)
                                    if is_target_muscle:
                                        new_exercises.append({
                                            "id": eid,
                                            "name": exercise_data["name"],
                                            "sets": 3,
                                            "reps": 10,
                                            "note": None
                                        })
                                        used_ids.add(eid)
                        
                        # ENFORCE 8-EXERCISE MAXIMUM
                        if len(new_exercises) > 8:
                            new_exercises = new_exercises[:8]
                        
                        # Update the specific day
                        day_data = updated["days"][matching_day_key].copy()
                        day_data["exercises"] = new_exercises
                        day_data["muscle_groups"] = [muscle.title()]
                        day_data["title"] = f"{muscle.title()} Day"
                        updated["days"][matching_day_key] = day_data
                        
                        exercise_names = [ex["name"] for ex in new_exercises]
                        return updated, f"Changed {matching_day_key} to only {muscle} exercises: {', '.join(exercise_names[:3])}{'...' if len(exercise_names) > 3 else ''}"
                    else:
                        return template, f"No {muscle} exercises found in database"
                else:
                    available_days = list(updated.get("days", {}).keys())
                    return template, f"Day '{target_day}' not found. Available days: {', '.join(available_days)}"
    
    # UNIVERSAL ALTERNATIVE AND REPLACEMENT HANDLER
    is_alternative_request = ("alternative" in instruction_lower or "alternate" in instruction_lower or
                             "different exercise" in instruction_lower or "something else" in instruction_lower)
    is_replacement_request = any(word in instruction_lower for word in ["replace", "change", "swap", "substitute"])

    if is_alternative_request or is_replacement_request:
        print(f"🔍 Processing exercise modification request: {instruction}")

        # STEP 1: Extract exercise name and replacement (if any) from instruction
        target_exercise_name = None
        replacement_exercise_name = None

        # Method 1: Handle replacement patterns first
        if is_replacement_request:
            replacement_patterns = [
                r'replace\s+(.+?)\s+with\s+(.+)',          # "replace X with Y"
                r'change\s+(.+?)\s+to\s+(.+)',             # "change X to Y"
                r'swap\s+(.+?)\s+for\s+(.+)',              # "swap X for Y"
                r'substitute\s+(.+?)\s+with\s+(.+)',       # "substitute X with Y"
            ]

            for pattern in replacement_patterns:
                match = re.search(pattern, instruction_lower)
                if match:
                    target_exercise_name = match.group(1).strip()
                    replacement_exercise_name = match.group(2).strip()
                    print(f"🎯 Replacement: '{target_exercise_name}' -> '{replacement_exercise_name}'")
                    break

        # Method 2: Handle alternative patterns
        if not target_exercise_name and is_alternative_request:
            for_patterns = [
                r'(?:alternate|alternative)\s+for\s+(.+?)$',
                r'(?:alternate|alternative)\s+(.+?)$',
                r'different\s+exercise\s+for\s+(.+?)$',
                r'something\s+else\s+for\s+(.+?)$',
                r'for\s+(.+?)$',
            ]

            for pattern in for_patterns:
                match = re.search(pattern, instruction_lower)
                if match:
                    potential_name = match.group(1).strip()
                    print(f"🎯 Extracted potential exercise: '{potential_name}'")

                    # STEP 2: Use fuzzy matching to find the exercise in database
                    best_match = None
                    best_score = 0

                    for eid, exercise_data in cat["by_id"].items():
                        db_name = exercise_data["name"].lower()

                        # Calculate similarity score using the same function as add
                        score = calculate_similarity(potential_name, db_name)

                        if score > best_score and score > 0.4:  # Lower threshold for alternatives
                            best_score = score
                            best_match = exercise_data["name"]

                    if best_match:
                        target_exercise_name = best_match
                        print(f"✅ Fuzzy matched '{potential_name}' -> '{target_exercise_name}' (score: {best_score:.2f})")
                        break
        
        # STEP 3: If fuzzy matching failed, try finding ANY exercise in the current template
        if not target_exercise_name:
            print("🔍 Fuzzy matching failed, searching in current template...")
            
            # Extract any word that might be an exercise name
            words = re.findall(r'[a-zA-Z]+', instruction_lower)
            
            for word_combo_length in [3, 2, 1]:  # Try 3-word, 2-word, 1-word combinations
                for i in range(len(words) - word_combo_length + 1):
                    test_phrase = ' '.join(words[i:i + word_combo_length])
                    
                    # Check if this phrase matches any exercise in current template
                    for day_key, day_data in updated["days"].items():
                        for exercise in day_data.get("exercises", []):
                            exercise_name = exercise.get("name", "").lower()
                            
                            if (test_phrase in exercise_name or 
                                calculate_similarity(test_phrase, exercise_name) > 0.6):
                                target_exercise_name = exercise.get("name")
                                print(f"✅ Found in template: '{test_phrase}' -> '{target_exercise_name}'")
                                break
                        if target_exercise_name:
                            break
                    if target_exercise_name:
                        break
                if target_exercise_name:
                    break
        
        print(f"🎯 Final target exercise: {target_exercise_name}")
        
        # STEP 4: Find and replace the exercise if we identified it
        if target_exercise_name:
            # Find the exercise in the template
            for day_key, day_data in updated["days"].items():
                exercises = day_data.get("exercises", [])
                for i, exercise in enumerate(exercises):
                    exercise_name = exercise.get("name", "")
                    
                    # Check if this is the exercise to replace
                    if (target_exercise_name.lower() == exercise_name.lower() or
                        calculate_similarity(target_exercise_name.lower(), exercise_name.lower()) > 0.8):
                        
                        print(f"🔄 Found exercise to replace: '{exercise_name}' in {day_key}")
                        
                        # Get muscle groups for this day to find appropriate alternative
                        muscle_groups = day_data.get("muscle_groups", [])
                        
                        # Get all currently used exercise IDs in this day to avoid duplicates
                        used_ids = set(ex.get("id") for ex in exercises if ex.get("id"))
                        
                        # Find alternative exercises from database
                        alternative_ids = []
                        
                        # Try muscle groups from the day
                        for muscle in muscle_groups:
                            alt_ids = pick_from_muscles([muscle.lower()], cat, used_ids=used_ids, n=5)
                            alternative_ids.extend(alt_ids)
                        
                        # If no muscle-specific alternatives, try related muscle groups
                        if not alternative_ids:
                            related_muscles = {
                                'chest': ['upper body', 'push'],
                                'upper body': ['chest', 'push'],
                                'push': ['chest', 'upper body'],
                                'back': ['pull', 'upper body'],
                                'pull': ['back', 'upper body'],
                                'legs': ['lower body', 'quadriceps', 'hamstrings'],
                                'lower body': ['legs', 'quadriceps', 'hamstrings'],
                                'core': ['cardio', 'full body'],
                                'cardio': ['core', 'full body'],
                                'full body': ['core', 'cardio']
                            }
                            
                            for muscle in muscle_groups:
                                if muscle.lower() in related_muscles:
                                    for related in related_muscles[muscle.lower()]:
                                        alt_ids = pick_from_muscles([related], cat, used_ids=used_ids, n=3)
                                        alternative_ids.extend(alt_ids)
                        
                        # If still no alternatives, try ANY exercise from database (excluding current)
                        if not alternative_ids:
                            current_id = exercise.get("id")
                            alternative_ids = [eid for eid in cat["by_id"].keys() 
                                             if eid != current_id and eid not in used_ids][:5]
                        
                        print(f"🔍 Found {len(alternative_ids)} potential alternatives")
                        
                        # Pick replacement exercise
                        current_exercise_id = exercise.get("id")
                        replacement_id = None

                        # If user specified a specific replacement, try to find it first
                        if replacement_exercise_name:
                            print(f"🎯 Looking for specific replacement: '{replacement_exercise_name}'")
                            best_replacement_score = 0
                            best_replacement_id = None

                            for eid, exercise_data in cat["by_id"].items():
                                if eid != current_exercise_id and eid not in used_ids:
                                    score = calculate_similarity(replacement_exercise_name.lower(), exercise_data["name"].lower())
                                    if score > best_replacement_score and score > 0.6:  # Higher threshold for specific requests
                                        best_replacement_score = score
                                        best_replacement_id = eid

                            if best_replacement_id:
                                replacement_id = best_replacement_id
                                print(f"✅ Found specific replacement: {cat['by_id'][replacement_id]['name']} (score: {best_replacement_score:.2f})")

                        # If no specific replacement found, use alternatives
                        if not replacement_id:
                            for alt_id in alternative_ids:
                                if (alt_id != current_exercise_id and
                                    alt_id in cat["by_id"] and
                                    alt_id not in used_ids):
                                    replacement_id = alt_id
                                    break
                        
                        if replacement_id:
                            # Replace with database exercise
                            replacement_exercise = cat["by_id"][replacement_id]
                            original_sets = exercise.get("sets", 3)
                            original_reps = exercise.get("reps", 10)
                            
                            exercises[i] = {
                                "id": replacement_id,
                                "name": replacement_exercise["name"],
                                "sets": original_sets,
                                "reps": original_reps,
                                "note": None
                            }
                            
                            print(f"✅ Successfully replaced '{exercise_name}' with '{replacement_exercise['name']}'")
                            return updated, f"Replaced '{exercise_name}' with '{replacement_exercise['name']}' in {day_key.title()}"
                        else:
                            return template, f"No suitable alternative found for '{exercise_name}' (all similar exercises already in use)"
            
            return template, f"Exercise '{target_exercise_name}' not found in current template"
        else:
            return template, "Could not identify which exercise you want an alternative for. Please specify the exercise name more clearly."

def handle_remove_exercise(template: Dict[str, Any], instruction: str, instruction_lower: str) -> Tuple[Dict[str, Any], str]:
    """
    Enhanced exercise removal with intelligent fuzzy matching:
    1. If day is specified: remove only from that day
    2. If no day specified: remove from ALL days
    3. Find best candidate match even with spelling mistakes
    """
    import re

    updated = template.copy()

    # Check if specific day is mentioned
    day_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                   'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7']
    target_day = None

    # Look for day-specific patterns
    for day in day_keywords:
        if day in instruction_lower:
            for day_key in updated.get("days", {}).keys():
                if day in day_key.lower() or day_key.lower() in day:
                    target_day = day_key
                    break
            if target_day:
                break

    # Extract potential exercise names using improved logic
    # First, try to extract exercise names more intelligently

    # Apply exercise-specific preprocessing for better matching
    processed_instruction = instruction_lower

    # Apply exercise-specific corrections first
    corrections = {
        'dumbell': 'dumbbell',
        'dumbel': 'dumbbell',
        'benchpress': 'bench press',
        'shoulderpress': 'shoulder press',
        'chestpress': 'chest press',
        'legpress': 'leg press',
        'pushup': 'push up',
        'pullup': 'pull up',
        'situp': 'sit up',
        'chinup': 'chin up'
    }

    for wrong, correct in corrections.items():
        processed_instruction = processed_instruction.replace(wrong, correct)

    words = re.findall(r'[a-zA-Z]+', processed_instruction)

    # Skip common command words (reduced list to avoid filtering exercise words)
    skip_words = {'remove', 'delete', 'take', 'out', 'from', 'monday', 'tuesday', 'wednesday',
                  'thursday', 'friday', 'saturday', 'sunday', 'day', 'exercise', 'workout',
                  'action', 'specified', 'exercises', 'constraints', 'maximum', 'per',
                  'keep', 'existing', 'structure', 'unless', 'specifically', 'asked', 'to', 'change',
                  'user', 'request'}

    # Filter words and create candidate phrases
    filtered_words = [w for w in words if w not in skip_words and len(w) > 2]

    candidate_phrases = []

    # Generate phrases of different lengths (4, 3, 2, 1 words) - longer first for better specificity
    for word_combo_length in [4, 3, 2, 1]:
        for i in range(len(filtered_words) - word_combo_length + 1):
            phrase = ' '.join(filtered_words[i:i + word_combo_length])
            if len(phrase) > 3:  # Only consider meaningful phrases
                candidate_phrases.append(phrase)

    # Add special handling for common exercise name patterns
    # Look for "X Y Z" patterns that might be exercise names
    exercise_keywords = ['press', 'curl', 'row', 'squat', 'lunge', 'raise', 'extension', 'fly', 'dip', 'push', 'pull']
    for keyword in exercise_keywords:
        if keyword in filtered_words:
            # Find phrases that end with this keyword
            for i, word in enumerate(filtered_words):
                if word == keyword:
                    # Create phrases ending with this keyword
                    for start in range(max(0, i-3), i):
                        phrase = ' '.join(filtered_words[start:i+1])
                        if len(phrase) > 3 and phrase not in candidate_phrases:
                            candidate_phrases.append(phrase)

    if not candidate_phrases:
        print(f"❌ No meaningful exercise name found in instruction")
        return None, None

    # Collect all exercises with their similarity scores
    all_candidates = []

    def collect_exercise_candidates(day_key, exercises, phrases):
        """Collect exercise candidates with similarity scores"""
        candidates = []
        for phrase in phrases:
            for idx, ex in enumerate(exercises):
                exercise_name = ex.get("name", "").lower()
                similarity = calculate_similarity(phrase, exercise_name)
                if similarity > 0.3:  # Lower threshold to catch more candidates
                    candidates.append({
                        'day': day_key,
                        'index': idx,
                        'exercise': ex,
                        'phrase': phrase,
                        'similarity': similarity,
                        'name': ex.get('name', '')
                    })
        return candidates

    if target_day:
        # Search only in specified day
        day_data = updated["days"].get(target_day, {})
        exercises = day_data.get("exercises", [])
        all_candidates = collect_exercise_candidates(target_day, exercises, candidate_phrases)
    else:
        # Search in all days
        for day_key, day_data in updated["days"].items():
            exercises = day_data.get("exercises", [])
            day_candidates = collect_exercise_candidates(day_key, exercises, candidate_phrases)
            all_candidates.extend(day_candidates)

    if not all_candidates:
        print(f"❌ No matching exercises found for any candidate phrase")
        return None, None

    # Smart sorting: prioritize longer phrases and higher similarity
    def candidate_score(candidate):
        phrase_length = len(candidate['phrase'].split())
        similarity = candidate['similarity']
        # Bonus for longer phrases (more specific)
        length_bonus = phrase_length * 0.1
        # Penalty for single word matches unless very high similarity
        if phrase_length == 1 and similarity < 0.8:
            length_bonus = -0.2
        return similarity + length_bonus

    all_candidates.sort(key=candidate_score, reverse=True)

    # Log top candidates for debugging
    print(f"🔍 Top matching candidates:")
    for i, candidate in enumerate(all_candidates[:3]):
        phrase_len = len(candidate['phrase'].split())
        score_with_bonus = candidate_score(candidate)
        print(f"  {i+1}. '{candidate['name']}' in {candidate['day']} (score: {candidate['similarity']:.2f}, phrase: '{candidate['phrase']}' [{phrase_len} words], final: {score_with_bonus:.2f})")

    # Select the best candidate
    best_candidate = all_candidates[0]

    # Apply higher threshold for final decision (better accuracy)
    if best_candidate['similarity'] < 0.5:
        print(f"❌ Best match '{best_candidate['name']}' has low confidence ({best_candidate['similarity']:.2f}). Falling back to LLM.")
        return None, None

    # Remove the best matching exercise
    target_day_key = best_candidate['day']
    target_exercise = best_candidate['exercise']

    day_data = updated["days"][target_day_key]
    exercises = day_data.get("exercises", [])

    # Remove the specific exercise
    updated_exercises = [ex for ex in exercises if ex != target_exercise]
    day_data["exercises"] = updated_exercises

    removed_name = best_candidate['name']

    print(f"🗑️ Removing '{removed_name}' from {target_day_key} (matched '{best_candidate['phrase']}' with score {best_candidate['similarity']:.2f})")
    print(f"✅ Successfully removed exercise from {target_day_key}. Count: {len(exercises)} -> {len(updated_exercises)}")

    return updated, f"Removed '{removed_name}' from {target_day_key}"

    # Note: Legacy remove logic removed - now handled by handle_remove_exercise() above


def enhanced_edit_template(oai, model: str, template: Dict[str,Any], instruction: str, profile_hint: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    """Enhanced edit with support for bulk operations and flexible requests"""
    import re  # Ensure re is available in this function scope
    original_days = list(template.get("days", {}).keys())
    instruction_lower = instruction.lower()
    
    print(f"🔄 Enhanced edit called with instruction: '{instruction}'")
    
    # Helper function to enforce 6-8 exercise limits
    def enforce_exercise_limits(template_dict, respect_user_intent=True):
        """Ensure all days have 6-8 exercises, but respect explicit user reduction requests"""
        from .exercise_catalog_db import load_catalog, pick_from_muscles

        # Handle test mode when db is None
        if db is None:
            print(f"🧪 Test mode: Skipping exercise limit enforcement (no database)")
            return template_dict

        cat = load_catalog(db)
        if not cat:
            return template_dict

        # Check if user explicitly wants to reduce/remove things
        reduction_keywords = [
            'reduce', 'remove', 'delete', 'fewer', 'less', 'cut', 'drop',
            'take out', 'get rid', 'eliminate', 'decrease', 'minimize'
        ]
        user_wants_reduction = respect_user_intent and any(keyword in instruction_lower for keyword in reduction_keywords)

        if user_wants_reduction:
            print(f"🚫 Skipping exercise auto-fill - user requested reduction: {instruction}")
            return template_dict  # Don't auto-fill when user wants to reduce

        updated_template = template_dict.copy()
        days = updated_template.get("days", {})

        
        for day_key, day_data in days.items():
            exercises = day_data.get("exercises", [])
            muscle_groups = day_data.get("muscle_groups", [])
            
            print(f"📊 Day {day_key}: {len(exercises)} exercises before enforcement")
            
            # Enforce minimum 6 exercises
            if len(exercises) < 6:
                used_ids = set(ex.get("id") for ex in exercises if ex.get("id"))
                attempts = 0
                max_attempts = 20  # Prevent infinite loops
                
                while len(exercises) < 6 and attempts < max_attempts:
                    picked = pick_from_muscles(muscle_groups or ["full body"], cat, used_ids=used_ids, n=1)
                    if picked and picked[0] in cat["by_id"]:
                        eid = picked[0]
                        if eid not in used_ids:  # Double-check to avoid duplicates
                            canon = cat["by_id"][eid]
                            exercises.append({
                                "id": eid,
                                "name": canon["name"],
                                "sets": 3,
                                "reps": 10,
                                "note": None,
                            })
                            used_ids.add(eid)
                            print(f"✅ Added exercise: {canon['name']} to {day_key}")
                    else:
                        # Fallback: try any available exercise from catalog
                        available_ids = [eid for eid in cat["by_id"].keys() if eid not in used_ids]
                        if available_ids:
                            eid = available_ids[0]
                            canon = cat["by_id"][eid]
                            exercises.append({
                                "id": eid,
                                "name": canon["name"],
                                "sets": 3,
                                "reps": 10,
                                "note": None,
                            })
                            used_ids.add(eid)
                            print(f"🔄 Fallback added: {canon['name']} to {day_key}")
                        else:
                            break  # No more exercises available
                    attempts += 1
            
            # Enforce maximum 8 exercises
            if len(exercises) > 8:
                exercises = exercises[:8]
                print(f"⚠️ Trimmed {day_key} to 8 exercises (was {len(day_data.get('exercises', []))})")
            
            print(f"📊 Day {day_key}: {len(exercises)} exercises after enforcement")
            day_data["exercises"] = exercises
            days[day_key] = day_data
        
        updated_template["days"] = days
        return updated_template
    
    # Check for bulk operations first - using the local function we just added
    bulk_info = extract_bulk_operation_info(instruction)  # This calls our new function
    if bulk_info['is_bulk_operation'] and bulk_info['target_muscle'] and bulk_info['operation']:
        print(f"🔄 Detected bulk operation: {bulk_info}")
        
        result, summary = SmartWorkoutEditor.handle_bulk_muscle_change(
            template, 
            bulk_info['target_muscle'],
            bulk_info['operation'],
            bulk_info['target_days'],
            bulk_info.get('specific_count'),
            db
        )
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result, respect_user_intent=True)
        return result, summary
    
    if ("change all" in instruction_lower and "exercise" in instruction_lower) or ("replace all" in instruction_lower and "exercise" in instruction_lower):
        print(f"🔄 Detected 'change all exercises' request")
        
        try:
            from .exercise_catalog_db import load_catalog

            # Handle test mode when db is None
            if db is None:
                print(f"🧪 Test mode: Skipping change all exercises (no database)")
                return template, "Test mode: Database operations skipped"

            cat = load_catalog(db)
            if not cat:
                return template, "Could not load exercise database"
            
            updated = template.copy()
            days = updated.get("days", {})
            
            for day_key, day_data in days.items():
                current_exercises = day_data.get("exercises", [])
                current_ids = set(ex.get("id") for ex in current_exercises if ex.get("id"))
                muscle_groups = day_data.get("muscle_groups", [])
                
                print(f"🔄 Changing all exercises for {day_key}")
                print(f"   Current IDs: {current_ids}")
                print(f"   Muscle groups: {muscle_groups}")
                
                # Get ALL available exercises from database
                all_exercise_ids = list(cat["by_id"].keys())
                
                # Filter out current exercises
                available_ids = [eid for eid in all_exercise_ids if eid not in current_ids]
                
                # For leg workouts, prioritize leg exercises
                if any("leg" in mg.lower() for mg in muscle_groups):
                    leg_exercise_names = [
                        'squat', 'lunge', 'leg press', 'leg extension', 'leg curl', 
                        'calf raise', 'bulgarian split squat', 'step up', 'wall sit',
                        'goblet squat', 'romanian deadlift', 'glute bridge', 'hip thrust',
                        'single leg deadlift', 'pistol squat', 'jump squat'
                    ]
                    
                    # Find different leg exercises
                    new_leg_ids = []
                    for eid in available_ids:
                        if len(new_leg_ids) >= 6:  # Changed to 6 for minimum
                            break
                        if eid in cat["by_id"]:
                            exercise_name = cat["by_id"][eid]["name"].lower()
                            if any(leg_name in exercise_name for leg_name in leg_exercise_names):
                                new_leg_ids.append(eid)
                    
                    if len(new_leg_ids) >= 6:  # Changed to 6
                        selected_ids = new_leg_ids[:6]
                    else:
                        # Fallback: use any available exercises
                        selected_ids = available_ids[:6]
                else:
                    # For non-leg workouts, just pick different exercises
                    selected_ids = available_ids[:6]
                
                # Create new exercises
                new_exercises = []
                for eid in selected_ids:
                    if eid in cat["by_id"]:
                        exercise_data = cat["by_id"][eid]
                        new_exercises.append({
                            "id": eid,
                            "name": exercise_data["name"],
                            "sets": 3,
                            "reps": 10,
                            "note": None
                        })
                
                print(f"✅ New exercises for {day_key}: {[ex['name'] for ex in new_exercises]}")
                print(f"   New IDs: {[ex['id'] for ex in new_exercises]}")
                
                # Update the day
                day_data["exercises"] = new_exercises
                days[day_key] = day_data
            
            updated["days"] = days
            
            exercise_names = []
            for day_data in days.values():
                exercise_names.extend([ex.get('name') for ex in day_data.get('exercises', [])])
            
            # Apply exercise limits enforcement
            updated = enforce_exercise_limits(updated, respect_user_intent=True)
            return updated, f"Replaced all exercises with: {', '.join(exercise_names[:3])}{'...' if len(exercise_names) > 3 else ''}"
            
        except Exception as e:
            print(f"❌ Error in change all exercises: {e}")
            return template, f"Could not change all exercises: {str(e)}"
    
    # CRITICAL FIX: Check for remove exercise requests FIRST (highest priority)
    if "remove" in instruction_lower or "delete" in instruction_lower:
        print(f"🗑️ Detected remove exercise request: {instruction}")
        result, summary = handle_remove_exercise(template, instruction, instruction_lower)
        if result is not None:  # If removal was successful
            # Apply exercise limits enforcement (but respect reduction intent)
            result = enforce_exercise_limits(result, respect_user_intent=True)
            return result, summary
        # If result is None, fall through to LLM processing

    # Check for add exercise requests
    if "add" in instruction_lower:
        print(f"🔄 Detected add exercise request: {instruction}")
        result, summary = handle_specific_exercise_addition(template, instruction, db)
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result, respect_user_intent=True)
        return result, summary

    # PRIORITY 1: Title change handling - check this BEFORE replacement keywords
    title_analysis = SmartWorkoutEditor.analyze_title_change(instruction)
    if title_analysis['wants_title_change']:
        print(f"🔄 Detected title change request: {title_analysis}")
        result, summary = SmartWorkoutEditor.apply_title_change(
            template,
            title_analysis['target_day'],
            title_analysis['new_title']
        )
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result, respect_user_intent=True)
        return result, summary

    # PRIORITY 2: Check for alternative/alternate/replacement requests (after title changes)
    replacement_keywords = ["alternate", "alternative", "replace", "change", "swap", "substitute", "different exercise", "something else"]
    print(f"🔍 Checking replacement keywords in '{instruction_lower}': {[kw for kw in replacement_keywords if kw in instruction_lower]}")
    if any(keyword in instruction_lower for keyword in replacement_keywords):
        print(f"🔄 Detected exercise replacement/alternative request: {instruction}")
        result, summary = apply_manual_edit(template, instruction, db)

        # IMPORTANT: Don't enforce limits after removal - user explicitly removed exercises
        if "remove" in instruction_lower or "delete" in instruction_lower:
            print(f"🚫 Skipping exercise auto-fill after removal - user explicitly removed exercises")
            return result, summary

        # Apply exercise limits enforcement only for non-removal operations
        result = enforce_exercise_limits(result, respect_user_intent=True)
        return result, summary
    
    # Continue with existing LLM edit logic...
    print(f"🤖 Falling back to LLM processing for: '{instruction}'")
    try:
        updated, summary = llm_edit_template(oai, model, template, instruction, profile_hint, db)
        
        validation_passed = True
        updated_days = list(updated.get("days", {}).keys())

        # Check if user explicitly wants to change days
        day_reduction_keywords = ['reduce', 'fewer', 'less', 'cut down', 'decrease', 'minimize']
        day_expansion_keywords = ['add', 'more', 'increase', 'expand', 'extra', 'additional']
        user_wants_day_reduction = any(keyword in instruction_lower for keyword in day_reduction_keywords) and 'day' in instruction_lower
        user_wants_day_expansion = any(keyword in instruction_lower for keyword in day_expansion_keywords) and 'day' in instruction_lower

        # ENHANCED: Also check for number-based day changes (e.g., "make it to 4 days", "change to 3 days")
        if ('day' in instruction_lower and not user_wants_day_reduction and not user_wants_day_expansion):
            import re
            # Look for patterns like "to X days", "X days", "make it X days", "for X days"
            number_patterns = [
                r'(?:to|make.*?to|change.*?to|for)\s*(\d+)\s*days?',
                r'(\d+)\s*days?(?:\s+(?:only|total|workout))?',
                r'template.*?for.*?(\d+)\s*days?',
                r'make.*?template.*?(\d+)\s*days?',
            ]

            for pattern in number_patterns:
                match = re.search(pattern, instruction_lower)
                if match:
                    target_days = int(match.group(1))
                    current_days = len(original_days)
                    if target_days < current_days:
                        user_wants_day_reduction = True
                        print(f"🎯 Detected day reduction request: {current_days} → {target_days} days")
                    elif target_days > current_days:
                        user_wants_day_expansion = True
                        print(f"🎯 Detected day expansion request: {current_days} → {target_days} days")
                    break

        if set(updated_days) != set(original_days):
            if (user_wants_day_reduction and len(updated_days) < len(original_days)) or \
               (user_wants_day_expansion and len(updated_days) > len(original_days)):
                change_type = "reduction" if user_wants_day_reduction else "expansion"
                print(f"✅ User requested day {change_type}: {len(original_days)} → {len(updated_days)} days")
                validation_passed = True  # Allow day changes
            else:
                validation_passed = False  # Reject other day structure changes
        
        if validation_passed:
            print(f"✅ LLM edit successful: {summary}")
            # Apply exercise limits enforcement
            updated = enforce_exercise_limits(updated, respect_user_intent=True)
            return updated, summary
        else:
            print(f"❌ LLM validation failed, trying manual edit")
            result, summary = apply_manual_edit(template, instruction, db)

            # IMPORTANT: Don't enforce limits after removal - user explicitly removed exercises
            if "remove" in instruction_lower or "delete" in instruction_lower:
                print(f"🚫 Skipping exercise auto-fill after removal - user explicitly removed exercises")
                return result, summary

            # Apply exercise limits enforcement only for non-removal operations
            result = enforce_exercise_limits(result, respect_user_intent=True)
            return result, summary
            
    except Exception as e:
        print(f"❌ LLM edit exception: {e}, trying manual edit")
        result, summary = apply_manual_edit(template, instruction, db)

        # IMPORTANT: Don't enforce limits after removal - user explicitly removed exercises
        if "remove" in instruction_lower or "delete" in instruction_lower:
            print(f"🚫 Skipping exercise auto-fill after removal - user explicitly removed exercises")
            return result, summary

        # Apply exercise limits enforcement only for non-removal operations
        result = enforce_exercise_limits(result, respect_user_intent=True)
        return result, summary
    
# ───────────────────── LLM: explain rationale ───────────────
def explain_template_with_llm(oai, model: str, profile: Dict[str,Any], template: Dict[str,Any]) -> str:
    sys = "Explain briefly (2–4 sentences) the training logic. Plain English. No markdown."
    usr = "Client profile:\n" + orjson.dumps(profile).decode() + "\n\nTemplate (Mon–Sat only):\n" + orjson.dumps(template).decode()
    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return "Compound-first approach with weekly distribution tailored to your goal, experience, and Mon–Sat frequency."


