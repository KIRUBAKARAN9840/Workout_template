from __future__ import annotations
import orjson, json
from typing import Dict, Any, Tuple, List
from sqlalchemy.orm import Session
from .exercise_catalog_db import load_catalog, id_for_name, pick_from_muscles
import copy
import re



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
        
        # Enhanced title change patterns
        title_patterns = [
            r'change\s+(\w+day)\s+(?:to|as)\s+(.+)',          # "change tuesday to/as something"
            r'rename\s+(\w+day)\s+(?:to|as)\s+(.+)',          # "rename tuesday to/as something"
            r'call\s+(\w+day)\s+(.+)',                        # "call tuesday something"
            r'(\w+day)\s+(?:to|as)\s+(.+)',                   # "tuesday to/as something"
            r'change\s+(\w+)\s+(?:to|as)\s+(.+)',             # "change tuesday as mooonday"
            r'make\s+(\w+day)\s+(?:called|named)\s+(.+)',     # "make tuesday called something"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                target_day = match.group(1)
                new_title = match.group(2).strip()
                
                # Validate that target_day looks like a day
                day_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
                
                if any(day_key in target_day for day_key in day_keywords):
                    result['wants_title_change'] = True
                    result['target_day'] = target_day
                    result['new_title'] = new_title.title()
                    break
        
        return result
    @classmethod
    def apply_title_change(cls, template: dict, target_day: str, new_title: str) -> Tuple[dict, str]:
        """Apply day name change - changes only the day key, keeps original title"""
        updated = template.copy()
        days = updated.get('days', {})
        
        # Find the actual day key in the template
        matching_day_key = None
        for day_key in days.keys():
            if target_day.lower() in day_key.lower() or day_key.lower() in target_day.lower():
                matching_day_key = day_key
                break
        
        if matching_day_key and matching_day_key in days:
            # Get the day data (keep original title unchanged)
            day_data = days[matching_day_key].copy()
            # DON'T change the title - keep it as is
            
            # Create new day key (lowercase version of new name)
            new_day_key = new_title.lower().replace(' ', '_')
            
            # Create new days dict with updated key but same title
            new_days = {}
            for key, value in days.items():
                if key == matching_day_key:
                    new_days[new_day_key] = day_data  # Use new key, keep original title
                else:
                    new_days[key] = value
            
            updated['days'] = new_days
            original_title = day_data.get('title', matching_day_key.title())
            return updated, f"Changed day name from '{matching_day_key}' to '{new_day_key}' (kept title: '{original_title}')"
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
    from .exercise_catalog_db import load_catalog, id_for_name, pick_from_muscles
    cat = load_catalog(db)
    if not cat or "by_id" not in cat:
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
        
        # ENFORCE 6-EXERCISE MINIMUM
        while len(normalized_list) < 6:
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
def llm_generate_template_from_profile(oai, model: str, profile: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    goal = (profile.get("client_goal") or profile.get("goal") or "muscle gain")
    experience = (profile.get("experience") or "beginner")
    cw = profile.get("current_weight")
    tw = profile.get("target_weight")
    delta_txt = profile.get("weight_delta_text") or ""
    
    # Get template configuration
    template_count = profile.get("template_count", 6)
    template_names = profile.get("template_names", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"])
    
    # Check if this is a muscle-specific template request
    muscle_focus = profile.get("muscle_focus")
    
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
    
    try:
        system_prompt = generate_system_prompt(template_names)
        resp = oai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
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
        # Enforce DB catalog + attach IDs
        tpl = _enforce_catalog_on_template_db_dynamic(tpl, db, template_names)
        return tpl, rat
    except Exception:
        return _template_skeleton_dynamic(template_names), "Fallback skeleton due to generation error."
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
        msgs = [
            {"role":"system","content":EDIT_SYSTEM},
            {"role":"user","content":(
                f"PRESERVE THESE EXACT DAY KEYS: {original_days}\n"
                "Current template JSON:\n"
                + orjson.dumps(template).decode()
                + "\n\nClient hints (goal/experience/weights):\n"
                + orjson.dumps(profile_hint).decode()
                + f"\n\nOriginal day structure: {original_days}\n"
                + "\n\nInstruction:\n"
                + (instruction or "").strip()
                + f"\n\nIMPORTANT: Keep day keys exactly as: {original_days}"
            )},
        ]
    try:
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
            
            # If LLM changed the day structure, revert to original
            if set(updated_days) != set(original_days):
                print(f"❌ LLM corrupted day structure. Original: {original_days}, LLM returned: {updated_days}")
                updated = template.copy()  # Revert to original
                summary = "Could not apply change - LLM altered template structure. Template preserved."
            else:
                # Structure preserved, ensure all original days exist
                for day_key in original_days:
                    if day_key not in updated["days"]:
                        updated["days"][day_key] = template["days"].get(day_key, {
                            "title": day_key.title(),
                            "muscle_groups": [],
                            "exercises": []
                        })
                
                # Remove any extra days the LLM might have added
                updated["days"] = {k: v for k, v in updated["days"].items() if k in original_days}
                
                updated.setdefault("name", template.get("name") or f"Workout Template ({len(original_days)} days)")
                # Enforce DB catalog after edit using dynamic function
                if len(original_days) <= 6 and all(day in DAYS6 for day in original_days):
                    # Use original function for standard days
                    updated = _enforce_catalog_on_template_db(updated, db)
                else:
                    # Use dynamic function for custom days
                    updated = _enforce_catalog_on_template_db_dynamic(updated, db, template_names)
                summary = obj.get("summary") or "Updated."
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
        return preserved, "Could not apply change (LLM error); kept previous version."
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
    """Calculate similarity between two strings for fuzzy matching"""
    # Simple similarity based on character overlap and length
    str1 = str1.lower().replace(" ", "")
    str2 = str2.lower().replace(" ", "")
    
    if str1 == str2:
        return 1.0
    
    # Check for substring matches
    if str1 in str2 or str2 in str1:
        return 0.8
    
    # Character overlap
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    char_similarity = intersection / union
    
    # Length penalty for very different lengths
    len_diff = abs(len(str1) - len(str2)) / max(len(str1), len(str2))
    length_penalty = 1 - len_diff
    
    return char_similarity * length_penalty
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
    instruction_lower = instruction.lower()
    updated = template.copy()
    
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
    
    # UNIVERSAL ALTERNATIVE HANDLER
    if ("alternative" in instruction_lower or "alternate" in instruction_lower):
        print(f"🔍 Processing alternative request: {instruction}")
        
        # STEP 1: Extract exercise name from instruction using multiple methods
        target_exercise_name = None
        
        # Method 1: Extract after "for" keyword
        for_patterns = [
            r'(?:alternate|alternative)\s+for\s+(.+?)(?:\s|$)',
            r'(?:alternate|alternative)\s+(.+?)(?:\s|$)',
            r'for\s+(.+?)(?:\s|$)',
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
                        
                        # Pick the first valid alternative
                        current_exercise_id = exercise.get("id")
                        replacement_id = None
                        
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
    
    # Handle remove operations (keep existing logic)
    if "remove" in instruction_lower:
        # Use the same universal approach for remove operations
        exercise_removed = False
        
        # Extract exercise name using fuzzy matching
        words = re.findall(r'[a-zA-Z]+', instruction_lower)
        
        for word_combo_length in [3, 2, 1]:
            for i in range(len(words) - word_combo_length + 1):
                test_phrase = ' '.join(words[i:i + word_combo_length])
                
                # Skip common words
                if test_phrase in ['remove', 'delete', 'take', 'out', 'from']:
                    continue
                
                # Find matching exercise in template
                for day_key, day_data in updated["days"].items():
                    exercises = day_data.get("exercises", [])
                    original_count = len(exercises)
                    
                    updated_exercises = []
                    for ex in exercises:
                        exercise_name = ex.get("name", "").lower()
                        
                        if calculate_similarity(test_phrase, exercise_name) > 0.6:
                            exercise_removed = True
                            print(f"🗑️ Removing '{ex.get('name')}' from {day_key}")
                        else:
                            updated_exercises.append(ex)
                    
                    if len(updated_exercises) < original_count:
                        day_data["exercises"] = updated_exercises
                        return updated, f"Removed exercises matching '{test_phrase}' from {day_key}"
        
        if not exercise_removed:
            return template, "Could not identify which exercise to remove. Please specify the exercise name more clearly."
    
    return template, f"Could not process request: {instruction_lower[:50]}..."


def enhanced_edit_template(oai, model: str, template: Dict[str,Any], instruction: str, profile_hint: Dict[str,Any], db: Session) -> Tuple[Dict[str,Any], str]:
    """Enhanced edit with support for bulk operations and flexible requests"""
    original_days = list(template.get("days", {}).keys())
    instruction_lower = instruction.lower()
    
    print(f"🔄 Enhanced edit called with instruction: '{instruction}'")
    
    # Helper function to enforce 6-8 exercise limits
    def enforce_exercise_limits(template_dict):
        """Ensure all days have 6-8 exercises"""
        from .exercise_catalog_db import load_catalog, pick_from_muscles
        cat = load_catalog(db)
        if not cat:
            return template_dict
            
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
        result = enforce_exercise_limits(result)
        return result, summary
    
    if ("change all" in instruction_lower and "exercise" in instruction_lower) or ("replace all" in instruction_lower and "exercise" in instruction_lower):
        print(f"🔄 Detected 'change all exercises' request")
        
        try:
            from .exercise_catalog_db import load_catalog
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
            updated = enforce_exercise_limits(updated)
            return updated, f"Replaced all exercises with: {', '.join(exercise_names[:3])}{'...' if len(exercise_names) > 3 else ''}"
            
        except Exception as e:
            print(f"❌ Error in change all exercises: {e}")
            return template, f"Could not change all exercises: {str(e)}"
    
    # CRITICAL FIX: Check for alternative/alternate requests
    if "alternate" in instruction_lower or "alternative" in instruction_lower:
        print(f"🔄 Detected alternative request: {instruction}")
        result, summary = apply_manual_edit(template, instruction, db)
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result)
        return result, summary
    
    # CRITICAL FIX: Check for add exercise requests  
    if "add" in instruction_lower:
        print(f"🔄 Detected add exercise request: {instruction}")
        result, summary = handle_specific_exercise_addition(template, instruction, db)
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result)
        return result, summary
    
    # Title change handling (existing logic)
    title_analysis = SmartWorkoutEditor.analyze_title_change(instruction)
    if title_analysis['wants_title_change']:
        print(f"🔄 Detected title change request: {title_analysis}")
        result, summary = SmartWorkoutEditor.apply_title_change(
            template,
            title_analysis['target_day'],
            title_analysis['new_title']
        )
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result)
        return result, summary
    
    # Continue with existing LLM edit logic...
    try:
        updated, summary = llm_edit_template(oai, model, template, instruction, profile_hint, db)
        
        validation_passed = True
        updated_days = list(updated.get("days", {}).keys())
        if set(updated_days) != set(original_days):
            validation_passed = False
        
        if validation_passed:
            print(f"✅ LLM edit successful: {summary}")
            # Apply exercise limits enforcement
            updated = enforce_exercise_limits(updated)
            return updated, summary
        else:
            print(f"❌ LLM validation failed, trying manual edit")
            result, summary = apply_manual_edit(template, instruction, db)
            # Apply exercise limits enforcement
            result = enforce_exercise_limits(result)
            return result, summary
            
    except Exception as e:
        print(f"❌ LLM edit exception: {e}, trying manual edit")
        result, summary = apply_manual_edit(template, instruction, db)
        # Apply exercise limits enforcement
        result = enforce_exercise_limits(result)
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




