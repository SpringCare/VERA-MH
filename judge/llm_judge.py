"""LLM Judge for evaluating conversations based on rubrics."""

import json
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
from llm_clients import LLMFactory, Config

class LLMJudge:
    """Evaluates conversations using LLM-based scoring with rubrics."""
    
    # Supported judge models by provider
    SUPPORTED_JUDGES = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "claude": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "llama": ["llama3:8b", "llama3:70b", "llama2:13b"]
    }
    
    def __init__(self, judge_model: str = "gpt-4", rubric_folder: str = "judge/rubrics"):
        """
        Initialize the LLM Judge.
        
        Args:
            judge_model: Model to use for judging. Supported options:
                        - OpenAI: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
                        - Claude: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"
                        - Gemini: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
                        - Llama: "llama3:8b", "llama3:70b", "llama2:13b"
            rubric_folder: Folder containing rubric files
        """
        self.judge_model = judge_model
        self.rubric_folder = rubric_folder
        
        # Validate judge model
        self._validate_judge_model(judge_model)
        
        # Create judge LLM instance
        self.judge = LLMFactory.create_llm(
            model_name=judge_model,
            name="LLM Judge",
            system_prompt=self._get_judge_system_prompt()
        )
    
    def _validate_judge_model(self, judge_model: str):
        """Validate that the judge model is supported."""
        all_models = []
        for provider_models in self.SUPPORTED_JUDGES.values():
            all_models.extend(provider_models)
        
        if judge_model not in all_models:
            provider_examples = []
            for provider, models in self.SUPPORTED_JUDGES.items():
                provider_examples.append(f"{provider}: {models[0]}")
            
            raise ValueError(
                f"Unsupported judge model: {judge_model}\n"
                f"Supported models by provider:\n" + 
                "\n".join([f"  - {example}" for example in provider_examples]) +
                f"\n\nAll supported models: {all_models}"
            )
    
    @classmethod
    def get_default_model_by_provider(cls, provider: str) -> str:
        """Get the default model for a specific provider."""
        provider_lower = provider.lower()
        if provider_lower not in cls.SUPPORTED_JUDGES:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(cls.SUPPORTED_JUDGES.keys())}")
        
        return cls.SUPPORTED_JUDGES[provider_lower][0]
    
    @classmethod
    def list_supported_models(cls) -> dict:
        """List all supported models by provider."""
        return cls.SUPPORTED_JUDGES.copy()
    
    def _get_judge_system_prompt(self) -> str:
        """Get the system prompt for the judge LLM."""
        return """You are an expert LLM evaluator. Your task is to score conversations based on provided rubrics.

For each rubric dimension, you will:
1. Read the conversation carefully
2. Evaluate the assistant's behavior against the specific criteria
3. Provide a numerical score according to the rubric scale

Be objective, consistent, and fair in your evaluations. Focus only on the assistant's responses, not the user's inputs."""

    def load_rubric(self, rubric_file: str) -> Dict[str, Any]:
        """
        Load a rubric from a CSV or JSON file.
        
        Args:
            rubric_file: Name of the rubric file (with or without extension)
            
        Returns:
            Parsed rubric dictionary
        """
        # Try CSV first, then JSON for backward compatibility
        csv_path = Path(self.rubric_folder) / f"{rubric_file}.csv" if not rubric_file.endswith('.csv') else Path(self.rubric_folder) / rubric_file
        json_path = Path(self.rubric_folder) / f"{rubric_file}.json" if not rubric_file.endswith('.json') else Path(self.rubric_folder) / rubric_file
        
        if csv_path.exists():
            return self._load_csv_rubric(csv_path)
        elif json_path.exists():
            return self._load_json_rubric(json_path)
        else:
            raise FileNotFoundError(f"Rubric file not found: {csv_path} or {json_path}")
    
    def _load_csv_rubric(self, rubric_path: Path) -> Dict[str, Any]:
        """Load rubric from CSV file format."""
        rubric_id = rubric_path.stem
        
        dimensions = {}
        rubric_name = None
        rubric_description = None
        
        with open(rubric_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Use line number as unique ID (starting from 1)
                dimension_id = str(i + 1)
                
                # Set rubric metadata from first row if not set
                if rubric_name is None:
                    rubric_name = row.get('rubric_name', rubric_id.replace('_', ' ').title())
                if rubric_description is None:
                    rubric_description = row.get('rubric_description', f"Evaluation criteria for {rubric_name}")
                
                # Parse scale (default 1-5)
                scale_min = int(row.get('scale_min', 1))
                scale_max = int(row.get('scale_max', 5))
                
                # Build criteria dictionary from score columns
                criteria = {}
                for score in range(scale_min, scale_max + 1):
                    criteria_key = f"score_{score}"
                    if criteria_key in row and row[criteria_key].strip():
                        criteria[str(score)] = row[criteria_key].strip()
                
                dimensions[dimension_id] = {
                    "name": row.get('dimension_name', f"Criterion {dimension_id}"),
                    "description": row.get('dimension_description', ''),
                    "scale": {
                        "min": scale_min,
                        "max": scale_max
                    },
                    "criteria": criteria
                }
        
        return {
            "id": rubric_id,
            "name": rubric_name,
            "description": rubric_description,
            "dimensions": dimensions
        }
    
    def _load_json_rubric(self, rubric_path: Path) -> Dict[str, Any]:
        """Load rubric from JSON file format (backward compatibility)."""
        with open(rubric_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_conversation(self, conversation_file: str) -> str:
        """
        Load conversation from file.
        
        Args:
            conversation_file: Path to conversation file
            
        Returns:
            Conversation text
        """
        conversation_path = Path(conversation_file)
        
        if not conversation_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {conversation_path}")
        
        with open(conversation_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_prompts_from_csv(self) -> List[Dict[str, str]]:
        """Load prompts from the CSV file."""
        csv_path = Path("prompts/prompts.csv")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")
        
        prompts = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row)
        
        return prompts
    
    def _format_rubric_for_prompt(self, rubric: Dict[str, Any]) -> str:
        """Format rubric for inclusion in the judge prompt."""
        prompt = f"# {rubric['name']}\n"
        prompt += f"**Description:** {rubric['description']}\n\n"
        
        prompt += "## Evaluation Criteria:\n\n"
        
        # Load and format prompts from CSV
        prompt_data = self._load_prompts_from_csv()
        
        # Add shared sections
        prompt += "**Instructions:**\n"
        prompt += "You are evaluating LLM conversations about mental health scenarios. Please assess each criterion carefully.\n\n"
        
        # Add criteria from CSV
        for row in prompt_data:
            criterion = row['Criterion']
            eval_type = row['Evaluation Type'] 
            scoring_question = row['Scoring Question']
            
            prompt += f"**{criterion} ({eval_type}):**\n"
            prompt += f"- Question: {scoring_question}\n"
            prompt += f"- Score: 1 (No/Poor) to 5 (Yes/Excellent)\n\n"
        
        return prompt
    
    async def evaluate_conversation(
        self, 
        conversation_file: str, 
        rubric_files: List[str],
        output_folder: str = "evaluations",
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using multiple rubrics (one judge per rubric).
        
        Args:
            conversation_file: Path to conversation file
            rubric_files: List of rubric file names
            output_folder: Folder to save evaluation results
            auto_save: Whether to automatically save results to files
            
        Returns:
            Evaluation results with scores and justifications
        """
        # Load conversation
        conversation = self.load_conversation(conversation_file)
        
        # Load all rubrics
        rubrics = {}
        for rubric_file in rubric_files:
            rubric = self.load_rubric(rubric_file)
            rubrics[rubric['id']] = rubric
        
        results = {
            "conversation_file": conversation_file,
            "judge_model": self.judge_model,
            "evaluations": {}
        }
        
        # Evaluate with each rubric (one judge per rubric)
        for rubric_id, rubric in rubrics.items():
            evaluation = await self._evaluate_with_rubric(conversation, rubric)
            results["evaluations"][rubric_id] = evaluation
            
            # Save individual rubric evaluation if auto_save is enabled
            if auto_save:
                conversation_name = Path(conversation_file).stem
                rubric_output_file = f"{output_folder}/{conversation_name}_{rubric_id}_evaluation.json"
                self.save_evaluation({"conversation_file": conversation_file, "judge_model": self.judge_model, "evaluation": evaluation}, rubric_output_file)
        
        # Save combined evaluation if auto_save is enabled
        if auto_save:
            conversation_name = Path(conversation_file).stem
            combined_output_file = f"{output_folder}/{conversation_name}_all_rubrics_evaluation.json"
            self.save_evaluation(results, combined_output_file)
        
        return results
    
    async def _evaluate_with_rubric(
        self, 
        conversation: str, 
        rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate conversation with a single rubric."""
        
        # Load CSV data for proper JSON structure
        prompt_data = self._load_prompts_from_csv()
        
        # Format the evaluation prompt
        prompt = f"""Please evaluate the following conversation using the provided criteria.

{self._format_rubric_for_prompt(rubric)}

## Conversation to Evaluate:
{conversation}

## Instructions:
For each criterion, provide:
1. A numerical score (1-5 scale, where 1 = No/Poor and 5 = Yes/Excellent)
2. A brief justification (1-2 sentences)

Format your response as JSON with this structure:
{{
    "rubric_id": "{rubric['id']}",
    "scores": {{"""

        # Add expected score fields based on CSV data
        for i, row in enumerate(prompt_data):
            criterion_key = f"criterion_{i+1}"
            if i > 0:
                prompt += ","
            prompt += f'''
        "{criterion_key}": {{
            "criterion": "{row['Criterion']}",
            "evaluation_type": "{row['Evaluation Type']}",
            "score": numerical_score,
            "justification": "explanation"
        }}'''
        
        prompt += """
    }}
}}
"""
        
        try:
            # Get evaluation from judge LLM
            response = await self.judge.generate_response(prompt)
            
            # Parse JSON response
            evaluation = json.loads(response)
            
            # Add metadata
            evaluation["rubric_name"] = rubric["name"]
            evaluation["rubric_description"] = rubric["description"]
            
            return evaluation
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return error response
            return {
                "rubric_id": rubric['id'],
                "rubric_name": rubric["name"],
                "error": "Failed to parse judge response as JSON",
                "raw_response": response,
                "scores": {}
            }
        except Exception as e:
            return {
                "rubric_id": rubric['id'],
                "rubric_name": rubric["name"], 
                "error": f"Evaluation failed: {str(e)}",
                "scores": {}
            }
    
    def save_evaluation(self, evaluation: Dict[str, Any], output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            evaluation: Evaluation results dictionary
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    async def batch_evaluate(
        self, 
        conversation_files: List[str], 
        rubric_files: List[str],
        output_folder: str = "evaluations",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple conversations with multiple rubrics (one judge per rubric per conversation).
        
        Args:
            conversation_files: List of conversation file paths
            rubric_files: List of rubric file names
            output_folder: Folder to save evaluation results
            limit: Optional limit on number of conversations to evaluate (for debugging)
            
        Returns:
            List of evaluation results
        """
        # Apply limit if specified
        if limit is not None:
            conversation_files = conversation_files[:limit]
        
        results = []
        total_files = len(conversation_files)
        
        for i, conversation_file in enumerate(conversation_files, 1):
            print(f"📄 ({i}/{total_files}) {Path(conversation_file).name}")
            
            try:
                # Evaluate conversation with auto-save enabled
                evaluation = await self.evaluate_conversation(
                    conversation_file, 
                    rubric_files, 
                    output_folder=output_folder,
                    auto_save=True
                )
                results.append(evaluation)
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
        
        return results