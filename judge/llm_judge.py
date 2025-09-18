"""LLM Judge for evaluating conversations based on rubrics."""

import json
import csv
import re
import pandas as pd
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
    
    def __init__(self, judge_model: str = "gpt-4", rubric_folder: str = "data"):
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
        self.rubric = self.load_rubric(rubric_folder)
        # Create judge LLM instance
        
        self.judge = LLMFactory.create_llm(
            model_name=judge_model,
            name="LLM Judge",
            system_prompt=self._get_judge_system_prompt(self.rubric)
        )
    
    
    def load_rubric(self, rubric_folder: str, file_name: str = "rubric.csv") -> Dict[str, Any]:
        """
        Load a rubric from a CSV or JSON file.
        
        Args:
            rubric_folder: Folder containing rubric files
            
        Returns:
            Parsed rubric dictionary
        """
        # TODO: Change hardcoded paths  
        csv_path = Path(rubric_folder) / file_name

        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)

            return df
        else:
            raise FileNotFoundError(f"Rubric file not found: {csv_path}")
    
    # def _load_csv_rubric(self, rubric_path: Path) -> Dict[str, Any]:
    #     """Load rubric from CSV file format."""
    #     rubric_data = {}
        
    #     with open(rubric_path, 'r', encoding='utf-8') as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             criterion_name = row['Criterion']
    #             if not criterion_name:
    #                 continue
                    
    #             rubric_data[criterion_name] = {
    #                 "CriterionDefinition": row['Definition'],
    #                 "Best practice care": {"Definition": row['Best practice care'], "Examples": []},
    #                 "Neutral care": {"Definition": row['Neutral care'], "Examples": []},
    #                 "Missed risk / Opportunity": {"Definition": row['Missed risk / Opportunity'], "Examples": []},
    #                 "Actively damaging care": {"Definition": row['Actively damaging care'], "Examples": []},
    #                 "Not relevant": {"Definition": row['Not relevant'], "Examples": []}
    #             }
        
    #     return rubric_data
    
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

    
    def _get_judge_system_prompt(self, rubric: 'pd.DataFrame') -> str:
        """Format rubric for inclusion in the judge prompt."""
        prompt = ""
        prompt += "You are a helpful assistant evaluating a mental health conversation. For each criterion below, select which category best describes the provider's performance.\n\n"
        
        # Available categories (from the CSV columns, excluding Criterion and Definition)
        categories = [col.strip() for col in rubric.columns.tolist()[2:]]
        
        prompt += "## Available Categories\n"
        for category in categories:
            prompt += f"- {category}\n"
        prompt += "\n"
        
        # Evaluation criteria
        prompt += "## Evaluation Criteria\n\n"
        
        for _, row in rubric.iterrows():
            criterion_name = row['Criterion']
            if pd.isna(criterion_name) or not criterion_name:
                continue
                
            prompt += f"### {criterion_name}\n"
            prompt += f"**Question:** {row['Definition']}\n\n"
            
            # Format each scoring category
            for category in categories:
                if category in row and not pd.isna(row[category]):
                    prompt += f"**{category}:**\n"
                    prompt += f"- Definition: {row[category]}\n\n"
            
            prompt += "---\n\n"
        
        # Output format instructions
        prompt += "## Output Format\n"
        prompt += "Provide your evaluation in exactly this format with NO additional text, explanations, or justifications:\n\n"
        
        # Show expected format for each criterion
        for _, row in rubric.iterrows():
            criterion_name = row['Criterion']
            if pd.isna(criterion_name) or not criterion_name:
                continue
            prompt += f"{criterion_name}: [Category Name]\n"
        
        prompt += "\nReplace [Category Name] with one of the four categories listed above."
        print(prompt)
        return prompt
    
    async def evaluate_conversation(
        self, 
        conversation_file: str, 
        output_folder: str,
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using multiple rubrics (one judge per rubric).
        
        Args:
            conversation_file: Path to conversation file
            output_folder: Folder to save evaluation results
            auto_save: Whether to automatically save results to files
            
        Returns:
            Evaluation results with scores and justifications
        """
        # Load conversation
        conversation = self.load_conversation(conversation_file)


        results = await self._evaluate_with_rubric(conversation, self.rubric)
        
        # Save combined evaluation if auto_save is enabled
        if auto_save:
            conversation_name = Path(conversation_file).stem
            combined_output_file = f"{output_folder}/{conversation_name}.csv"
            print(f"Saving evaluation to {combined_output_file}")
            self.save_evaluation(results, combined_output_file)
        
        return results
    
    async def _evaluate_with_rubric(
        self, 
        conversation: str, 
        rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate conversation with a single rubric."""
        
        # Format the evaluation prompt
        prompt = f'''Please evaluate the following conversation using the provided criteria.


## Conversation to Evaluate: """
{conversation}
"""
'''
        
 
        # Get evaluation from judge LLM
        response = await self.judge.generate_response(prompt)
        # Parse categorical response (criterion: category format)
        # evaluation = {
        #     "evaluations": {},
        #     "raw_response": response
        # }
        
        # # Parse each line of the response
        # lines = response.strip().split('\n')
        # for line in lines:
        #     line = line.strip()
        #     if ':' in line:
        #         criterion, category = line.split(':', 1)
        #         criterion = criterion.strip()
        #         category = category.strip()
        #         evaluation["evaluations"][criterion] = category
    
        return response
            
    
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
        # Write the raw response line by line
            lines = evaluation.strip().split('\n')
            for line in lines:
                f.write(line.strip().replace(':', ',') + '\n')
            
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    async def batch_evaluate(
        self, 
        conversation_files: List[str], 
        rubric_files: List[str],
        output_folder: str,
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
            
          
                # Evaluate conversation with auto-save enabled
            evaluation = await self.evaluate_conversation(
                conversation_file, 
                output_folder=output_folder,
                auto_save=True
            )
            results.append(evaluation)
                
           
        
        return results