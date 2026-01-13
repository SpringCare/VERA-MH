#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."


folders=(
    # "conversations/p_gpt_4o__a_claude_opus_4_1_20250805__t20__r5__20251216_123504"              
    # "conversations/p_gpt_4o__a_llmo_workflow_be_heard__t20__r5__20260107_213951"
    # "conversations/p_gpt_4o__a_claude_sonnet_4_5_20250929__t20__r5__20251216_123733"                
    # "conversations/p_gpt_4o__a_llmo_workflow_be_heard__t30__r5__20251215_201330"
    # "conversations/p_gpt_4o__a_gemini_2_5_flash__t20__r5__20260105_193354"                         
    # "conversations/p_gpt_4o__a_llmo_workflow_be_heard__t30__r5__20260105_192036"
    # "conversations/p_gpt_4o__a_gemini_3_pro_preview__t20__r5__20260105_193955"                      
    # "conversations/p_gpt_4o__a_llmo_workflow_intake__t20__r5__20260107_223950"
    # "conversations/p_gpt_4o__a_gpt_4o__t20__r5__20251216_122707"                                    
    # "conversations/p_gpt_4o__a_llmo_workflow_intake__t30__r5__20251215_231251"
    # "conversations/p_gpt_4o__a_gpt_5_2_{'max_completion_tokens': 5000}__t20__r5__20251216_122933"   
    # "conversations/p_gpt_4o__a_llmo_workflow_intake__t30__r5__20260105_204523"
    # "conversations/p_gpt_4o__a_llmo__t30__r5__20251104_222027"
    # "conversations/p_gpt_4o__a_gpt_5_2_max_completion_tokens_5000__t20__r5__20251216_122933"
)

judges=(
    'gpt-4o'
    
    # 'claude-sonnet-4-5-20250929'
)

for folder in "${folders[@]}"; do
    for judge in "${judges[@]}"; do
        echo "Running with folder: $folder, judge: $judge"
        echo "python3 judge.py -f $folder -j $judge"
        python3 judge.py -f $folder -j $judge
    done
done


