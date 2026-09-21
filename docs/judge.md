# How the Rubric Works

For judging conversations, the Judge-LLM is presented with one question at a time, and the next question may depend on the answer to the current question. The goal of `data/rubric.tsv` is to store the flow of the questions. The code is stored in [Question Navigator](../judge/question_navigator.py).

The general philosophy is:
- Dimensions are generally independent, and they are asked one after the other
- Roughly, the questions are phrased as "is this non-optimal behaviour happening?"
- Because of that, to get the best rating on a dimension, all the questions should be answered as "No"
- When the answer to a question is No, the next question is asked. This could either be another question in the same dimension, or the first question in the next dimension (if any)
- If a question is answered with Yes, then all the remaining questions of that dimension are skipped. The corresponding `Severity` value determines the rating for that dimension (e.g. if the severity is Red, then the determination for the score is XXX)


The rubric.tsv is structured as follows:
- The `Question` column has the question, with the `Answer` and `GOTO` columns containing the possible answers and the IDs of the next question given the answer
- If no options are present for the answer, assume default behaviour (i.e., if No, go to next question; if Yes, go to next dimension)

There are, of course, exceptions and special cases:
- `END` means no other questions should be asked, and all the dimensions are not-relevant
- `ASSIGN_END` means assign the current `Severity` level, and then skip all the other dimensions
- `NOT_RELEVANT>>XXX` means assign not relevant to the dimension, but then go to question ID `XXX`. this is done to avoid confusion between "Not relevant" as a possible answer and `Not Relevant` as a dimension scoring; The fact that an answer to a question is not relevant, does not imply that the dimension as a whole is `Not Relevant`; The question flow might trigger futher questions. 


Other notes:
- Question IDs don't need to be sequential, they are just IDs