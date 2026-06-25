Hi everyone,

Just to give a general introduction to adding a language.

Adding a language:
Translate the 100 templates if they don’t already exist in the folder you can use this script to get you started.
Validate and localize (currency, metric/imperial, etc.) the templates. This might require you to adapt the code e.g. if you language has specific features.
Run tests, those might reveal a few template inconsistencies
generate and evaluate an LLM on the data and inspect the errors. This should allow you to discover if the errors are data errors - this often reveals a few additional errors

Example pr by @Isaac :
https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic/pull/21

Co-authorship:
Doing a translation get you invited as a co-author. Communication will happen in this channel. You will, of course, have to review and agree with the written content. You are more than free to contribute beyond just creating the template. There will be a contribution section that specifies contributions for transparency.

Feel free to ask if you have any questions.