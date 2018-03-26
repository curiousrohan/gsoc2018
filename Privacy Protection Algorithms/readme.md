# Privacy Protection Algorithms
To demonstrate the proof of work for this project, I thought of doing something new. Using the Rocket Chat API, I pulled all the massages from the gsoc2018 channel of clips. Then, I cleaned the messages by removing mentions, links etc. I also tried Stanford NER module, but the code took a long time to run and was't that effective, so I removed it.Tools used are: Regular expressions and simple python functions.
The final result is a clean csv file called output.csv which can now be used for any kind of text analysis.
