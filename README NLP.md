# InternshipMeduzzen
Here is my opinion on Natural Language Processing challenge.
## Libraries I am going to use ##
- **Gradio**<br/>
  This library is convenient to create a simple GUI and it is easy to start with.
- **spaCy**<br/>
  Modern NLP library, which supports around 70 languages and has pipelines for 25+ languages. As for now, I think I will use spaCy as a main NLP library in the project because it has built-in lemmatization and is more convenient for   
  smaller projects.
- **NLTK**<br/>
  This library might be useful, as it is more flexible than spaCy and gives more control of NLP pipeline.
- **Pandas**<br/>
  This library is useful for data structurization. I could structurize previous inputs and outputs to show as an example for users.
- **TensorFlow Keras**<br/>
  TensorFlow offers a wide variety of pre-trained models for NLP, such as BERT, GPT or Word2Vec, which might be useful.
- **Gensim**<br/>
  Provides vectorization tools and topic modelling tools, which might be useful for text analysis.
## Vision of the project ##
As for now, I think there will be following steps:
1. **Reading and saving the text**: Given the features of Gradio, it means "reading user input and giving it to the function".
2. **Tokenization & lemmatization of the text**: These processes are needed for further lemma analysis. As I see it, 
3. **Demonstration of the text analysis**: Showing dependencies, parts of speech, named entities. As for now, I believe that *displacy* from spaCy will be enough for that.
There also can be some other steps, such as user input validation, some NLG (maybe, response generation). I will elaborate on that as soon as I get more requirements for the project.
